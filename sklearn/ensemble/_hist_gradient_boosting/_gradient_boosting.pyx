# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython.parallel import prange
import numpy as np

from sklearn.ensemble._hist_gradient_boosting.common import Y_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common cimport Y_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting._team cimport team_active, team_run
from sklearn.ensemble._hist_gradient_boosting._team cimport task_fn
from sklearn.ensemble._hist_gradient_boosting.common cimport G_H_DTYPE_C
from sklearn._loss._loss cimport CyLossFunction, double_pair
from sklearn.utils._typedefs cimport intp_t
from libc.math cimport exp
from libc.stdlib cimport malloc, free

# Number of samples whose gradient is computed by one chunk of the team.
cdef intp_t GRADIENT_CHUNK_SIZE = 8192


def _update_raw_predictions(
        Y_DTYPE_C [::1] raw_predictions,  # OUT
        grower,
        n_threads,
):
    """Update raw_predictions with the predictions of the newest tree.

    This is equivalent to (and much faster than):
        raw_predictions += last_estimator.predict(X_train)

    It's only possible for data X_train that is used to train the trees (it
    isn't usable for e.g. X_val).
    """
    cdef:
        unsigned int [::1] starts  # start of each leaf in partition
        unsigned int [::1] stops  # end of each leaf in partition
        Y_DTYPE_C [::1] values  # value of each leaf
        const unsigned int [::1] partition = grower.splitter.partition
        list leaves

    leaves = grower.finalized_leaves
    starts = np.array([leaf.partition_start for leaf in leaves],
                      dtype=np.uint32)
    stops = np.array([leaf.partition_stop for leaf in leaves],
                     dtype=np.uint32)
    values = np.array([leaf.value for leaf in leaves], dtype=Y_DTYPE)

    _update_raw_predictions_helper(raw_predictions, starts, stops, partition,
                                   values, n_threads)


cdef inline void _update_raw_predictions_helper(
        Y_DTYPE_C [::1] raw_predictions,  # OUT
        const unsigned int [::1] starts,
        const unsigned int [::1] stops,
        const unsigned int [::1] partition,
        const Y_DTYPE_C [::1] values,
        int n_threads,
):

    cdef:
        unsigned int position
        int leaf_idx
        int n_leaves = starts.shape[0]
        update_task_ctx ctx

    if team_active():
        ctx.raw_predictions = &raw_predictions[0]
        ctx.starts = &starts[0]
        ctx.stops = &stops[0]
        ctx.partition = &partition[0]
        ctx.values = &values[0]
        with nogil:
            team_run(_update_task, &ctx, n_leaves, 1)
        return

    for leaf_idx in prange(n_leaves, schedule='static', nogil=True,
                           num_threads=n_threads):
        for position in range(starts[leaf_idx], stops[leaf_idx]):
            raw_predictions[partition[position]] += values[leaf_idx]


cdef struct update_task_ctx:
    Y_DTYPE_C * raw_predictions
    const unsigned int * starts
    const unsigned int * stops
    const unsigned int * partition
    const Y_DTYPE_C * values


cdef void _update_task(void * ctx_, intp_t first, intp_t last) noexcept nogil:
    cdef:
        update_task_ctx * ctx = <update_task_ctx *> ctx_
        intp_t leaf_idx
        unsigned int position

    for leaf_idx in range(first, last):
        for position in range(ctx.starts[leaf_idx], ctx.stops[leaf_idx]):
            ctx.raw_predictions[ctx.partition[position]] += ctx.values[leaf_idx]


cdef class _GradientTask:
    cdef:
        CyLossFunction closs
        const double * y_true
        const double * raw_prediction
        const double * sample_weight
        G_H_DTYPE_C * gradient
        G_H_DTYPE_C * hessian


cdef void _gradient_task(_GradientTask task, intp_t start, intp_t stop) noexcept nogil:
    cdef:
        intp_t i
        double_pair grad_hess
        double weight = 1

    if task.hessian == NULL:
        for i in range(start, stop):
            if task.sample_weight != NULL:
                weight = task.sample_weight[i]
            task.gradient[i] = weight * task.closs.cy_gradient(
                task.y_true[i], task.raw_prediction[i]
            )
    else:
        for i in range(start, stop):
            if task.sample_weight != NULL:
                weight = task.sample_weight[i]
            grad_hess = task.closs.cy_grad_hess(
                task.y_true[i], task.raw_prediction[i]
            )
            task.gradient[i] = weight * grad_hess.val1
            task.hessian[i] = weight * grad_hess.val2


def _team_gradient(
    CyLossFunction closs,
    const double[::1] y_true,
    const double[::1] raw_prediction,
    const double[::1] sample_weight,  # or None
    G_H_DTYPE_C[::1] gradient_out,
    G_H_DTYPE_C[::1] hessian_out,     # or None
):
    """Gradients (and hessians) of a single-output loss, on the team.

    Stands for `loss.gradient` and `loss.gradient_hessian`, whose OpenMP loops
    would run on one thread inside the region of the team.
    """
    cdef:
        _GradientTask task = _GradientTask.__new__(_GradientTask)
        intp_t n_samples = y_true.shape[0]

    task.closs = closs
    task.y_true = &y_true[0]
    task.raw_prediction = &raw_prediction[0]
    task.sample_weight = NULL if sample_weight is None else &sample_weight[0]
    task.gradient = &gradient_out[0]
    task.hessian = NULL if hessian_out is None else &hessian_out[0]
    with nogil:
        team_run(<task_fn> _gradient_task, <void*> task, n_samples,
                 GRADIENT_CHUNK_SIZE)


cdef struct multinomial_task_ctx:
    const double * y_true
    const double * raw_prediction   # (n_samples, n_classes), F-ordered
    const double * sample_weight
    G_H_DTYPE_C * gradient          # (n_samples, n_classes), F-ordered
    G_H_DTYPE_C * hessian
    intp_t n_samples
    int n_classes


cdef void _multinomial_task(void * ctx_, intp_t start, intp_t stop) noexcept nogil:
    # Same operations, in the same order, as
    # CyHalfMultinomialLoss.gradient_hessian.
    cdef:
        multinomial_task_ctx * ctx = <multinomial_task_ctx *> ctx_
        intp_t n = ctx.n_samples
        int n_classes = ctx.n_classes
        intp_t i
        int k
        double max_value, sum_exps
        double * p = <double *> malloc(sizeof(double) * n_classes)

    for i in range(start, stop):
        max_value = ctx.raw_prediction[i]
        sum_exps = 0
        for k in range(1, n_classes):
            if max_value < ctx.raw_prediction[k * n + i]:
                max_value = ctx.raw_prediction[k * n + i]
        for k in range(n_classes):
            p[k] = exp(ctx.raw_prediction[k * n + i] - max_value)
            sum_exps += p[k]
        for k in range(n_classes):
            p[k] /= sum_exps
            if ctx.sample_weight == NULL:
                ctx.gradient[k * n + i] = p[k] - (ctx.y_true[i] == k)
                ctx.hessian[k * n + i] = p[k] * (1. - p[k])
            else:
                ctx.gradient[k * n + i] = (
                    (p[k] - (ctx.y_true[i] == k)) * ctx.sample_weight[i]
                )
                ctx.hessian[k * n + i] = (p[k] * (1. - p[k])) * ctx.sample_weight[i]
    free(p)


def _team_gradient_multinomial(
    const double[::1] y_true,
    const double[::1, :] raw_prediction,
    const double[::1] sample_weight,  # or None
    G_H_DTYPE_C[::1, :] gradient_out,
    G_H_DTYPE_C[::1, :] hessian_out,
):
    """Gradients and hessians of the multinomial loss, on the team."""
    cdef multinomial_task_ctx ctx

    ctx.y_true = &y_true[0]
    ctx.raw_prediction = &raw_prediction[0, 0]
    ctx.sample_weight = NULL if sample_weight is None else &sample_weight[0]
    ctx.gradient = &gradient_out[0, 0]
    ctx.hessian = &hessian_out[0, 0]
    ctx.n_samples = y_true.shape[0]
    ctx.n_classes = raw_prediction.shape[1]
    with nogil:
        team_run(_multinomial_task, &ctx, ctx.n_samples, GRADIENT_CHUNK_SIZE // 4)
