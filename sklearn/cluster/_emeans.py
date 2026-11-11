"""E-means clustering."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import _fit_context
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.cluster._k_means_emeans import _emeans_step_dense, _emeans_step_sparse
from sklearn.cluster._kmeans import _BaseKMeans, _labels_inertia_threadpool_limit
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.utils.parallel import _get_threadpool_controller
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    validate_data,
)

# The constants below are the ones of the reference implementation
# (faiss/contrib/e_means/base.py), which differ slightly from the paper.

# Ratio between the last and the first step size of the cosine schedule.
_MIN_STEP_SIZE_FACTOR = 0.1

# Dead clusters are revived every `_REVIVAL_PERIOD` steps.
_REVIVAL_PERIOD = 5

# A cluster is dead when its EMA weight is below this fraction of the mean EMA
# weight.
_DEAD_CLUSTER_THRESHOLD = 0.1

# Relative perturbation of the two halves of a split donor center.
_SPLIT_EPSILON = 1e-4

# Initial EMA weight of each center, relative to the mean sample weight. Tiny,
# so that the first batch moves the centers to the mean of their rows.
_INIT_WEIGHT = 2.0**-16

# The reference implementation turns revival off for n_features <= 2, where it
# was found to hurt the held-out inertia.
_NO_REVIVAL_MAX_FEATURES = 2

# The "quality" batch law (8x smaller batches) is used for n_features <= 16,
# where more steps help more, and the "throughput" law otherwise.
_QUALITY_BATCH_MAX_FEATURES = 16

# Batch law: batch_size = c * n_clusters**a_k * n_samples**a_n * n_epochs**a_t,
# stored as (c, a_k, a_n, a_t).
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Step law: step_size = c * n_clusters**a_k * batch_size**a_b * n_samples**a_n
# * n_epochs**a_t, stored as (c, a_k, a_b, a_n, a_t).
_STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    return low if value / low < 2 * low / value else 2 * low


def _resolve_batch_size(n_samples, n_clusters, n_features, n_epochs):
    """Batch size given by the fitted batch law."""
    if n_features <= _QUALITY_BATCH_MAX_FEATURES:
        law = "quality"
        max_batch_size = n_samples // 2
    else:
        law = "throughput"
        max_batch_size = _round_power_of_two(n_samples / 2)
    c, a_k, a_n, a_t = _BATCH_LAWS[law]
    batch_size = _round_power_of_two(
        c * n_clusters**a_k * n_samples**a_n * n_epochs**a_t
    )
    return max(1, min(batch_size, max_batch_size, n_samples))


def _resolve_initial_step_size(n_samples, n_clusters, n_epochs, batch_size):
    """Initial EMA step size given by the fitted step law."""
    c, a_k, a_b, a_n, a_t = _STEP_LAW
    step_size = c * n_clusters**a_k * batch_size**a_b * n_samples**a_n * n_epochs**a_t
    return min(max(step_size, 1e-4), 1.0)


def _epochs_n_rows(n_samples, n_epochs):
    """Number of rows seen in each epoch.

    A fractional epoch only sees a prefix of its permutation. It comes before
    the last epoch, which is always complete since the final centers are the
    means of its assignments.
    """
    n_full = math.floor(n_epochs)
    remainder = round((n_epochs - n_full) * n_samples)
    if n_full == 0:
        return [max(remainder, 1)]
    return [n_samples] * (n_full - 1) + ([remainder] if remainder else []) + [n_samples]


def _scheduled_step_size(initial_step_size, step, n_steps):
    """Cosine annealing from `initial_step_size` to a tenth of it."""
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
    return initial_step_size * (
        _MIN_STEP_SIZE_FACTOR + (1.0 - _MIN_STEP_SIZE_FACTOR) * cosine
    )


def _revive_dead_clusters(centers, weight_ema, random_state):
    """Split heavy clusters to replace the dead ones, inplace.

    A cluster is dead when its EMA weight is below a fraction of the mean EMA
    weight. Each dead cluster takes half of a donor cluster, drawn without
    replacement with a probability proportional to its EMA weight. The donor
    center is split into two centers moved by a tiny relative offset in
    opposite directions, and its EMA weight is shared between them.

    Returns the number of revived clusters.
    """
    dead = weight_ema < _DEAD_CLUSTER_THRESHOLD * weight_ema.mean()
    donors = ~dead & (weight_ema > 0)
    n_revived = min(np.count_nonzero(dead), np.count_nonzero(donors))
    if n_revived == 0:
        return 0

    # Gumbel top-k trick: the n_revived largest log(w) + Gumbel noise are a
    # sample without replacement with probabilities proportional to w.
    uniform = np.maximum(random_state.uniform(size=len(weight_ema)), 1e-300)
    keys = np.full(len(weight_ema), -np.inf)
    keys[donors] = np.log(weight_ema[donors]) - np.log(-np.log(uniform[donors]))
    donor_indices = np.argsort(keys)[::-1][:n_revived]
    dead_indices = np.flatnonzero(dead)[:n_revived]

    offset = np.full(centers.shape[1], _SPLIT_EPSILON, dtype=centers.dtype)
    offset[1::2] *= -1
    donor_centers = centers[donor_indices]
    centers[dead_indices] = donor_centers * (1 + offset)
    centers[donor_indices] = donor_centers * (1 - offset)

    # A null center would give two identical halves: separate them along the
    # first feature.
    same = (centers[dead_indices] == centers[donor_indices]).all(axis=1)
    if same.any():
        same_indices = dead_indices[same]
        centers[same_indices, 0] = np.nextafter(
            centers[same_indices, 0], centers.dtype.type(np.inf)
        )

    weight_ema[donor_indices] /= 2
    weight_ema[dead_indices] = weight_ema[donor_indices]
    return n_revived


class EMeans(_BaseKMeans):
    """E-means clustering.

    E-means is a mini-batch variant of K-means where each center is an
    exponential moving average (EMA) of the samples assigned to it. The EMA
    step size decreases along a cosine schedule, and clusters that receive
    too few samples are revived by splitting heavier ones. It typically
    reaches the inertia of :class:`KMeans` in a few passes over the data
    (epochs) instead of tens of iterations.

    The batch size and the initial step size are given by scaling laws fitted
    by the authors, as functions of the number of samples, of clusters and of
    epochs, so that the number of epochs is the only parameter to choose.

    Read more in the :ref:`User Guide <emeans>`.

    .. versionadded:: 1.10

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'random', 'k-means++'}, callable or array-like of shape \
            (n_clusters, n_features), default='random'
        Method for initialization:

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. It slightly improves the final inertia, but it is much
        slower than the e-means iterations themselves when `n_clusters` is
        large.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_epochs : float, default=5
        Number of passes over the data. Fractional values are supported: when
        `n_epochs > 1`, the fractional pass comes before the last epoch, which
        is always a full pass.

    batch_size : int or 'auto', default='auto'
        Size of the mini batches. 'auto' uses the scaling law of the authors,
        which grows almost linearly with the number of samples and favors
        smaller batches when `n_features <= 16`. The initial step size is
        always given by the scaling law, as a function of the batch size.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization, the
        order of the samples in each epoch and the revival of dead clusters.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of epochs run.

    n_steps_ : int
        Number of mini-batches processed.

    batch_size_ : int
        Batch size used for fitting.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm.
    MiniBatchKMeans : Mini-batch K-means with per-center learning rates.

    Notes
    -----
    The algorithm is described in [1]_. For each mini-batch of samples, each
    sample is assigned to its nearest center, then each center :math:`c_j` is
    updated from the sum :math:`s_j` and the weight :math:`n_j` of the samples
    assigned to it:

    .. math::

        N_j \\leftarrow (1 - \\alpha) N_j + \\alpha n_j, \\qquad
        c_j \\leftarrow \\frac{(1 - \\alpha) N_j c_j + \\alpha s_j}{N_j'}

    where :math:`N_j'` is the updated :math:`N_j` and :math:`\\alpha` the step
    size. The final centers are the means of the samples assigned to each
    center during the last epoch. `labels_` and `inertia_` are then computed
    with an extra pass over the data, so that they match the final centers.

    References
    ----------
    .. [1] S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard,
       M. Douze and H. Jégou, "e-means: outperforming k-means (in one-tenth
       the epochs)", 2026. :arxiv:`2611.09501`

    Examples
    --------
    >>> from sklearn.cluster import EMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> emeans = EMeans(n_clusters=2, random_state=0).fit(X)
    >>> emeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> emeans.predict([[0, 0], [12, 3]])
    array([0, 1], dtype=int32)
    >>> emeans.cluster_centers_
    array([[ 1.,  2.],
           [10.,  2.]])
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_epochs": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="random",
        n_epochs=5,
        batch_size="auto",
        verbose=0,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_input(self, X):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "EMeans is known to have a memory leak on Windows with MKL, when "
            "there are less chunks than available threads. You can prevent it "
            f"by setting batch_size >= {self._n_threads * CHUNK_SIZE} or by "
            "setting the environment variable "
            f"OMP_NUM_THREADS={n_active_threads}"
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute e-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        x_squared_norms = row_norms(X, squared=True)
        centers = self._init_centroids(
            X,
            x_squared_norms=x_squared_norms,
            init=init,
            random_state=random_state,
            sample_weight=sample_weight,
            n_threads=self._n_threads,
        )
        centers = np.array(centers, dtype=X.dtype, order="C", copy=True)

        if self.batch_size == "auto":
            batch_size = _resolve_batch_size(
                n_samples, n_clusters, n_features, self.n_epochs
            )
        else:
            batch_size = min(self.batch_size, n_samples)
        initial_step_size = _resolve_initial_step_size(
            n_samples, n_clusters, self.n_epochs, batch_size
        )
        self._check_mkl_vcomp(X, batch_size)
        # No more threads than chunks of a batch: small batches are faster on
        # fewer threads.
        n_threads = min(self._n_threads, math.ceil(batch_size / CHUNK_SIZE))

        epochs_n_rows = _epochs_n_rows(n_samples, self.n_epochs)
        n_steps = sum(math.ceil(n_rows / batch_size) for n_rows in epochs_n_rows)
        revival = n_features > _NO_REVIVAL_MAX_FEATURES

        # Scaled by the mean sample weight, so that the result doesn't depend
        # on the scale of the weights.
        weight_ema = np.full(
            n_clusters, _INIT_WEIGHT * sample_weight.mean(), dtype=np.float64
        )
        epoch_sums = np.zeros((n_clusters, n_features), dtype=np.float64)
        epoch_weights = np.zeros(n_clusters, dtype=np.float64)
        emeans_step = _emeans_step_sparse if sp.issparse(X) else _emeans_step_dense

        if self.verbose:
            print(
                f"Clustering {n_samples} samples into {n_clusters} clusters in "
                f"{self.n_epochs} epochs, batch_size={batch_size}, initial "
                f"step size={initial_step_size:.3g}"
            )

        step = 0
        with _get_threadpool_controller().limit(limits="sequential_blas_under_openmp"):
            for epoch, n_rows in enumerate(epochs_n_rows):
                is_last_epoch = epoch == len(epochs_n_rows) - 1
                permutation = random_state.permutation(n_samples)
                epoch_inertia, n_revived = 0.0, 0
                for start in range(0, n_rows, batch_size):
                    epoch_inertia += emeans_step(
                        X,
                        sample_weight,
                        x_squared_norms,
                        permutation[start : min(start + batch_size, n_rows)],
                        centers,
                        weight_ema,
                        _scheduled_step_size(initial_step_size, step, n_steps),
                        epoch_sums if is_last_epoch else None,
                        epoch_weights if is_last_epoch else None,
                        n_threads,
                    )
                    step += 1
                    if revival and step % _REVIVAL_PERIOD == 0:
                        n_revived += _revive_dead_clusters(
                            centers, weight_ema, random_state
                        )
                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}/{len(epochs_n_rows)}: inertia "
                        f"{epoch_inertia} (measured on moving centers), "
                        f"{n_revived} revived clusters"
                    )

        # The final centers are the means of the last epoch's assignments.
        # Centers that got no sample in the last epoch keep their EMA position.
        assigned = epoch_weights > 0
        centers[assigned] = epoch_sums[assigned] / epoch_weights[assigned, None]

        self.cluster_centers_ = centers
        self._n_features_out = n_clusters
        self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
            X, sample_weight, centers, n_threads=self._n_threads
        )
        self.n_iter_ = len(epochs_n_rows)
        self.n_steps_ = n_steps
        self.batch_size_ = batch_size
        return self
