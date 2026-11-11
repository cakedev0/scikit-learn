# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""One e-means step on a mini-batch.

A step has two parallel phases:

1. Assignment: the batch is cut in chunks of rows. Each chunk is gathered into
   a contiguous buffer and assigned to its nearest center with one GEMM, as in
   the Lloyd kernel.
2. Update: the batch rows are counting-sorted by label, then each center
   computes the weighted sum of its rows and applies the EMA update. The
   parallelism is over centers, so no thread-local (n_clusters, n_features)
   buffer is needed and the result doesn't depend on the number of threads.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn.utils._cython_blas cimport _gemm
from sklearn.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn.utils._typedefs cimport intp_t


# Number of batch rows assigned by one GEMM call, same as in the Lloyd kernel.
cdef intp_t CHUNK_SIZE = 256


def _emeans_step_dense(
    const floating[:, ::1] X,              # IN
    const floating[::1] sample_weight,     # IN
    const intp_t[::1] batch_indices,       # IN
    floating[:, ::1] centers,              # INOUT
    double[::1] weight_ema,                # INOUT
    double step_size,
    double[:, ::1] epoch_sums,             # INOUT, or None
    double[::1] epoch_weights,             # INOUT, or None
    int n_threads,
):
    """One e-means step on the dense rows `X[batch_indices]`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    batch_indices : ndarray of shape (batch_size,), dtype=intp
        Indices of the rows of the batch.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        EMA centers, updated inplace.

    weight_ema : ndarray of shape (n_clusters,), dtype=float64
        EMA of the weight assigned to each center, updated inplace.

    step_size : float
        EMA step alpha of this batch, in (0, 1].

    epoch_sums : ndarray of shape (n_clusters, n_features), dtype=float64
        If not None, the weighted sums of the batch rows assigned to each
        center are added to it.

    epoch_weights : ndarray of shape (n_clusters,), dtype=float64
        If not None, the weight of the batch rows assigned to each center is
        added to it.

    n_threads : int
        The number of threads to be used by openmp. All the parallel regions
        use this exact number of threads, since changing the size of the
        OpenMP team between consecutive regions is slow.

    Returns
    -------
    inertia : float
        Inertia of the batch, measured against the centers before the update.
    """
    cdef:
        intp_t n_batch = batch_indices.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_batch + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, j
        double *epoch_sums_ptr = NULL
        double *epoch_weights_ptr = NULL
        int *labels
        intp_t *cluster_starts
        intp_t *batch_order
        double *chunk_inertia
        floating *centers_squared_norms
        floating *X_chunk = NULL
        floating *distances = NULL
        double *sums = NULL
        double inertia = 0.0

    if n_batch == 0:
        return 0.0

    if epoch_sums is not None:
        epoch_sums_ptr = &epoch_sums[0, 0]
        epoch_weights_ptr = &epoch_weights[0]

    labels = <int*> malloc(n_batch * sizeof(int))
    cluster_starts = <intp_t*> malloc((n_clusters + 1) * sizeof(intp_t))
    batch_order = <intp_t*> malloc(n_batch * sizeof(intp_t))
    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))

    with nogil:
        for j in prange(n_clusters, num_threads=n_threads, schedule='static'):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        # Phase 1: assignment, parallel over chunks of batch rows.
        with parallel(num_threads=n_threads):
            X_chunk = <floating*> malloc(CHUNK_SIZE * n_features * sizeof(floating))
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))

            for chunk_idx in prange(n_chunks, schedule='dynamic'):
                start = chunk_idx * CHUNK_SIZE
                chunk_inertia[chunk_idx] = _assign_chunk_dense(
                    &X[0, 0],
                    &sample_weight[0],
                    &batch_indices[start],
                    min(CHUNK_SIZE, n_batch - start),
                    &centers[0, 0],
                    centers_squared_norms,
                    n_clusters,
                    n_features,
                    labels + start,
                    X_chunk,
                    distances,
                )

            free(X_chunk)
            free(distances)

        _sort_by_label(&batch_indices[0], labels, n_batch, n_clusters,
                       cluster_starts, batch_order)

        # Phase 2: EMA update, parallel over centers.
        with parallel(num_threads=n_threads):
            sums = <double*> malloc(n_features * sizeof(double))

            for j in prange(n_clusters, schedule='guided'):
                _update_center_dense(
                    &X[0, 0],
                    &sample_weight[0],
                    batch_order + cluster_starts[j],
                    cluster_starts[j + 1] - cluster_starts[j],
                    n_features,
                    &centers[j, 0],
                    &weight_ema[j],
                    step_size,
                    NULL if epoch_sums_ptr == NULL
                    else epoch_sums_ptr + j * n_features,
                    NULL if epoch_weights_ptr == NULL else epoch_weights_ptr + j,
                    sums,
                )

            free(sums)

    for chunk_idx in range(n_chunks):
        inertia += chunk_inertia[chunk_idx]

    free(labels)
    free(cluster_starts)
    free(batch_order)
    free(chunk_inertia)
    free(centers_squared_norms)

    return inertia


def _emeans_step_sparse(
    X,                                     # IN
    const floating[::1] sample_weight,     # IN
    const intp_t[::1] batch_indices,       # IN
    floating[:, ::1] centers,              # INOUT
    double[::1] weight_ema,                # INOUT
    double step_size,
    double[:, ::1] epoch_sums,             # INOUT, or None
    double[::1] epoch_weights,             # INOUT, or None
    int n_threads,
):
    """One e-means step on the sparse rows `X[batch_indices]`.

    Same as `_emeans_step_dense`, with `X` a CSR matrix.
    """
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr
        intp_t n_batch = batch_indices.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_batch + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, j
        double *epoch_sums_ptr = NULL
        double *epoch_weights_ptr = NULL
        int *labels
        intp_t *cluster_starts
        intp_t *batch_order
        double *chunk_inertia
        floating *centers_squared_norms
        double *sums = NULL
        double inertia = 0.0

    if n_batch == 0:
        return 0.0

    if epoch_sums is not None:
        epoch_sums_ptr = &epoch_sums[0, 0]
        epoch_weights_ptr = &epoch_weights[0]

    labels = <int*> malloc(n_batch * sizeof(int))
    cluster_starts = <intp_t*> malloc((n_clusters + 1) * sizeof(intp_t))
    batch_order = <intp_t*> malloc(n_batch * sizeof(intp_t))
    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))

    with nogil:
        for j in prange(n_clusters, num_threads=n_threads, schedule='static'):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        # Phase 1: assignment, parallel over chunks of batch rows.
        for chunk_idx in prange(n_chunks, num_threads=n_threads, schedule='dynamic'):
            start = chunk_idx * CHUNK_SIZE
            chunk_inertia[chunk_idx] = _assign_chunk_sparse(
                &X_data[0],
                &X_indices[0],
                &X_indptr[0],
                &sample_weight[0],
                &batch_indices[start],
                min(CHUNK_SIZE, n_batch - start),
                &centers[0, 0],
                centers_squared_norms,
                n_clusters,
                n_features,
                labels + start,
            )

        _sort_by_label(&batch_indices[0], labels, n_batch, n_clusters,
                       cluster_starts, batch_order)

        # Phase 2: EMA update, parallel over centers.
        with parallel(num_threads=n_threads):
            sums = <double*> malloc(n_features * sizeof(double))

            for j in prange(n_clusters, schedule='guided'):
                _update_center_sparse(
                    &X_data[0],
                    &X_indices[0],
                    &X_indptr[0],
                    &sample_weight[0],
                    batch_order + cluster_starts[j],
                    cluster_starts[j + 1] - cluster_starts[j],
                    n_features,
                    &centers[j, 0],
                    &weight_ema[j],
                    step_size,
                    NULL if epoch_sums_ptr == NULL
                    else epoch_sums_ptr + j * n_features,
                    NULL if epoch_weights_ptr == NULL else epoch_weights_ptr + j,
                    sums,
                )

            free(sums)

    for chunk_idx in range(n_chunks):
        inertia += chunk_inertia[chunk_idx]

    free(labels)
    free(cluster_starts)
    free(batch_order)
    free(chunk_inertia)
    free(centers_squared_norms)

    return inertia


cdef inline floating _squared_norm(
    const floating *x,
    intp_t n_features,
) noexcept nogil:
    cdef:
        floating result = 0.0
        intp_t k

    for k in range(n_features):
        result += x[k] * x[k]
    return result


cdef double _assign_chunk_dense(
    const floating *X,                      # IN (n_samples, n_features)
    const floating *sample_weight,          # IN (n_samples,)
    const intp_t *rows,                     # IN (n_rows,)
    intp_t n_rows,
    const floating *centers,                # IN (n_clusters, n_features)
    const floating *centers_squared_norms,  # IN (n_clusters,)
    intp_t n_clusters,
    intp_t n_features,
    int *labels,                            # OUT (n_rows,)
    floating *X_chunk,                      # TMP (n_rows, n_features)
    floating *distances,                    # TMP (n_rows, n_clusters)
) noexcept nogil:
    """Assign `X[rows]` to their nearest center and return their inertia."""
    cdef:
        intp_t r, j, label
        floating dist, min_dist, x_squared_norm
        double inertia = 0.0

    for r in range(n_rows):
        memcpy(&X_chunk[r * n_features], &X[rows[r] * n_features],
               n_features * sizeof(floating))
        for j in range(n_clusters):
            distances[r * n_clusters + j] = centers_squared_norms[j]

    # ||x - c||² = ||x||² - 2 x.c + ||c||², and the argmin over c only needs
    # distances = ||c||² - 2 x.c
    _gemm(RowMajor, NoTrans, Trans, <int> n_rows, <int> n_clusters,
          <int> n_features, -2.0, X_chunk, <int> n_features, centers,
          <int> n_features, 1.0, distances, <int> n_clusters)

    for r in range(n_rows):
        min_dist = distances[r * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            dist = distances[r * n_clusters + j]
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[r] = <int> label
        # ||x||² of the gathered row, while it is in cache: computing it here
        # avoids a full pass over X before the fit.
        x_squared_norm = _squared_norm(&X_chunk[r * n_features], n_features)
        inertia += sample_weight[rows[r]] * max(x_squared_norm + min_dist, 0.0)

    return inertia


cdef double _assign_chunk_sparse(
    const floating *X_data,                 # IN (nnz,)
    const int *X_indices,                   # IN (nnz,)
    const int *X_indptr,                    # IN (n_samples + 1,)
    const floating *sample_weight,          # IN (n_samples,)
    const intp_t *rows,                     # IN (n_rows,)
    intp_t n_rows,
    const floating *centers,                # IN (n_clusters, n_features)
    const floating *centers_squared_norms,  # IN (n_clusters,)
    intp_t n_clusters,
    intp_t n_features,
    int *labels,                            # OUT (n_rows,)
) noexcept nogil:
    """Assign `X[rows]` to their nearest center and return their inertia."""
    cdef:
        intp_t r, i, j, k, label
        floating dot, dist, min_dist, x_squared_norm
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        double inertia = 0.0

    for r in range(n_rows):
        i = rows[r]
        min_dist = max_floating
        label = 0
        for j in range(n_clusters):
            dot = 0.0
            for k in range(X_indptr[i], X_indptr[i + 1]):
                dot = dot + centers[j * n_features + X_indices[k]] * X_data[k]
            # distance up to the ||x||² term, as in the dense case
            dist = centers_squared_norms[j] - 2 * dot
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[r] = <int> label
        x_squared_norm = 0.0
        for k in range(X_indptr[i], X_indptr[i + 1]):
            x_squared_norm = x_squared_norm + X_data[k] * X_data[k]
        inertia += sample_weight[i] * max(x_squared_norm + min_dist, 0.0)

    return inertia


cdef void _sort_by_label(
    const intp_t *batch_indices,  # IN (n_batch,)
    const int *labels,            # IN (n_batch,)
    intp_t n_batch,
    intp_t n_clusters,
    intp_t *cluster_starts,       # OUT (n_clusters + 1,)
    intp_t *batch_order,          # OUT (n_batch,)
) noexcept nogil:
    """Counting sort of the batch rows by label.

    The rows assigned to center j are
    `batch_order[cluster_starts[j]:cluster_starts[j + 1]]`, in batch order.
    """
    cdef intp_t p, j

    memset(cluster_starts, 0, (n_clusters + 1) * sizeof(intp_t))
    for p in range(n_batch):
        cluster_starts[labels[p] + 1] += 1
    for j in range(n_clusters):
        cluster_starts[j + 1] += cluster_starts[j]

    # Fill each segment, using cluster_starts[j] as the write cursor of center
    # j. It ends at the start of center j + 1, so shift it back afterwards.
    for p in range(n_batch):
        batch_order[cluster_starts[labels[p]]] = batch_indices[p]
        cluster_starts[labels[p]] += 1
    for j in range(n_clusters, 0, -1):
        cluster_starts[j] = cluster_starts[j - 1]
    cluster_starts[0] = 0


cdef void _update_center_dense(
    const floating *X,              # IN (n_samples, n_features)
    const floating *sample_weight,  # IN (n_samples,)
    const intp_t *members,          # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    floating *center,               # INOUT (n_features,)
    double *weight_ema,             # INOUT
    double step_size,
    double *epoch_sum,              # INOUT (n_features,), or NULL
    double *epoch_weight,           # INOUT, or NULL
    double *sums,                   # TMP (n_features,)
) noexcept nogil:
    cdef:
        intp_t m, k
        const floating *x
        double w, weight = 0.0

    if n_members > 0:
        # s_j = sum_i w_i x_i and n_j = sum_i w_i over the rows i of center j
        memset(sums, 0, n_features * sizeof(double))
        for m in range(n_members):
            x = &X[members[m] * n_features]
            w = sample_weight[members[m]]
            weight += w
            for k in range(n_features):
                sums[k] += w * x[k]

    _ema_update(center, weight_ema, sums, weight, step_size, epoch_sum,
                epoch_weight, n_features)


cdef void _update_center_sparse(
    const floating *X_data,         # IN (nnz,)
    const int *X_indices,           # IN (nnz,)
    const int *X_indptr,            # IN (n_samples + 1,)
    const floating *sample_weight,  # IN (n_samples,)
    const intp_t *members,          # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    floating *center,               # INOUT (n_features,)
    double *weight_ema,             # INOUT
    double step_size,
    double *epoch_sum,              # INOUT (n_features,), or NULL
    double *epoch_weight,           # INOUT, or NULL
    double *sums,                   # TMP (n_features,)
) noexcept nogil:
    cdef:
        intp_t m, i, k
        double w, weight = 0.0

    if n_members > 0:
        # s_j = sum_i w_i x_i and n_j = sum_i w_i over the rows i of center j
        memset(sums, 0, n_features * sizeof(double))
        for m in range(n_members):
            i = members[m]
            w = sample_weight[i]
            weight += w
            for k in range(X_indptr[i], X_indptr[i + 1]):
                sums[X_indices[k]] += w * X_data[k]

    _ema_update(center, weight_ema, sums, weight, step_size, epoch_sum,
                epoch_weight, n_features)


cdef inline void _ema_update(
    floating *center,     # INOUT (n_features,)
    double *weight_ema,   # INOUT
    const double *sums,   # IN (n_features,)
    double weight,
    double step_size,
    double *epoch_sum,    # INOUT (n_features,), or NULL
    double *epoch_weight,  # INOUT, or NULL
    intp_t n_features,
) noexcept nogil:
    """EMA update of one center from the weighted sum of its batch rows.

    The center c_j = S_j / N_j is the ratio of two EMAs with step alpha:
        S_j <- (1 - alpha) S_j + alpha s_j
        N_j <- (1 - alpha) N_j + alpha n_j
    so with N'_j the new N_j:
        c_j <- ((1 - alpha) N_j c_j + alpha s_j) / N'_j
    """
    cdef:
        intp_t k
        double retained = (1.0 - step_size) * weight_ema[0]
        double new_weight, center_coef, sums_coef

    if weight <= 0:
        # No row (or only zero weight rows): only N_j decays, c_j stays.
        weight_ema[0] = retained
        return

    if epoch_sum != NULL:
        epoch_weight[0] += weight
        for k in range(n_features):
            epoch_sum[k] += sums[k]

    new_weight = retained + step_size * weight
    center_coef = retained / new_weight
    sums_coef = step_size / new_weight
    for k in range(n_features):
        center[k] = <floating> (center_coef * center[k] + sums_coef * sums[k])
    weight_ema[0] = new_weight
