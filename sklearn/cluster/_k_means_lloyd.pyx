# Licence: BSD 3 clause

from cython cimport floating
from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport malloc, calloc, free
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn.utils.extmath import row_norms
from sklearn.utils._cython_blas cimport _gemm
from sklearn.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_dense
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_sparse
from sklearn.cluster._k_means_common cimport _average_centers, _center_shift


cdef int _chunk_size_dense(int n_samples, int n_clusters, int n_threads):
    """Number of samples per data chunk for dense input.

    Each chunk does a BLAS call. With many threads, use bigger chunks to make
    fewer concurrent BLAS calls: they scale badly with the number of threads (at
    least with OpenBLAS, which takes a global lock in each call). With few threads,
    small chunks balance the work better. The buffer of pairwise distances of
    shape (chunk_size, n_clusters) of each thread should fit in cache, and there
    should be at least one chunk per thread.
    """
    cdef int chunk_size = CHUNK_SIZE * max(1, n_threads // 16)
    chunk_size = min(chunk_size, max(CHUNK_SIZE, (1 << 18) // n_clusters))
    chunk_size = min(
        chunk_size, max(CHUNK_SIZE, (n_samples + n_threads - 1) // n_threads))
    return min(chunk_size, n_samples)


cdef void _reduce_thread_buffers(
        floating **centers_new_chunks,         # IN
        floating **weight_in_clusters_chunks,  # IN
        int n_threads,
        floating[:, ::1] centers_new,          # INOUT
        floating[::1] weight_in_clusters,      # INOUT
        bint update_centers) noexcept nogil:
    """Add the buffers of each thread to the result and free them."""
    cdef int thread_idx, j, k
    cdef int n_clusters, n_features
    for thread_idx in range(n_threads):
        # A thread of the team may not have run, e.g. if the OpenMP runtime gave
        # fewer threads than requested.
        if centers_new_chunks[thread_idx] != NULL and update_centers:
            n_clusters = centers_new.shape[0]
            n_features = centers_new.shape[1]
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunks[thread_idx][j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunks[thread_idx][j * n_features + k]
        free(centers_new_chunks[thread_idx])
        free(weight_in_clusters_chunks[thread_idx])
    free(centers_new_chunks)
    free(weight_in_clusters_chunks)


def lloyd_iter_chunked_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outliers.
        return

    cdef:
        int n_samples_chunk = _chunk_size_dense(n_samples, n_clusters, n_threads)
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        int thread_idx
        floating *centers_new_chunk
        floating *weight_in_clusters_chunk
        floating *pairwise_distances_chunk
        floating **centers_new_chunks
        floating **weight_in_clusters_chunks

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    # Each thread allocates its own buffers, and they are reduced after the
    # parallel region instead of under a lock inside it: threads waiting for each
    # other inside the parallel region scale badly with the number of threads.
    centers_new_chunks = <floating **> calloc(n_threads, sizeof(floating *))
    weight_in_clusters_chunks = <floating **> calloc(n_threads, sizeof(floating *))

    with nogil, parallel(num_threads=n_threads):
        thread_idx = threadid()
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))
        centers_new_chunks[thread_idx] = centers_new_chunk
        weight_in_clusters_chunks[thread_idx] = weight_in_clusters_chunk

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            # Pass the chunk as indices: slicing memoryviews here would make
            # Cython take the GIL in every thread of the parallel region.
            _update_chunk_dense(
                X,
                sample_weight,
                centers_old,
                centers_squared_norms,
                labels,
                start,
                end,
                centers_new_chunk,
                weight_in_clusters_chunk,
                pairwise_distances_chunk,
                update_centers)

        free(pairwise_distances_chunk)

    if update_centers:
        centers_new[...] = 0
        weight_in_clusters[...] = 0
    _reduce_thread_buffers(centers_new_chunks, weight_in_clusters_chunks, n_threads,
                           centers_new, weight_in_clusters, update_centers)

    if update_centers:
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_dense(
        const floating[:, ::1] X,                   # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        int start,
        int end,
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        floating *pairwise_distances,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of the data chunk X[start:end] to the
    labels and centers.
    """
    cdef:
        int n_samples = end - start
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, &X[start, 0], n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[start + i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[start + i]
            for k in range(n_features):
                centers_new[label * n_features + k] += (
                    X[start + i, k] * sample_weight[start + i]
                )


def lloyd_iter_chunked_sparse(
        X,                                   # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outliers.
        return

    cdef:
        # Choose same as for dense. Does not have the same impact since with
        # sparse data the pairwise distances matrix is not precomputed.
        # However, splitting in chunks is necessary to get parallelism.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start = 0, end = 0

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        int thread_idx
        floating *centers_new_chunk
        floating *weight_in_clusters_chunk
        floating **centers_new_chunks
        floating **weight_in_clusters_chunks

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    # Thread local buffers, see lloyd_iter_chunked_dense.
    centers_new_chunks = <floating **> calloc(n_threads, sizeof(floating *))
    weight_in_clusters_chunks = <floating **> calloc(n_threads, sizeof(floating *))

    with nogil, parallel(num_threads=n_threads):
        thread_idx = threadid()
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        centers_new_chunks[thread_idx] = centers_new_chunk
        weight_in_clusters_chunks[thread_idx] = weight_in_clusters_chunk

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_sparse(
                X_data,
                X_indices,
                X_indptr,
                sample_weight,
                centers_old,
                centers_squared_norms,
                labels,
                start,
                end,
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

    if update_centers:
        centers_new[...] = 0
        weight_in_clusters[...] = 0
    _reduce_thread_buffers(centers_new_chunks, weight_in_clusters_chunks, n_threads,
                           centers_new, weight_in_clusters, update_centers)

    if update_centers:
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_sparse(
        const floating[::1] X_data,                 # IN
        const int[::1] X_indices,                   # IN
        const int[::1] X_indptr,                    # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        int start,
        int end,
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of the data chunk X[start:end] to the
    labels and centers.
    """
    cdef:
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(start, end):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i], X_indptr[i + 1]):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i], X_indptr[i + 1]):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
