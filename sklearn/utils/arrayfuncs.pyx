"""A small collection of auxiliary functions that operate on arrays."""

from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport fabs
from libc.float cimport DBL_MAX, FLT_MAX

import numpy as np

from sklearn.utils._cython_blas cimport _copy, _rotg, _rot
from sklearn.utils._typedefs cimport intp_t


ctypedef fused real_numeric:
    short
    int
    long
    long long
    float
    double


# Side of the square tiles copied by _transpose_copy: the tile of the input and
# the tile of the output fit in the L1 cache.
cdef intp_t TILE_SIZE = 32

# Below _PARALLEL_MIN_SIZE values, _asfortranarray and _ascontiguousarray fall
# back to the single-threaded numpy conversion: starting the threads costs more
# than they save. Above, they use at most one thread per _MIN_SIZE_PER_THREAD
# values.
_PARALLEL_MIN_SIZE = 500_000
_MIN_SIZE_PER_THREAD = 50_000

# Rows per chunk of the column sums of _mean_variance: the sums don't depend on
# the number of threads, and the float64 sums of all the chunks take less than 2%
# of the memory of X.
cdef intp_t MEAN_VARIANCE_CHUNK_SIZE = 256


def min_pos(const floating[:] X):
    """Find the minimum value of an array over positive values.

    Returns the maximum representable value of the input dtype if none of the
    values are positive.

    Parameters
    ----------
    X : ndarray of shape (n,)
        Input array.

    Returns
    -------
    min_val : float
        The smallest positive value in the array, or the maximum representable value
         of the input dtype if no positive values are found.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.arrayfuncs import min_pos
    >>> X = np.array([0, -1, 2, 3, -4, 5])
    >>> min_pos(X)
    2.0
    """
    cdef Py_ssize_t i
    cdef floating min_val = FLT_MAX if floating is float else DBL_MAX
    for i in range(X.size):
        if 0. < X[i] < min_val:
            min_val = X[i]
    return min_val


def _all_with_any_reduction_axis_1(real_numeric[:, :] array, real_numeric value):
    """Check whether any row contains all values equal to `value`.

    It is equivalent to `np.any(np.all(X == value, axis=1))`, but it avoids to
    materialize the temporary boolean matrices in memory.

    Parameters
    ----------
    array: array-like
        The array to be checked.
    value: short, int, long, float, or double
        The value to use for the comparison.

    Returns
    -------
    any_all_equal: bool
        Whether or not any rows contains all values equal to `value`.
    """
    cdef Py_ssize_t i, j

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != value:
                break
        else:  # no break
            return True
    return False


# General Cholesky Delete.
# Remove an element from the cholesky factorization
# m = columns
# n = rows
#
# TODO: put transpose as an option
def cholesky_delete(floating[:, :] L, int go_out):
    cdef:
        int n = L.shape[0]
        int m = L.strides[0]
        floating c, s
        floating *L1
        int i

    if floating is float:
        m /= sizeof(float)
    else:
        m /= sizeof(double)

    # delete row go_out
    L1 = &L[0, 0] + (go_out * m)
    for i in range(go_out, n-1):
        _copy(i + 2, L1 + m, 1, L1, 1)
        L1 += m

    L1 = &L[0, 0] + (go_out * m)
    for i in range(go_out, n-1):
        _rotg(L1 + i, L1 + i + 1, &c, &s)
        if L1[i] < 0:
            # Diagonals cannot be negative
            L1[i] = fabs(L1[i])
            c = -c
            s = -s

        L1[i + 1] = 0.  # just for cleanup
        L1 += m

        _rot(n - i - 2, L1 + i, m, L1 + i + 1, m, c, s)


def _asfortranarray(X, int n_threads):
    """Return the 2D array X in Fortran order, using n_threads threads.

    Same result as `np.asfortranarray(X)`, whose copy is single-threaded. X is
    returned without copy if it is already in Fortran order.
    """
    n_threads = _transpose_copy_n_threads(X, X.flags.c_contiguous, n_threads)
    if n_threads < 2:
        return np.asfortranarray(X)
    X_out = np.empty(X.shape, dtype=X.dtype, order="F")
    # X_out.T in C order is the transpose of X in C order
    _transpose_copy(X, X_out.T, n_threads)
    return X_out


def _ascontiguousarray(X, int n_threads):
    """Return the 2D array X in C order, using n_threads threads.

    Same result as `np.ascontiguousarray(X)`, whose copy is single-threaded. X
    is returned without copy if it is already in C order.
    """
    n_threads = _transpose_copy_n_threads(X, X.flags.f_contiguous, n_threads)
    if n_threads < 2:
        return np.ascontiguousarray(X)
    X_out = np.empty(X.shape, dtype=X.dtype, order="C")
    # X_out in C order is the transpose of X.T in C order
    _transpose_copy(X.T, X_out, n_threads)
    return X_out


def _transpose_copy_n_threads(X, in_other_order, int n_threads):
    """Number of threads to convert X to the other memory order, 0 for numpy.

    Small arrays, arrays in neither order (strided views) and other dtypes are
    left to numpy. So are arrays in both orders (one row or one column),
    returned without copy.
    """
    if not (
        X.size >= _PARALLEL_MIN_SIZE
        and X.ndim == 2
        and X.dtype in (np.float32, np.float64)
        and in_other_order
        and not (X.flags.c_contiguous and X.flags.f_contiguous)
    ):
        return 0
    return min(n_threads, X.size // _MIN_SIZE_PER_THREAD)


def _transpose_copy(
    const floating[:, ::1] X,  # IN
    floating[:, ::1] X_out,    # OUT
    int n_threads,
):
    """X_out = X.T, by square tiles of TILE_SIZE x TILE_SIZE values."""
    cdef:
        intp_t n_rows = X.shape[0]
        intp_t n_cols = X.shape[1]
        intp_t n_tile_rows = (n_rows + TILE_SIZE - 1) // TILE_SIZE
        intp_t n_tile_cols = (n_cols + TILE_SIZE - 1) // TILE_SIZE
        intp_t tile, row_start, row_end, col_start, col_end, row, col

    # Consecutive tiles are along the rows of X: each thread reads a contiguous
    # band of X
    n_threads = min(n_threads, n_tile_rows * n_tile_cols)
    for tile in prange(n_tile_rows * n_tile_cols, nogil=True,
                       num_threads=n_threads, schedule='static'):
        row_start = (tile // n_tile_cols) * TILE_SIZE
        row_end = min(row_start + TILE_SIZE, n_rows)
        col_start = (tile % n_tile_cols) * TILE_SIZE
        col_end = min(col_start + TILE_SIZE, n_cols)
        for col in range(col_start, col_end):
            for row in range(row_start, row_end):
                X_out[col, row] = X[row, col]


def _n_threads_for_size(X, int n_threads):
    """Number of threads for a pass over the values of X, at least 1."""
    if X.size < _PARALLEL_MIN_SIZE:
        return 1
    return max(1, min(n_threads, X.size // _MIN_SIZE_PER_THREAD))


def _mean_variance(X, int n_threads):
    """Return the mean and the variance of the columns of X in float64.

    Same result as `X.mean(axis=0)` and `X.var(axis=0)` computed in float64,
    up to rounding, in a single pass over X with n_threads threads. The sums
    are accumulated by fixed chunks of rows, so that the result doesn't depend
    on n_threads, and on the deviations from the first row of X, so that the
    variance doesn't cancel out when X is far from the origin.

    X must be a C-contiguous 2D array of float32 or float64, with at least one
    row.
    """
    n_threads = _n_threads_for_size(X, n_threads)
    n_samples = X.shape[0]
    n_chunks = (n_samples + MEAN_VARIANCE_CHUNK_SIZE - 1) // MEAN_VARIANCE_CHUNK_SIZE
    chunk_sums = np.empty((n_chunks, X.shape[1]), dtype=np.float64)
    chunk_sq_sums = np.empty((n_chunks, X.shape[1]), dtype=np.float64)
    _chunk_deviation_sums(X, chunk_sums, chunk_sq_sums, n_threads)

    deviation_mean = chunk_sums.sum(axis=0) / n_samples
    mean = X[0].astype(np.float64) + deviation_mean
    # Can be slightly negative because of rounding errors for constant columns
    var = np.maximum(chunk_sq_sums.sum(axis=0) / n_samples - deviation_mean**2, 0)
    return mean, var


def _center(X, mean, int n_threads, bint copy):
    """Return X - mean, using n_threads threads.

    Same result as `X - mean` if `copy` is True, or `X -= mean` if `copy` is
    False (X is then modified and returned).

    X must be a C-contiguous 2D array of float32 or float64, and mean a 1D
    array of the same dtype.
    """
    n_threads = _n_threads_for_size(X, n_threads)
    # A copy is first written by the threads rather than by the main thread: its
    # memory is spread over their NUMA nodes
    X_out = np.empty_like(X) if copy else X
    _subtract_row(X, mean, X_out, n_threads)
    return X_out


def _chunk_deviation_sums(
    const floating[:, ::1] X,  # IN
    double[:, ::1] chunk_sums,  # OUT
    double[:, ::1] chunk_sq_sums,  # OUT
    int n_threads,
):
    """Sums of X - X[0] and of its square over each chunk of MEAN_VARIANCE_CHUNK_SIZE rows.

    chunk_sums[c] = (X[rows] - X[0]).sum(axis=0)
    chunk_sq_sums[c] = ((X[rows] - X[0]) ** 2).sum(axis=0)
    with rows = c * MEAN_VARIANCE_CHUNK_SIZE, ..., (c + 1) * MEAN_VARIANCE_CHUNK_SIZE - 1
    """
    cdef:
        intp_t n_rows = X.shape[0]
        intp_t n_cols = X.shape[1]
        intp_t chunk, row, col
        double deviation

    for chunk in prange(chunk_sums.shape[0], nogil=True, num_threads=n_threads,
                        schedule='static'):
        for col in range(n_cols):
            chunk_sums[chunk, col] = 0
            chunk_sq_sums[chunk, col] = 0
        for row in range(chunk * MEAN_VARIANCE_CHUNK_SIZE,
                         min((chunk + 1) * MEAN_VARIANCE_CHUNK_SIZE, n_rows)):
            for col in range(n_cols):
                deviation = <double> X[row, col] - X[0, col]
                chunk_sums[chunk, col] += deviation
                chunk_sq_sums[chunk, col] += deviation * deviation


def _subtract_row(
    const floating[:, ::1] X,  # IN
    const floating[::1] values,  # IN
    floating[:, ::1] X_out,  # OUT, can be X
    int n_threads,
):
    """X_out = X - values"""
    cdef:
        intp_t n_rows = X.shape[0]
        intp_t n_cols = X.shape[1]
        intp_t row, col

    for row in prange(n_rows, nogil=True, num_threads=n_threads,
                      schedule='static'):
        for col in range(n_cols):
            X_out[row, col] = X[row, col] - values[col]
