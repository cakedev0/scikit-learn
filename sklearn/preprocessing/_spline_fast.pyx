# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.math cimport isnan
from libc.stdlib cimport free, malloc

from sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t


# X and the output can have different dtypes: the periodic mapping of float32 data
# is computed in float64.
ctypedef fused X_DTYPE:
    float32_t
    float64_t


cdef inline intp_t _find_span(
    const float64_t* t, intp_t k, intp_t n_basis, float64_t x
) noexcept nogil:
    """Return the largest i in [k, n_basis - 1] such that t[i] <= x.

    Values below t[k] get i = k, so that they are evaluated with the polynomial
    of the first interval, like scipy's BSpline with extrapolate=True.
    """
    cdef intp_t lo = k, hi = n_basis - 1, mid
    if x >= t[hi]:
        return hi
    # Invariant: x < t[hi] and (t[lo] <= x or lo == k).
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


cdef inline intp_t _nonzero_basis(
    const float64_t* t,
    intp_t k,
    intp_t n_basis,
    float64_t x,
    bint clamp,
    const float64_t* slopes_lo,
    const float64_t* slopes_hi,
    float64_t* values,
) noexcept nogil:
    """Compute the k + 1 B-splines of degree k that can be non-zero at x.

    They are stored in `values` and the index of the first one is returned.

    If `clamp` is True, x is clipped to [t[k], t[n_basis]] and the B-splines are
    continued linearly beyond with the slopes `slopes_lo` and `slopes_hi`, which
    are the derivatives of all B-splines at t[k] and t[n_basis].
    """
    cdef:
        intp_t i, j, r
        float64_t dx = 0.0, saved, tmp, t_left, t_right
        const float64_t* slopes = slopes_lo

    if clamp:
        if x < t[k]:
            dx = x - t[k]
            x = t[k]
        elif x > t[n_basis]:
            dx = x - t[n_basis]
            x = t[n_basis]
            slopes = slopes_hi

    i = _find_span(t, k, n_basis, x)

    # Cox-de Boor recursion, with 0 / 0 := 0 for repeated knots. A constant
    # feature, i.e. an empty base interval, is encoded as zeros.
    values[0] = 1.0 if t[k] < t[n_basis] else 0.0
    for j in range(1, k + 1):
        saved = 0.0
        for r in range(j):
            t_left = t[i + r + 1 - j]
            t_right = t[i + r + 1]
            tmp = values[r] / (t_right - t_left) if t_right > t_left else 0.0
            values[r] = saved + (t_right - x) * tmp
            saved = (x - t_left) * tmp
        values[j] = saved

    if dx != 0.0:
        for r in range(k + 1):
            values[r] += dx * slopes[i - k + r]
    return i - k


def _spline_transform_dense(
    const X_DTYPE[:, :] X,
    const float64_t[:, ::1] knots,
    const float64_t[:, ::1] slopes_lo,
    const float64_t[:, ::1] slopes_hi,
    intp_t degree,
    intp_t n_splines,
    bint include_bias,
    bint clamp,
    floating[:, :] out,
    int n_threads,
):
    """Write the B-splines of each feature of X into the zero-initialized `out`.

    knots, slopes_lo and slopes_hi have one row per feature. Missing values are
    encoded as zeros. Spline indices beyond n_splines wrap around, which is how
    periodic splines are built from their n_splines + degree B-splines.
    """
    cdef:
        intp_t n_samples = X.shape[0]
        intp_t n_features = X.shape[1]
        intp_t n_basis = knots.shape[1] - degree - 1
        intp_t n_cols = n_splines - 1 + include_bias
        intp_t sample_idx, feature_idx, first, col, r
        float64_t x
        float64_t* values = NULL

    with nogil, parallel(num_threads=n_threads):
        values = <float64_t*> malloc((degree + 1) * sizeof(float64_t))
        for sample_idx in prange(n_samples, schedule="static"):
            for feature_idx in range(n_features):
                x = X[sample_idx, feature_idx]
                if isnan(x):
                    continue
                first = _nonzero_basis(
                    &knots[feature_idx, 0], degree, n_basis, x, clamp,
                    &slopes_lo[feature_idx, 0], &slopes_hi[feature_idx, 0], values,
                )
                for r in range(degree + 1):
                    col = first + r
                    if col >= n_splines:
                        col = col - n_splines
                    # The last spline is dropped when include_bias=False.
                    if col < n_cols:
                        out[sample_idx, feature_idx * n_cols + col] += values[r]
        free(values)


def _spline_transform_sparse(
    const X_DTYPE[:, :] X,
    const float64_t[:, ::1] knots,
    const float64_t[:, ::1] slopes_lo,
    const float64_t[:, ::1] slopes_hi,
    intp_t degree,
    intp_t n_splines,
    bint include_bias,
    bint clamp,
    floating[:, ::1] data,
    int32_t[:, ::1] indices,
    int n_threads,
):
    """Same as `_spline_transform_dense` but write the CSR data and indices.

    Each row has degree + 1 entries per feature. Entries of missing values and of
    dropped splines are explicit zeros, and periodic splines can produce duplicate
    indices: the caller is expected to sum duplicates and eliminate zeros.
    """
    cdef:
        intp_t n_samples = X.shape[0]
        intp_t n_features = X.shape[1]
        intp_t n_basis = knots.shape[1] - degree - 1
        intp_t n_cols = n_splines - 1 + include_bias
        intp_t sample_idx, feature_idx, first, col, r, pos
        float64_t x
        float64_t* values = NULL

    with nogil, parallel(num_threads=n_threads):
        values = <float64_t*> malloc((degree + 1) * sizeof(float64_t))
        for sample_idx in prange(n_samples, schedule="static"):
            for feature_idx in range(n_features):
                pos = feature_idx * (degree + 1)
                x = X[sample_idx, feature_idx]
                if isnan(x):
                    for r in range(degree + 1):
                        data[sample_idx, pos + r] = 0.0
                        indices[sample_idx, pos + r] = feature_idx * n_cols
                    continue
                first = _nonzero_basis(
                    &knots[feature_idx, 0], degree, n_basis, x, clamp,
                    &slopes_lo[feature_idx, 0], &slopes_hi[feature_idx, 0], values,
                )
                for r in range(degree + 1):
                    col = first + r
                    if col >= n_splines:
                        col = col - n_splines
                    if col < n_cols:
                        data[sample_idx, pos + r] = values[r]
                        indices[sample_idx, pos + r] = feature_idx * n_cols + col
                    else:
                        data[sample_idx, pos + r] = 0.0
                        indices[sample_idx, pos + r] = feature_idx * n_cols
        free(values)
