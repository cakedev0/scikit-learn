"""This module contains routines for building histograms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

cimport cython
from cython.parallel import prange
from libc.string cimport memset

import numpy as np

from sklearn.ensemble._hist_gradient_boosting.common import HISTOGRAM_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common cimport hist_struct
from sklearn.ensemble._hist_gradient_boosting.common cimport X_BINNED_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common cimport G_H_DTYPE_C
from sklearn.utils._typedefs cimport uint8_t, uint32_t, int64_t


# Notes:
# - IN views are read-only, OUT views are write-only
# - In a lot of functions here, we pass feature_idx and the whole 2d
#   histograms arrays instead of just histograms[feature_idx]. This is because
#   Cython generated C code will have strange Python interactions (likely
#   related to the GIL release and the custom histogram dtype) when using 1d
#   histogram arrays that come from 2d arrays.
# - The for loops are un-wrapped, for example:
#
#   for i in range(n):
#       array[i] = i
#
#   will become
#
#   for i in range(n // 4):
#       array[i] = i
#       array[i + 1] = i + 1
#       array[i + 2] = i + 2
#       array[i + 3] = i + 3
#
#   This is to hint gcc that it can auto-vectorize these 4 operations and
#   perform them all at once.


@cython.final
cdef class HistogramBuilder:
    """A Histogram builder... used to build histograms.

    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.

    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    """
    cdef public:
        const X_BINNED_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        G_H_DTYPE_C [::1] gradients
        G_H_DTYPE_C [::1] hessians
        G_H_DTYPE_C [::1] ordered_gradients
        G_H_DTYPE_C [::1] ordered_hessians
        uint8_t hessians_are_constant
        int n_threads
        const X_BINNED_DTYPE_C [:, ::1] X_binned_rowmajor
        bint rowwise
        const uint32_t [::1] sparse_codes
        const int64_t [::1] sparse_row_ptr
        const X_BINNED_DTYPE_C [::1] most_freq_bins
        bint sparse_rowwise
        const uint32_t [::1] n_used_bins
        bint used_bins_only

    def __init__(self, const X_BINNED_DTYPE_C [::1, :] X_binned,
                 unsigned int n_bins, G_H_DTYPE_C [::1] gradients,
                 G_H_DTYPE_C [::1] hessians,
                 uint8_t hessians_are_constant,
                 int n_threads, X_binned_rowmajor=None, sparse_rows=None,
                 n_bins_non_missing=None):

        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <n_bins> bins, but some of the
        # bins may be unused if a feature has a small number of unique values.
        self.n_bins = n_bins
        self.gradients = gradients
        self.hessians = hessians
        # for root node, gradients and hessians are already ordered
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = n_threads
        self.rowwise = X_binned_rowmajor is not None
        if self.rowwise:
            self.X_binned_rowmajor = X_binned_rowmajor
        self.sparse_rowwise = sparse_rows is not None
        # Bins at or above n_bins_non_missing (except the missing-values bin)
        # are never read by the splitter, so non-root histograms can skip them.
        self.used_bins_only = n_bins_non_missing is not None
        if self.used_bins_only:
            self.n_used_bins = n_bins_non_missing
        if self.sparse_rowwise:
            self.sparse_codes, self.sparse_row_ptr, self.most_freq_bins = sparse_rows

    def compute_histograms_brute(
        HistogramBuilder self,
        const unsigned int [::1] sample_indices,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        """Compute the histograms of the node by scanning through all the data.

        For a given feature, the complexity is O(n_samples)

        Parameters
        ----------
        sample_indices : array of int, shape (n_samples_at_node,)
            The indices of the samples at the node to split.

        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape (n_features, n_bins)
            The computed histograms of the current node.
        """
        cdef:
            int n_samples
            int feature_idx
            int f_idx
            int i
            # need local views to avoid python interactions
            uint8_t hessians_are_constant = self.hessians_are_constant
            int n_allowed_features = self.n_features
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            G_H_DTYPE_C [::1] gradients = self.gradients
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            G_H_DTYPE_C [::1] hessians = self.hessians
            # Histograms will be initialized to zero later within a prange
            hist_struct [:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads
            bint rowwise = self.rowwise and not has_interaction_cst
            bint sparse_rowwise = self.sparse_rowwise and not has_interaction_cst
            bint partial_zero
            int n_blocks, block
            unsigned int f_start, f_end
            unsigned int f_zero
            const uint32_t [::1] n_used_bins = (
                self.n_used_bins if self.used_bins_only else None
            )

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        with nogil:
            n_samples = sample_indices.shape[0]

            # Populate ordered_gradients and ordered_hessians. (Already done
            # for root) Ordering the gradients and hessians helps to improve
            # cache hit.
            if sample_indices.shape[0] != gradients.shape[0]:
                if hessians_are_constant:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                else:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                        ordered_hessians[i] = hessians[sample_indices[i]]

            partial_zero = (
                self.used_bins_only and sample_indices.shape[0] != gradients.shape[0]
            )
            if sparse_rowwise:
                if partial_zero:
                    for f_zero in range(self.n_features):
                        _zero_used_bins(histograms, f_zero, self.n_bins,
                                        n_used_bins[f_zero])
                else:
                    memset(&histograms[0, 0], 0,
                           self.n_features * self.n_bins * sizeof(hist_struct))
            if sparse_rowwise:
                _build_histogram_rowwise_sparse(
                    sample_indices, self.sparse_codes, self.sparse_row_ptr,
                    self.most_freq_bins, ordered_gradients, ordered_hessians,
                    hessians_are_constant, self.n_bins, self.n_features,
                    gradients.shape[0], n_used_bins, histograms)
            elif rowwise:
                # One contiguous block of features per thread, each built
                # row-wise over the node's samples: no reduction needed.
                n_blocks = min(n_threads, self.n_features)
                for block in prange(n_blocks, schedule='static',
                                    num_threads=n_threads):
                    f_start = block * self.n_features // n_blocks
                    f_end = (block + 1) * self.n_features // n_blocks
                    for f_zero in range(f_start, f_end):
                        if partial_zero:
                            _zero_used_bins(histograms, f_zero, self.n_bins,
                                            n_used_bins[f_zero])
                        else:
                            memset(&histograms[f_zero, 0], 0,
                                   self.n_bins * sizeof(hist_struct))
                    _build_histogram_rowwise(
                        sample_indices, self.X_binned_rowmajor,
                        ordered_gradients, ordered_hessians,
                        hessians_are_constant, self.n_bins, f_start, f_end,
                        histograms)
            else:
                # Compute histogram of each feature
                for f_idx in prange(
                    n_allowed_features, schedule='static', num_threads=n_threads
                ):
                    if has_interaction_cst:
                        feature_idx = allowed_features[f_idx]
                    else:
                        feature_idx = f_idx

                    self._compute_histogram_brute_single_feature(
                        feature_idx, sample_indices, histograms
                    )

        return histograms

    cdef void _compute_histogram_brute_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,  # IN
            hist_struct [:, ::1] histograms) noexcept nogil:  # OUT
        """Compute the histogram for a given feature."""

        cdef:
            unsigned int n_samples = sample_indices.shape[0]
            const X_BINNED_DTYPE_C [::1] X_binned = \
                self.X_binned[:, feature_idx]
            unsigned int root_node = X_binned.shape[0] == n_samples
            G_H_DTYPE_C [::1] ordered_gradients = \
                self.ordered_gradients[:n_samples]
            G_H_DTYPE_C [::1] ordered_hessians = \
                self.ordered_hessians[:n_samples]
            uint8_t hessians_are_constant = \
                self.hessians_are_constant

        # Set histograms to zero.
        if self.used_bins_only and not root_node:
            _zero_used_bins(histograms, feature_idx, self.n_bins,
                            self.n_used_bins[feature_idx])
        else:
            memset(&histograms[feature_idx, 0], 0, self.n_bins * sizeof(hist_struct))

        if root_node:
            if hessians_are_constant:
                _build_histogram_root_no_hessian(feature_idx, X_binned,
                                                 ordered_gradients,
                                                 histograms)
            else:
                _build_histogram_root(feature_idx, X_binned,
                                      ordered_gradients, ordered_hessians,
                                      histograms)
        else:
            if hessians_are_constant:
                _build_histogram_no_hessian(feature_idx,
                                            sample_indices, X_binned,
                                            ordered_gradients, histograms)
            else:
                _build_histogram(feature_idx, sample_indices,
                                 X_binned, ordered_gradients,
                                 ordered_hessians, histograms)

    def compute_histograms_subtraction(
        HistogramBuilder self,
        hist_struct [:, ::1] parent_histograms,        # IN and OUT
        hist_struct [:, ::1] sibling_histograms,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        """Compute the histograms of the node using the subtraction trick.

        hist(parent) = hist(left_child) + hist(right_child)

        For a given feature, the complexity is O(n_bins). This is much more
        efficient than compute_histograms_brute, but it's only possible for one
        of the siblings.

        Parameters
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the parent.
        sibling_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the sibling.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            The computed histograms of the current node.
            We repurpose parent_histograms for this and don't need to allocate new
            memory.
        """

        cdef:
            int feature_idx
            int f_idx
            int n_allowed_features = self.n_features
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # Compute histogram of each feature
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True,
                            num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            if self.used_bins_only:
                _subtract_histograms(
                    feature_idx,
                    self.n_used_bins[feature_idx],
                    parent_histograms,
                    sibling_histograms,
                )
                _subtract_bin(self.n_bins - 1, feature_idx, parent_histograms,
                              sibling_histograms)
            else:
                _subtract_histograms(
                    feature_idx,
                    self.n_bins,
                    parent_histograms,
                    sibling_histograms,
                )
        return parent_histograms


cdef inline void _zero_used_bins(
        hist_struct [:, ::1] histograms,
        unsigned int feature_idx,
        unsigned int n_bins,
        unsigned int n_used) noexcept nogil:
    """Zero bins [0, n_used) and the missing-values bin (the last one)."""
    memset(&histograms[feature_idx, 0], 0, n_used * sizeof(hist_struct))
    memset(&histograms[feature_idx, n_bins - 1], 0, sizeof(hist_struct))


cdef inline void _subtract_bin(
        unsigned int bin_idx,
        int feature_idx,
        hist_struct [:, ::1] hist_a,  # IN and OUT
        hist_struct [:, ::1] hist_b,  # IN
) noexcept nogil:
    hist_a[feature_idx, bin_idx].sum_gradients -= hist_b[feature_idx, bin_idx].sum_gradients
    hist_a[feature_idx, bin_idx].sum_hessians -= hist_b[feature_idx, bin_idx].sum_hessians
    hist_a[feature_idx, bin_idx].count -= hist_b[feature_idx, bin_idx].count


cdef void _build_histogram_rowwise(
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [:, ::1] X_binned_rowmajor,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        uint8_t hessians_are_constant,
        unsigned int n_bins,
        unsigned int f_start,
        unsigned int f_end,
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Histograms of features [f_start, f_end) in one pass over the node's rows.

    Each row's bins are contiguous, and its gradient and hessian are read
    once instead of once per feature."""
    cdef:
        unsigned int n_samples = sample_indices.shape[0]
        bint root_node = n_samples == X_binned_rowmajor.shape[0]
        unsigned int i, f, sample_idx
        const X_BINNED_DTYPE_C* row
        hist_struct* hist = &out[0, 0]
        hist_struct* h
        G_H_DTYPE_C gradient, hessian

    for i in range(n_samples):
        sample_idx = i if root_node else sample_indices[i]
        row = &X_binned_rowmajor[sample_idx, 0]
        gradient = ordered_gradients[i]
        if hessians_are_constant:
            for f in range(f_start, f_end):
                h = hist + f * n_bins + row[f]
                h.sum_gradients += gradient
                h.count += 1
        else:
            hessian = ordered_hessians[i]
            for f in range(f_start, f_end):
                h = hist + f * n_bins + row[f]
                h.sum_gradients += gradient
                h.sum_hessians += hessian
                h.count += 1


cdef void _build_histogram_rowwise_sparse(
        const unsigned int [::1] sample_indices,  # IN
        const uint32_t [::1] codes,  # IN
        const int64_t [::1] row_ptr,  # IN
        const X_BINNED_DTYPE_C [::1] most_freq_bins,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        uint8_t hessians_are_constant,
        unsigned int n_bins,
        unsigned int n_features,
        unsigned int n_total_samples,
        const uint32_t [::1] n_used_bins,  # IN, None for all bins
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Row-wise histograms from rows that only store the bins that differ
    from their feature's most frequent bin (as feature_idx * n_bins + bin).

    The most frequent bin of each feature is filled in last, as the node's
    totals minus the feature's other bins."""
    cdef:
        unsigned int n_samples = sample_indices.shape[0]
        bint root_node = n_samples == n_total_samples
        unsigned int i, f, b, sample_idx, m
        int64_t k
        hist_struct* hist = &out[0, 0]
        hist_struct* h
        G_H_DTYPE_C gradient, hessian
        double sum_gradients = 0., sum_hessians = 0.
        double rest_gradients, rest_hessians
        unsigned int rest_count
        unsigned int n_scan
        bint all_bins = n_used_bins is None

    for i in range(n_samples):
        sample_idx = i if root_node else sample_indices[i]
        gradient = ordered_gradients[i]
        sum_gradients += gradient
        if hessians_are_constant:
            for k in range(row_ptr[sample_idx], row_ptr[sample_idx + 1]):
                h = hist + codes[k]
                h.sum_gradients += gradient
                h.count += 1
        else:
            hessian = ordered_hessians[i]
            sum_hessians += hessian
            for k in range(row_ptr[sample_idx], row_ptr[sample_idx + 1]):
                h = hist + codes[k]
                h.sum_gradients += gradient
                h.sum_hessians += hessian
                h.count += 1

    for f in range(n_features):
        m = most_freq_bins[f]
        rest_gradients = sum_gradients
        rest_hessians = sum_hessians
        rest_count = n_samples
        n_scan = n_bins if all_bins else n_used_bins[f]
        for b in range(n_scan):
            h = hist + f * n_bins + b
            rest_gradients -= h.sum_gradients
            rest_hessians -= h.sum_hessians
            rest_count -= h.count
        if not all_bins and m != n_bins - 1:
            h = hist + f * n_bins + n_bins - 1
            rest_gradients -= h.sum_gradients
            rest_hessians -= h.sum_hessians
            rest_count -= h.count
        h = hist + f * n_bins + m
        h.sum_gradients = rest_gradients
        h.sum_hessians = rest_hessians
        h.count = rest_count


cpdef void _build_histogram_naive(
        const int feature_idx,
        unsigned int [:] sample_indices,  # IN
        X_BINNED_DTYPE_C [:] binned_feature,  # IN
        G_H_DTYPE_C [:] ordered_gradients,  # IN
        G_H_DTYPE_C [:] ordered_hessians,  # IN
        hist_struct [:, :] out) noexcept nogil:  # OUT
    """Build histogram in a naive way, without optimizing for cache hit.

    Used in tests to compare with the optimized version."""
    cdef:
        unsigned int i
        unsigned int n_samples = sample_indices.shape[0]
        unsigned int sample_idx
        unsigned int bin_idx

    for i in range(n_samples):
        sample_idx = sample_indices[i]
        bin_idx = binned_feature[sample_idx]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _subtract_histograms(
        const int feature_idx,
        unsigned int n_bins,
        hist_struct [:, ::1] hist_a,  # IN and OUT
        hist_struct [:, ::1] hist_b,  # IN
) noexcept nogil:  # OUT
    """compute hist_a = hist_a - hist_b"""
    # Note that subtraction of large sums of floating point numbers, as we have here,
    # can exhibit catastrophic cancallation. This is in particular true for gradients
    # as they can be positive and negative, while hessians are non-negative.
    # Remember that gradients and hessians are originally computed in
    # G_H_DTYPE_C = float32 precision. Therefore, if sum_gradients and sum_hessians are
    # float64, we don't loose precision. But if we also used float32 for summation, we
    # would need to take care of floating point errors.
    #
    # Note that we could protect for negative hessians by setting:
    #     sum_hessians = max(0, sum_hessians)
    # But as we use float64 for summing float32, that's veeeery unlikely.
    cdef:
        unsigned int i = 0
    for i in range(n_bins):
        hist_a[feature_idx, i].sum_gradients -= hist_b[feature_idx, i].sum_gradients
        hist_a[feature_idx, i].sum_hessians -= hist_b[feature_idx, i].sum_hessians
        hist_a[feature_idx, i].count -= hist_b[feature_idx, i].count


cpdef void _build_histogram(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Return histogram for a given feature."""
    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += ordered_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_no_hessian(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Return histogram for a given feature, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_root(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        const G_H_DTYPE_C [::1] all_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Compute histogram of the root node.

    Unlike other nodes, the root node has to find the split among *all* the
    samples from the training set. binned_feature and all_gradients /
    all_hessians already have a consistent ordering.
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):

        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += all_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_root_no_hessian(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Compute histogram of the root node, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].count += 1
