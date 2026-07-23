# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _partitioner.pyx for details.

from libc.math cimport INFINITY

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, int32_t, intp_t, uint8_t
)
from sklearn.utils._bitset cimport BITSET_DTYPE_C, init_bitset, set_bitset
from sklearn.tree._splitter cimport SplitRecord

# Mitigate precision differences between 32 bit and 64 bit
cdef const float32_t FEATURE_THRESHOLD = 1e-7


cdef inline float64_t position_to_split_threshold(
    float32_t[::1] feature_values,
    intp_t p_prev,
    intp_t position,
    intp_t end_non_missing,
    bint missing_go_to_left,
) noexcept nogil:
    """Compute the threshold represented by a numerical split position.

    Parameters
    ----------
    feature_values : float32_t[::1]
        Sorted feature values for the current node.
    p_prev : intp_t
        Position of the last sample assigned to the left child.
    position : intp_t
        Position of the first sample assigned to the right child.
    end_non_missing : intp_t
        Position just after the last non-missing value.
    missing_go_to_left : bint
        Whether missing values are assigned to the left child.

    Returns
    -------
    float64_t
        The midpoint threshold, or +inf when ``position == end_non_missing``
        and missing values are assigned to the right child.
    """
    if position == end_non_missing and not missing_go_to_left:
        return INFINITY

    # Split between two non-missing values: sum of halves is
    # used to avoid infinite value.
    return feature_values[p_prev] / 2.0 + feature_values[position] / 2.0


# We provide here the abstract interface for a Partitioner that would be
# theoretically shared between the Dense and Sparse partitioners. However,
# we leave it commented out for now as it is not used in the current
# implementation due to the performance hit from vtable lookups when using
# inheritance based polymorphism. It is left here for future reference.
#
# Note: Instead, in `_splitter.pyx`, we define a fused type that can be used
# to represent both the dense and sparse partitioners.
#
# cdef class BasePartitioner:
#     cdef intp_t[::1] samples
#     cdef float32_t[::1] feature_values
#     cdef intp_t start
#     cdef intp_t end
#     cdef intp_t n_missing
#     cdef const uint8_t[::1] missing_values_in_feature_mask
#     cdef intp_t n_categories_current

#     cdef bint sort_samples_and_feature_values(
#         self, intp_t current_feature
#     ) noexcept nogil
#     cdef void init_node_split(
#         self,
#         intp_t start,
#         intp_t end
#     ) noexcept nogil
#     cdef void find_min_max(
#         self,
#         intp_t current_feature,
#         float32_t* min_feature_value_out,
#         float32_t* max_feature_value_out,
#     ) noexcept nogil
#     cdef void next_p(
#         self,
#         intp_t* p_prev,
#         intp_t* p,
#         bint missing_go_to_left
#     ) noexcept nogil
#     cdef intp_t partition_samples(
#         self,
#         float64_t current_threshold
#     ) noexcept nogil
#     cdef void partition_samples_final(
#         self,
#         const SplitRecord* best_split,
#     ) noexcept nogil


cdef class DensePartitioner:
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[:, :] X
    cdef const float64_t[:, :] y  # only for sorting of categoricals
    cdef const float64_t[::1] sample_weight
    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const uint8_t[::1] missing_values_in_feature_mask
    cdef char[::1] swap_buffer

    # memoryview of the n_categories_current in every feature
    cdef const intp_t[::1] n_categories
    cdef intp_t n_categories_current  # keep track of n_categories_current in current split

    # purely for Breiman shortcut
    cdef intp_t[::1] counts
    cdef float64_t[::1] weighted_counts
    cdef float64_t[::1] means
    cdef intp_t[::1] sorted_cat
    cdef intp_t[::1] offsets

    cdef bint sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil
    cdef void shift_missing_to_the_left(self) noexcept nogil
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    cdef void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil
    cdef inline void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left,
    ) noexcept nogil:
        """
        Compute the next p_prev and p for iterating over feature values.

        This method is used inside the best-split search function pass which starts
        by setting p = start at the beginning of each search pass and calls
        this method repeatedly with the same missing_go_to_left as for that pass.
        The expected layout of self.feature_values[start:end] is:
        - first pass (missing_go_to_left=False): after
          sort_samples_and_feature_values(), non-missing values are sorted and
          missing values are grouped at the right;
        - second pass (missing_go_to_left=True): after
          shift_missing_to_the_left(), missing values are grouped at the left.

        Given that layout, this method advances p to the next valid split
        position while skipping ties up to FEATURE_THRESHOLD:
        - if missing_go_to_left: iterate p in [start + n_missing + 1, end)
        - otherwise: iterate p in [start, end - n_missing].
          The special case p == end - n_missing corresponds to "all non-missing
          values on the left and all missing values on the right". The next
          call then sets p to end to terminate the search loop.
        """
        cdef intp_t end_non_missing = (
            self.end if missing_go_to_left
            else self.end - self.n_missing)

        # First pass special end marker: include "all non-missing left, all missing right"
        if p[0] == end_non_missing and not missing_go_to_left:
            p[0] = self.end
            p_prev[0] = self.end
            return

        # Second pass starts with missing on the left; jump to first non-missing
        if missing_go_to_left and p[0] == self.start:
            p[0] = self.start + self.n_missing

        # Move to next candidate split position
        p[0] += 1

        if self.n_categories_current > 0:
            while (
                p[0] < end_non_missing and
                self.feature_values[p[0]] == self.feature_values[p[0] - 1]
            ):
                p[0] += 1
        else:
            while (
                p[0] < end_non_missing and
                self.feature_values[p[0]] <= self.feature_values[p[0] - 1] + FEATURE_THRESHOLD
            ):
                p[0] += 1

        p_prev[0] = p[0] - 1
    cdef intp_t partition_samples(
        self,
        float64_t current_threshold,
        bint missing_go_to_left
    ) noexcept nogil
    cdef void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil

    cdef inline void cat_position_to_split_bitset(
        self,
        intp_t position,
        bint missing_go_to_left,
        BITSET_DTYPE_C left_cat_bitset,
    ) noexcept nogil:
        """Convert a categorical split position into a bitset."""
        cdef:
            intp_t n_left_non_missing = position - self.start
            intp_t offset = 0
            intp_t r
            intp_t c

        if missing_go_to_left:
            n_left_non_missing -= self.n_missing

        init_bitset(left_cat_bitset)

        if n_left_non_missing <= 0:
            return

        for r in range(self.n_categories_current):
            c = self.sorted_cat[r]
            set_bitset(left_cat_bitset, <uint8_t> c)
            offset += self.counts[c]
            if offset >= n_left_non_missing:
                break
    cdef void sort_categories(
        self,
        intp_t nc
    ) noexcept nogil

cdef class SparsePartitioner:
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[::1] X_data
    cdef const int32_t[::1] X_indices
    cdef const int32_t[::1] X_indptr
    cdef intp_t n_total_samples
    cdef intp_t[::1] index_to_samples
    cdef intp_t[::1] sorted_samples
    cdef intp_t start_positive
    cdef intp_t end_negative
    cdef bint is_samples_sorted

    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const uint8_t[::1] missing_values_in_feature_mask

    # memoryview of the n_categories_current in every feature
    cdef const intp_t[::1] n_categories
    cdef intp_t n_categories_current  # keep track of n_categories_current in current split

    # purely for Breiman shortcut for categorical features
    cdef intp_t[::1] counts
    cdef float64_t[::1] weighted_counts
    cdef float64_t[::1] means
    cdef intp_t[::1] sorted_cat
    cdef intp_t[::1] offsets

    cdef bint sort_samples_and_feature_values(
        self,
        intp_t current_feature
    ) noexcept nogil
    cdef void shift_missing_to_the_left(self) noexcept nogil
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    cdef void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil
    cdef inline void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left,
    ) noexcept nogil:
        """Compute the next p_prev and p for iterating over feature values.

        The missing_go_to_left argument is ignored for sparse data because
        sparse partitioning does not currently support missing values.
        """
        cdef intp_t p_next

        if p[0] + 1 != self.end_negative:
            p_next = p[0] + 1
        else:
            p_next = self.start_positive

        while (p_next < self.end and
                self.feature_values[p_next] <= self.feature_values[p[0]] + FEATURE_THRESHOLD):
            p[0] = p_next
            if p[0] + 1 != self.end_negative:
                p_next = p[0] + 1
            else:
                p_next = self.start_positive

        p_prev[0] = p[0]
        p[0] = p_next
    cdef intp_t partition_samples(
        self,
        float64_t current_threshold,
        bint missing_go_to_left,
    ) noexcept nogil
    cdef void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil

    cdef inline void cat_position_to_split_bitset(
        self,
        intp_t position,
        bint missing_go_to_left,
        BITSET_DTYPE_C left_cat_bitset,
    ) noexcept nogil:
        # Sparse categorical features are rejected before split search.
        init_bitset(left_cat_bitset)
    cdef void extract_nnz(
        self,
        intp_t feature
    ) noexcept nogil
    cdef intp_t _partition(
        self,
        float64_t threshold
    ) noexcept nogil


ctypedef fused array_data_type:
    intp_t
    float32_t
