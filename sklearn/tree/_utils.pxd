# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

cimport numpy as cnp
from ..neighbors._quad_tree cimport Cell
from ..utils._typedefs cimport (
    float32_t, float64_t, intp_t, uint8_t, int32_t, uint32_t, uint64_t
)


ctypedef union SplitValue:
    # Union type to generalize the concept of a threshold to categorical
    # features. The floating point view, i.e. ``split_value.threshold`` is used
    # for numerical features, where feature values less than or equal to the
    # threshold go left, and values greater than the threshold go right.
    #
    # For categorical features, TODO
    float64_t threshold
    uint64_t cat_split  # bitset


cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    intp_t left_child                    # id of the left child of the node
    intp_t right_child                   # id of the right child of the node
    intp_t feature                       # Feature used for splitting the node
    float64_t threshold                  # Threshold value at the node, for continuous split (-INF otherwise)
    uint64_t categorical_bitset          # Bitset for categorical split (0 otherwise)
    float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    intp_t n_node_samples                # Number of samples at the node
    float64_t weighted_n_node_samples    # Weighted number of samples at the node
    uint8_t missing_go_to_left     # Whether features have missing values


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (uint8_t*)
    (WeightedPQueueRecord*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Cell*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)


cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil


cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil


cdef float64_t log(float64_t x) noexcept nogil


cdef int swap_array_slices(
    void* array, intp_t start, intp_t end, intp_t n, size_t itemsize
) except -1 nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    float64_t data
    float64_t weight

cdef class WeightedPQueue:
    cdef intp_t capacity
    cdef intp_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef intp_t size(self) noexcept nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil
    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef intp_t initial_capacity
    cdef WeightedPQueue samples
    cdef float64_t total_weight
    cdef intp_t k
    cdef float64_t sum_w_0_k  # represents sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    cdef intp_t size(self) noexcept nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int update_median_parameters_post_push(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int update_median_parameters_post_remove(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    cdef float64_t get_median(self) noexcept nogil
