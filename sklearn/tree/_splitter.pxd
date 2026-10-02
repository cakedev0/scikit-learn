# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _splitter.pyx for details.

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, int8_t, int32_t, intp_t, uint8_t, uint32_t
)

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport ParentInfo
from sklearn.utils._bitset cimport BITSET_DTYPE_C

cdef struct SplitRecord:
    # Data to track sample split
    intp_t feature         # Which feature to split on.
    intp_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.

    # Threshold for numerical features splits:
    # - feature values less than or equal to the threshold go left, and values greater than the threshold go right.
    float64_t threshold

    # Threshold, or hash seed for categorical features splits:
    # - for SPLIT_CATEGORICAL_BITSET: left_cat_bitset stores the set of
    #   categories that go to the left child;
    # - for SPLIT_CATEGORICAL_HASH: left_cat_bitset[0] stores the hash seed.
    BITSET_DTYPE_C left_cat_bitset

    # The kind of split:
    # - SPLIT_NUMERIC: numerical split
    # - SPLIT_CATEGORICAL_BITSET: categorical split using bitset
    # - SPLIT_CATEGORICAL_HASH: categorical split using hash
    # - SPLIT_LEAF: no split, the node is a leaf
    int8_t split_kind

    float64_t improvement     # Impurity improvement given parent node.
    float64_t impurity_left   # Impurity of the left split.
    float64_t impurity_right  # Impurity of the right split.
    float64_t lower_bound     # Lower bound on value of both children for monotonicity
    float64_t upper_bound     # Upper bound on value of both children for monotonicity
    uint8_t missing_go_to_left  # Controls if missing values go to the left node.


cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public intp_t max_features      # Number of features to test
    cdef public intp_t min_samples_leaf  # Min samples in a leaf
    cdef public float64_t min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef uint32_t rand_r_state           # sklearn_rand_r random number state

    cdef intp_t[::1] samples             # Sample indices in X, y
    cdef intp_t n_samples                # X.shape[0]
    cdef float64_t weighted_n_samples    # Weighted number of samples
    cdef intp_t[::1] features            # Feature indices in X
    cdef intp_t[::1] constant_features   # Constant features indices
    cdef intp_t n_features               # X.shape[1]
    cdef float32_t[::1] feature_values   # temp. array holding feature values

    cdef intp_t start                    # Start position for the current node
    cdef intp_t end                      # End position for the current node

    cdef const float64_t[:, ::1] y
    # Monotonicity constraints for each feature.
    # The encoding is as follows:
    #   -1: monotonic decrease
    #    0: no constraint
    #   +1: monotonic increase
    cdef const int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst
    cdef const float64_t[:] sample_weight

    # Per-feature number of categories; -1 means the feature is numerical.
    cdef const intp_t[:] n_categories

    # PHASE TIMERS (benchmarking only): time in timer ticks (see _now) spent
    # in each phase of the tree construction, accumulated over all nodes.
    cdef public float64_t time_sort      # sort_samples_and_feature_values
    cdef public float64_t time_search    # split search loop over positions
    cdef public float64_t time_final     # partition_samples_final + children impurity
    cdef public float64_t time_node_reset  # node_reset (criterion.init)
    cdef public intp_t n_sorts

    # PHASE TIMERS BY NODE SIZE: the same phase times, plus the time per node
    # measured by the tree builder, by node size bucket b, for nodes with
    # 2**b <= n_node_samples < 2**(b + 1). Columns of bucket_times: sort,
    # search, final, node_reset, node_total. Columns of bucket_counts: number
    # of nodes, number of split nodes (node_split called), sum of depths.
    cdef public intp_t current_depth     # set by the tree builder
    cdef public float64_t[:, ::1] bucket_times
    cdef public intp_t[:, ::1] bucket_counts

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
        const intp_t[::1] n_categories,
        object rank_encoding,
    ) except -1

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil

    cdef void node_value(self, float64_t* dest) noexcept nogil

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil

    cdef float64_t node_impurity(self) noexcept nogil


# PHASE TIMERS (benchmarking only)
cdef enum:
    N_SIZE_BUCKETS = 48
    BUCKET_SORT = 0
    BUCKET_SEARCH = 1
    BUCKET_FINAL = 2
    BUCKET_NODE_RESET = 3
    BUCKET_NODE_TOTAL = 4
    BUCKET_N_NODES = 0
    BUCKET_N_SPLIT = 1
    BUCKET_DEPTH_SUM = 2


cdef extern from *:
    """
    #if defined(__x86_64__) || defined(_M_X64)
    #include <x86intrin.h>
    static inline unsigned long long sklearn_timer_ticks(void) { return __rdtsc(); }
    #else
    #include <time.h>
    static inline unsigned long long sklearn_timer_ticks(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (unsigned long long) ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }
    #endif
    """
    unsigned long long sklearn_timer_ticks() noexcept nogil


cdef inline float64_t _now() noexcept nogil:
    """Timer ticks: CPU timestamp counter on x86-64, ns elsewhere.

    Cheaper than clock_gettime; see sklearn.tree._phase_timers for the
    conversion to seconds.
    """
    return <float64_t> sklearn_timer_ticks()


cdef inline intp_t _size_bucket(intp_t n_node_samples) noexcept nogil:
    """b such that 2**b <= n_node_samples < 2**(b + 1)."""
    cdef intp_t b = 0
    while n_node_samples > 1:
        n_node_samples >>= 1
        b += 1
    return b
