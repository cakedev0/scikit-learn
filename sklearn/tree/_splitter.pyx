"""Splitting algorithms in the construction of a tree.

This module contains the main splitting algorithms for constructing a tree.
Splitting is concerned with finding the optimal partition of the data into
two groups. The impurity of the groups is minimized, and the impurity is measured
by some criterion, which is typically the Gini impurity or the entropy. Criterion
are implemented in the ``_criterion`` module.

Splitting evaluates a subset of features (defined by `max_features` also
known as mtry in the literature). The module supports two primary types
of splitting strategies:

- Best Split: A greedy approach to find the optimal split. This method
  ensures that the best possible split is chosen by examining various
  thresholds for each candidate feature.
- Random Split: A stochastic approach that selects a split randomly
  from a subset of the best splits. This method is faster but does
  not guarantee the optimal split.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport INFINITY, log
from libc.string cimport memcpy, memset

from sklearn.tree._criterion cimport (
    ClassificationCriterion,
    Criterion,
    RegressionCriterion,
)
from sklearn.tree._partitioner cimport (
    FEATURE_THRESHOLD, DensePartitioner, SparsePartitioner,
)
from sklearn.tree._utils cimport RAND_R_MAX, rand_int, rand_uniform

import numpy as np

# Introduce a fused-class to make it possible to share the split implementation
# between the dense and sparse cases in the node_split_best and node_split_random
# functions. The alternative would have been to use inheritance-based polymorphism
# but it would have resulted in a ~10% overall tree fitting performance
# degradation caused by the overhead frequent virtual method lookups.
ctypedef fused Partitioner:
    DensePartitioner
    SparsePartitioner


def _global_sorted_index(X, intp_t max_features):
    """Precompute feature-wise global sorted sample indices.

    Experimental RandomForest optimization: large dense nodes can filter this
    order instead of comparison-sorting node-local feature values.
    """
    X_array = np.asarray(X)
    n_samples, n_features = X_array.shape
    if max_features != n_features:
        return None, None

    sample_size = min(n_samples, 2048)
    sorted_samples = np.empty((n_features, n_samples), dtype=np.intp)
    feature_n_unique = np.zeros(n_features, dtype=np.intp)
    any_feature = False
    for feature_idx in range(X_array.shape[1]):
        # Cheap guard: the global sorted-index path is intended for continuous
        # or near-continuous features. Low-cardinality features are better left
        # to the current three-way sorter until they get a dedicated coded path.
        sample_values = X_array[:sample_size, feature_idx]
        if np.unique(sample_values).size < sample_size // 2:
            continue

        order = np.argsort(X_array[:, feature_idx], kind="mergesort")
        sorted_samples[feature_idx] = order
        ordered_values = X_array[order, feature_idx]
        if ordered_values.size == 0:
            feature_n_unique[feature_idx] = 0
        else:
            feature_n_unique[feature_idx] = 1 + np.count_nonzero(
                np.diff(ordered_values) > FEATURE_THRESHOLD
            )
        any_feature = True
    if not any_feature:
        return None, None
    return sorted_samples, feature_n_unique


def _hist_binned_features(X, intp_t max_bins, missing_values_in_feature_mask):
    """Precompute ordered feature bin codes for histogram splitting."""
    X_array = np.asarray(X)
    n_samples, n_features = X_array.shape
    codes = np.full((n_features, n_samples), -1, dtype=np.int32)
    bin_thresholds = np.empty((n_features, max_bins), dtype=np.float32)
    n_bins = np.zeros(n_features, dtype=np.intp)
    all_features_binned = True

    for feature_idx in range(n_features):
        values = X_array[:, feature_idx]
        if (
            missing_values_in_feature_mask is not None
            and missing_values_in_feature_mask[feature_idx]
        ):
            non_missing = ~np.isnan(values)
            feature_values = values[non_missing]
        else:
            non_missing = None
            feature_values = values

        uniques = np.unique(feature_values)
        if uniques.size == 0:
            n_bins[feature_idx] = 0
            continue

        if uniques.size <= max_bins:
            n_feature_bins = uniques.size
            thresholds = uniques[:-1] / 2.0 + uniques[1:] / 2.0
        else:
            percentile_ranks = np.linspace(0.0, 100.0, max_bins + 1)[1:-1]
            thresholds = np.percentile(
                feature_values, percentile_ranks, method="midpoint"
            )
            thresholds = np.unique(thresholds.astype(np.float32, copy=False))
            n_feature_bins = thresholds.size + 1

        thresholds = thresholds.astype(np.float32, copy=False)
        n_bins[feature_idx] = n_feature_bins
        if thresholds.size:
            bin_thresholds[feature_idx, : thresholds.size] = thresholds
        if non_missing is None:
            codes[feature_idx] = np.searchsorted(thresholds, values).astype(
                np.int32, copy=False
            )
        else:
            codes[feature_idx, non_missing] = np.searchsorted(
                thresholds, values[non_missing]
            ).astype(np.int32, copy=False)

    return codes, bin_thresholds, n_bins, all_features_binned


cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        """

        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=float64_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=float64_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:

        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef inline int node_split_best(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Find the best split
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_searches
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = -INFINITY
    cdef float64_t best_proxy_improvement = -INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    cdef intp_t p
    cdef intp_t p_prev

    cdef intp_t n_visited_features = 0
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            ((
                feature_values[end_non_missing - 1]
                <= feature_values[start] + FEATURE_THRESHOLD
            ) and n_missing == 0)
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0

        # Evaluate all splits

        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node
        # and the split with right node being only missing values is evaluated.
        # The second search will have all the missing values going to the left node.
        # This logic is governed by the partitionner and used here, so there is a strong coupling.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            if missing_go_to_left:
                partitioner.shift_missing_to_the_left()

            criterion.reset()

            p = start

            while p < end:
                partitioner.next_p(&p_prev, &p, missing_go_to_left)
                if p == end:
                    continue

                # Reject if min_samples_leaf is not guaranteed
                n_left = p - start
                n_right = end - p
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                # Reject if monotonicity constraints are not satisfied
                if (
                    with_monotonic_cst and
                    monotonic_cst[current_split.feature] != 0 and
                    not criterion.check_monotonicity(
                        monotonic_cst[current_split.feature],
                        lower_bound,
                        upper_bound,
                    )
                ):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    continue

                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    if p == end_non_missing and not missing_go_to_left:
                        # Split with the right node being only the missing values.
                        # Note that partioner.next_p never considers candidate
                        # splits for which the left node would move only the
                        # the missing values as this would be redundant with the
                        # split that only send missing values to the right.
                        # We use inf as a threshold because nan <= inf is false
                        # according to IEEE 754.
                        current_split.threshold = INFINITY
                    else:
                        # Split between two non-missing values: sum of halves is
                        # used to avoid infinite value.
                        current_split.threshold = (
                            feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        )

                    # If there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = current_split  # copy

    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            &best_split
        )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )

        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


cdef inline int node_split_random(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = - INFINITY
    cdef float64_t best_proxy_improvement = - INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants
    cdef intp_t n_visited_features = 0
    cdef float32_t min_feature_value
    cdef float32_t max_feature_value

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[

        current_split.feature = features[f_j]

        # Find min, max as we will randomly select a threshold between them
        partitioner.find_min_max(
            current_split.feature, &min_feature_value, &max_feature_value
        )
        n_missing = partitioner.n_missing

        if (
            # All values for this feature are missing, or
            end - start == n_missing or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            (max_feature_value <= min_feature_value + FEATURE_THRESHOLD and n_missing == 0)
        ):
            # We consider this feature constant in this case.
            # Since finding a split with a constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], current_split.feature

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0

        # Draw a random threshold
        current_split.threshold = rand_uniform(
            min_feature_value,
            max_feature_value,
            random_state,
        )

        if has_missing:
            # If there are missing values, then we randomly make all missing
            # values go to the right or left.
            #
            # Note: compared to the BestSplitter, we do not evaluate the
            # edge case where all the missing values go to the right node
            # and the non-missing values go to the left node. This is because
            # this would indicate a threshold outside of the observed range
            # of the feature. However, it is not clear how much probability weight should
            # be given to this edge case.
            missing_go_to_left = rand_int(0, 2, random_state)
        else:
            missing_go_to_left = 0

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

        # Partition
        current_split.pos = partitioner.partition_samples(
            current_split.threshold, missing_go_to_left
        )

        n_left = current_split.pos - start
        n_right = end - current_split.pos

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if min_weight_leaf is not satisfied
        if ((criterion.weighted_n_left < min_weight_leaf) or
                (criterion.weighted_n_right < min_weight_leaf)):
            continue

        # Reject if monotonicity constraints are not satisfied
        if (
                with_monotonic_cst and
                monotonic_cst[current_split.feature] != 0 and
                not criterion.check_monotonicity(
                    monotonic_cst[current_split.feature],
                    lower_bound,
                    upper_bound,
                )
        ):
            continue

        current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                current_split.missing_go_to_left = missing_go_to_left
            else:
                current_split.missing_go_to_left = n_left > n_right

            best_proxy_improvement = current_proxy_improvement
            best_split = current_split  # copy

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            partitioner.partition_samples_final(
                &best_split
            )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""
    cdef DensePartitioner partitioner
    cdef object global_sorted_samples
    cdef object feature_n_unique
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.global_sorted_samples, self.feature_n_unique = _global_sorted_index(
            X, self.max_features
        )
        self.partitioner = DensePartitioner(
            X,
            self.samples,
            self.feature_values,
            missing_values_in_feature_mask,
            self.global_sorted_samples,
            self.feature_n_unique,
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )


cdef class HistBestSplitter(BestSplitter):
    """Histogram splitter for dense data binned into at most ``max_bins`` bins."""
    cdef public intp_t max_bins
    cdef public object precomputed_bins
    cdef const int32_t[:, ::1] bin_codes
    cdef float32_t[:, ::1] bin_thresholds
    cdef intp_t[::1] n_bins
    cdef intp_t[::1] hist_counts
    cdef intp_t[::1] active_bins
    cdef float64_t[::1] hist_weights
    cdef float64_t[::1] hist_sums
    cdef float64_t[::1] hist_sq_sums
    cdef float64_t[:, ::1] hist_class_counts
    cdef float64_t[::1] missing_class_counts
    cdef float64_t[::1] left_class_counts
    cdef intp_t criterion_kind
    cdef intp_t n_classes
    cdef bint all_features_binned

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        cdef object codes
        cdef object bin_thresholds
        cdef object n_bins
        cdef object all_features_binned
        cdef intp_t workspace_bins
        cdef str criterion_name

        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.global_sorted_samples = None
        self.feature_n_unique = None
        self.partitioner = DensePartitioner(
            X,
            self.samples,
            self.feature_values,
            missing_values_in_feature_mask,
            self.global_sorted_samples,
            self.feature_n_unique,
        )
        if self.max_bins <= 0:
            self.all_features_binned = False
            return 0

        if self.precomputed_bins is None:
            codes, bin_thresholds, n_bins, all_features_binned = _hist_binned_features(
                X, self.max_bins, missing_values_in_feature_mask
            )
        else:
            codes, bin_thresholds, n_bins, all_features_binned = self.precomputed_bins
        self.bin_codes = codes
        self.bin_thresholds = bin_thresholds
        self.n_bins = n_bins
        self.all_features_binned = all_features_binned

        workspace_bins = self.max_bins
        self.hist_counts = np.empty(workspace_bins, dtype=np.intp)
        self.active_bins = np.empty(workspace_bins, dtype=np.intp)
        self.hist_weights = np.empty(workspace_bins, dtype=np.float64)
        self.hist_sums = np.empty(workspace_bins, dtype=np.float64)
        self.hist_sq_sums = np.empty(workspace_bins, dtype=np.float64)

        criterion_name = type(self.criterion).__name__
        if criterion_name == "Gini":
            self.criterion_kind = 1
        elif criterion_name == "Entropy":
            self.criterion_kind = 2
        elif criterion_name == "MSE":
            self.criterion_kind = 3
        else:
            self.criterion_kind = 0

        if self.criterion_kind in (1, 2):
            self.n_classes = (<ClassificationCriterion> self.criterion).n_classes[0]
            self.hist_class_counts = np.empty(
                (workspace_bins, self.n_classes), dtype=np.float64
            )
            self.missing_class_counts = np.empty(self.n_classes, dtype=np.float64)
            self.left_class_counts = np.empty(self.n_classes, dtype=np.float64)
        else:
            self.n_classes = 0
            self.hist_class_counts = np.empty((1, 1), dtype=np.float64)
            self.missing_class_counts = np.empty(1, dtype=np.float64)
            self.left_class_counts = np.empty(1, dtype=np.float64)
        return 0

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        cdef uint32_t saved_rand_r_state = self.rand_r_state
        cdef int hist_status

        if (
            not self.all_features_binned
            or self.criterion_kind == 0
            or self.with_monotonic_cst
        ):
            return node_split_best(
                self,
                self.partitioner,
                self.criterion,
                split,
                parent_record,
            )

        hist_status = self._node_split_hist(parent_record, split)
        if hist_status == 1:
            self.rand_r_state = saved_rand_r_state
            return node_split_best(
                self,
                self.partitioner,
                self.criterion,
                split,
                parent_record,
            )
        return hist_status

    cdef int _node_split_hist(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        cdef:
            SplitRecord best_split, current_split
            float64_t best_improvement = -INFINITY
            float64_t current_improvement
            float64_t impurity = parent_record.impurity
            intp_t start = self.start
            intp_t end = self.end
            intp_t node_size = end - start
            intp_t max_features = self.max_features
            intp_t min_samples_leaf = self.min_samples_leaf
            float64_t min_weight_leaf = self.min_weight_leaf
            uint32_t* random_state = &self.rand_r_state
            intp_t[::1] features = self.features
            intp_t[::1] constant_features = self.constant_features
            intp_t n_features = self.n_features
            intp_t n_known_constants = parent_record.n_constant_features
            intp_t n_total_constants = n_known_constants
            intp_t n_found_constants = 0
            intp_t n_drawn_constants = 0
            intp_t n_visited_features = 0
            intp_t f_i = n_features
            intp_t f_j
            intp_t current_feature
            intp_t n_feature_bins
            intp_t p
            intp_t i, j, c
            intp_t active_count
            intp_t missing_count
            intp_t left_count, right_count
            float64_t missing_weight
            float64_t missing_sum
            float64_t missing_sq_sum
            float64_t left_weight, right_weight
            float64_t left_sum, right_sum
            float64_t left_sq_sum, right_sq_sum
            float64_t impurity_left, impurity_right
            float64_t weighted_n_node_samples = (
                self.criterion.weighted_n_node_samples
            )
            float64_t weighted_n_samples = self.weighted_n_samples
            bint missing_go_to_left
            intp_t* n_bins_ptr = &self.n_bins[0]
            intp_t* hist_counts_ptr = &self.hist_counts[0]
            intp_t* active_bins_ptr = &self.active_bins[0]
            float64_t* hist_weights_ptr = &self.hist_weights[0]
            float64_t* hist_sums_ptr = &self.hist_sums[0]
            float64_t* hist_sq_sums_ptr = &self.hist_sq_sums[0]
            float64_t* hist_class_counts_ptr = &self.hist_class_counts[0, 0]
            float64_t* missing_class_counts_ptr = &self.missing_class_counts[0]
            float64_t* left_class_counts_ptr = &self.left_class_counts[0]
            const float32_t* bin_thresholds_ptr

        _init_split(&best_split, end)
        _init_split(&current_split, start)
        self.partitioner.init_node_split(start, end)

        while (
            f_i > n_total_constants
            and (n_visited_features < max_features or
                 n_visited_features <= n_found_constants + n_drawn_constants)
        ):
            n_visited_features += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = (
                    features[f_j], features[n_drawn_constants]
                )
                n_drawn_constants += 1
                continue

            f_j += n_found_constants
            current_feature = features[f_j]
            n_feature_bins = n_bins_ptr[current_feature]
            if n_feature_bins <= 0:
                return 1
            if node_size <= <intp_t>(0.02 * n_feature_bins):
                return 1

            active_count = self._build_feature_histogram(
                current_feature,
                n_feature_bins,
                &missing_count,
                &missing_weight,
                &missing_sum,
                &missing_sq_sum,
            )

            if active_count == 0:
                features[f_j], features[n_total_constants] = (
                    features[n_total_constants], features[f_j]
                )
                n_found_constants += 1
                n_total_constants += 1
                continue

            if active_count == 1 and missing_count == 0:
                features[f_j], features[n_total_constants] = (
                    features[n_total_constants], features[f_j]
                )
                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            current_split.feature = current_feature
            bin_thresholds_ptr = &self.bin_thresholds[current_feature, 0]

            for i in range(2 if missing_count > 0 else 1):
                missing_go_to_left = i == 1
                left_count = missing_count if missing_go_to_left else 0
                left_weight = missing_weight if missing_go_to_left else 0.0
                left_sum = missing_sum if missing_go_to_left else 0.0
                left_sq_sum = missing_sq_sum if missing_go_to_left else 0.0

                if self.criterion_kind in (1, 2):
                    for c in range(self.n_classes):
                        if missing_go_to_left:
                            left_class_counts_ptr[c] = missing_class_counts_ptr[c]
                        else:
                            left_class_counts_ptr[c] = 0.0

                for j in range(active_count):
                    p = active_bins_ptr[j]
                    left_count += hist_counts_ptr[p]
                    left_weight += hist_weights_ptr[p]

                    if self.criterion_kind == 3:
                        left_sum += hist_sums_ptr[p]
                        left_sq_sum += hist_sq_sums_ptr[p]
                    else:
                        for c in range(self.n_classes):
                            left_class_counts_ptr[c] += (
                                hist_class_counts_ptr[p * self.n_classes + c]
                            )

                    if j == active_count - 1 and not (
                        missing_count > 0 and not missing_go_to_left
                    ):
                        continue

                    right_count = node_size - left_count
                    if left_count < min_samples_leaf or right_count < min_samples_leaf:
                        continue

                    right_weight = weighted_n_node_samples - left_weight
                    if left_weight < min_weight_leaf or right_weight < min_weight_leaf:
                        continue

                    if self.criterion_kind == 3:
                        right_sum = (<RegressionCriterion> self.criterion).sum_total[0] - left_sum
                        right_sq_sum = (<RegressionCriterion> self.criterion).sq_sum_total - left_sq_sum
                        impurity_left = left_sq_sum / left_weight - (
                            left_sum / left_weight
                        ) * (left_sum / left_weight)
                        impurity_right = right_sq_sum / right_weight - (
                            right_sum / right_weight
                        ) * (right_sum / right_weight)
                    else:
                        self._hist_children_class_impurity(
                            left_weight,
                            right_weight,
                            &impurity_left,
                            &impurity_right,
                        )

                    current_improvement = (
                        weighted_n_node_samples / weighted_n_samples
                    ) * (
                        impurity
                        - left_weight / weighted_n_node_samples * impurity_left
                        - right_weight / weighted_n_node_samples * impurity_right
                    )

                    if current_improvement > best_improvement:
                        best_improvement = current_improvement
                        current_split.pos = start + left_count
                        current_split.improvement = current_improvement
                        current_split.impurity_left = impurity_left
                        current_split.impurity_right = impurity_right
                        if j == active_count - 1:
                            current_split.threshold = INFINITY
                        else:
                            current_split.threshold = bin_thresholds_ptr[p]
                        if missing_count == 0:
                            current_split.missing_go_to_left = left_count > right_count
                        else:
                            current_split.missing_go_to_left = missing_go_to_left
                        best_split = current_split

        if best_split.pos < end:
            self.partitioner.partition_samples_final(&best_split)
            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )
            best_split.improvement = self.criterion.impurity_improvement(
                impurity, best_split.impurity_left, best_split.impurity_right
            )

        memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)
        memcpy(&constant_features[n_known_constants],
               &features[n_known_constants],
               sizeof(intp_t) * n_found_constants)

        parent_record.n_constant_features = n_total_constants
        split[0] = best_split
        return 0

    cdef intp_t _build_feature_histogram(
        self,
        intp_t current_feature,
        intp_t n_feature_bins,
        intp_t* missing_count,
        float64_t* missing_weight,
        float64_t* missing_sum,
        float64_t* missing_sq_sum,
    ) noexcept nogil:
        cdef:
            intp_t i, p, sample_idx, c
            int32_t code
            float64_t w
            float64_t y_value
            intp_t active_count = 0
            intp_t* samples_ptr = &self.samples[0]
            const int32_t* codes_ptr = &self.bin_codes[current_feature, 0]
            intp_t* hist_counts_ptr = &self.hist_counts[0]
            intp_t* active_bins_ptr = &self.active_bins[0]
            float64_t* hist_weights_ptr = &self.hist_weights[0]
            float64_t* hist_sums_ptr = &self.hist_sums[0]
            float64_t* hist_sq_sums_ptr = &self.hist_sq_sums[0]
            float64_t* hist_class_counts_ptr = &self.hist_class_counts[0, 0]
            float64_t* missing_class_counts_ptr = &self.missing_class_counts[0]
            const float64_t* y_ptr = &self.y[0, 0]
            const float64_t* sample_weight_ptr = NULL

        if self.sample_weight is not None:
            sample_weight_ptr = &self.sample_weight[0]

        missing_count[0] = 0
        missing_weight[0] = 0.0
        missing_sum[0] = 0.0
        missing_sq_sum[0] = 0.0

        memset(hist_counts_ptr, 0, n_feature_bins * sizeof(intp_t))
        memset(hist_weights_ptr, 0, n_feature_bins * sizeof(float64_t))

        if self.criterion_kind == 3:
            memset(hist_sums_ptr, 0, n_feature_bins * sizeof(float64_t))
            memset(hist_sq_sums_ptr, 0, n_feature_bins * sizeof(float64_t))
        elif self.criterion_kind in (1, 2):
            memset(
                hist_class_counts_ptr,
                0,
                n_feature_bins * self.n_classes * sizeof(float64_t),
            )
            memset(missing_class_counts_ptr, 0, self.n_classes * sizeof(float64_t))

        for p in range(self.start, self.end):
            sample_idx = samples_ptr[p]
            code = codes_ptr[sample_idx]
            w = 1.0
            if sample_weight_ptr != NULL:
                w = sample_weight_ptr[sample_idx]

            if code < 0:
                missing_count[0] += 1
                missing_weight[0] += w
                if self.criterion_kind == 3:
                    y_value = y_ptr[sample_idx]
                    missing_sum[0] += w * y_value
                    missing_sq_sum[0] += w * y_value * y_value
                else:
                    c = <intp_t> y_ptr[sample_idx]
                    missing_class_counts_ptr[c] += w
                continue

            hist_counts_ptr[code] += 1
            hist_weights_ptr[code] += w
            if self.criterion_kind == 3:
                y_value = y_ptr[sample_idx]
                hist_sums_ptr[code] += w * y_value
                hist_sq_sums_ptr[code] += w * y_value * y_value
            else:
                c = <intp_t> y_ptr[sample_idx]
                hist_class_counts_ptr[code * self.n_classes + c] += w

        for i in range(n_feature_bins):
            if hist_counts_ptr[i] > 0:
                active_bins_ptr[active_count] = i
                active_count += 1

        return active_count

    cdef void _hist_children_class_impurity(
        self,
        float64_t left_weight,
        float64_t right_weight,
        float64_t* impurity_left,
        float64_t* impurity_right,
    ) noexcept nogil:
        cdef:
            intp_t c
            float64_t count_left
            float64_t count_right
            float64_t sq_left = 0.0
            float64_t sq_right = 0.0
            float64_t value
            float64_t* left_class_counts_ptr = &self.left_class_counts[0]
            const float64_t* total_class_counts_ptr = (
                &(<ClassificationCriterion> self.criterion).sum_total[0, 0]
            )

        if self.criterion_kind == 1:
            for c in range(self.n_classes):
                count_left = left_class_counts_ptr[c]
                count_right = total_class_counts_ptr[c] - count_left
                sq_left += count_left * count_left
                sq_right += count_right * count_right
            impurity_left[0] = 1.0 - sq_left / (left_weight * left_weight)
            impurity_right[0] = 1.0 - sq_right / (right_weight * right_weight)
        else:
            impurity_left[0] = 0.0
            impurity_right[0] = 0.0
            for c in range(self.n_classes):
                count_left = left_class_counts_ptr[c]
                if count_left > 0.0:
                    value = count_left / left_weight
                    impurity_left[0] -= value * log(value)

                count_right = total_class_counts_ptr[c] - count_left
                if count_right > 0.0:
                    value = count_right / right_weight
                    impurity_right[0] -= value * log(value)


cdef class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""
    cdef DensePartitioner partitioner
    cdef object global_sorted_samples
    cdef object feature_n_unique
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.global_sorted_samples, self.feature_n_unique = _global_sorted_index(
            X, self.max_features
        )
        self.partitioner = DensePartitioner(
            X,
            self.samples,
            self.feature_values,
            missing_values_in_feature_mask,
            self.global_sorted_samples,
            self.feature_n_unique,
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
