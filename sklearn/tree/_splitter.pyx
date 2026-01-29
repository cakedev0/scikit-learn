"""Splitting algorithms used to build decision trees.

Splitting searches for a feature/threshold that best partitions the samples
according to an impurity criterion (see ``_criterion``). A splitter evaluates
only a subset of features (``max_features`` / mtry) and relies on a
``BasePartitioner`` to handle dense vs. sparse data.

The module implements two strategies:

- Best Split: evaluates all candidate thresholds per feature and selects the
  split with maximal proxy improvement.
- Random Split: draws one random threshold per feature and keeps the best among
  those random candidates (faster, but not guaranteed optimal).
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport final
from libc.string cimport memcpy

from sklearn.utils._typedefs cimport int8_t
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._partitioner cimport FEATURE_THRESHOLD, BasePartitioner
from sklearn.tree._utils cimport RAND_R_MAX, rand_int, rand_uniform

import numpy as np

# Use a BasePartitioner interface so a single splitter implementation can
# handle dense and sparse data. This trades a bit of virtual-call overhead for
# reduced code duplication and easier maintenance.


cdef float64_t INFINITY = np.inf


cdef inline void swap(intp_t[::1] features, intp_t i, intp_t j) noexcept nogil:
    # Helper for features sampling
    features[i], features[j] = features[j], features[i]


cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(
        self,
        Criterion criterion,
        BasePartitioner partitioner,
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
            Initialized criterion holding the target values and sample weights.

        partitioner : BasePartitioner
            Dense or sparse partitioner that owns the samples vector and
            feature_values buffer shared with the splitter.

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
            Monotonicity constraints, or None.

        """
        self.criterion = criterion
        self.partitioner = partitioner

        n_features = partitioner.n_features
        self.features = np.arange(n_features, dtype=np.intp)
        self.constant_features = np.empty(n_features, dtype=np.intp)
        self.n_features = n_features

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf

        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

    cdef void node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) noexcept nogil:
        """Reset splitter on node samples[start:end].

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

        self.criterion.init_node_split(start, end)
        self.partitioner.init_node_split(start, end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """Find the best split on node samples[start:end]"""
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] constant_features = self.constant_features
        cdef intp_t n_features = self.n_features
        cdef int is_constant

        cdef intp_t max_features = self.max_features
        cdef uint32_t* random_state = &self.rand_r_state

        cdef intp_t f_i = n_features
        cdef intp_t f_j

        cdef intp_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef intp_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_known_constants = parent_record.n_constant_features
        # n_total_constants = n_known_constants + n_found_constants
        cdef intp_t n_total_constants = n_known_constants

        cdef SplitRecord best_split
        best_split.improvement = -INFINITY
        cdef float64_t proxy_improvement = -INFINITY

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (
            f_i > n_total_constants   # Stop early if remaining features are constant
            and (
                n_visited_features < max_features
                # At least one drawn features must be non constant
                or n_visited_features <= n_found_constants + n_drawn_constants
            )
        ):

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
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                swap(features, f_j, n_drawn_constants)
                n_drawn_constants += 1
                continue

            # f_j in the interval [n_known_constants, f_i - n_found_constants[
            f_j += n_found_constants
            # f_j in the interval [n_total_constants, f_i[

            is_constant = self.node_split_for_feature(
                features[f_j], parent_record, &best_split, &proxy_improvement
            )

            if is_constant == -1:  # error code
                return -1

            if is_constant == 1:
                swap(features, f_j, n_total_constants)
                n_found_constants += 1
                n_total_constants += 1
            else:
                f_i -= 1
                swap(features, f_i, f_j)

        if proxy_improvement > -INFINITY:
            # Reorganize into samples[start:pos] + samples[pos:end]
            best_split.pos = self.partitioner.partition_samples_final(
                best_split.threshold,
                best_split.feature,
                best_split.missing_go_to_left
            )

            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )

            best_split.improvement = self.criterion.impurity_improvement(
                parent_record.impurity,
                best_split.impurity_left,
                best_split.impurity_right
            )

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(
            &constant_features[n_known_constants],
            &features[n_known_constants],
            sizeof(intp_t) * n_found_constants
        )

        # Return values
        parent_record.n_constant_features = n_total_constants
        split[0] = best_split
        return 0

    cdef int node_split_for_feature(
        self,
        intp_t feature,
        ParentInfo* parent_record,
        SplitRecord* best_split,
        float64_t* best_proxy_improvement,
    ) except -1 nogil:
        pass


@final
cdef class BestSplitter(Splitter):

    cdef int node_split_for_feature(
        self,
        intp_t feature,
        ParentInfo* parent_record,
        SplitRecord* best_split,
        float64_t* best_proxy_improvement,
    ) except -1 nogil:
        """Find the best split on node samples[start:end]

        Returns
        -------
        int
            -1 on failure, 1 if the feature is constant, 0 otherwise.
        """
        # Find the best split
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_missing = 0
        cdef bint has_missing = 0
        cdef bint is_constant
        cdef intp_t n_searches
        cdef intp_t n_left, n_right
        cdef bint missing_go_to_left

        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf

        cdef float64_t current_proxy_improvement

        cdef intp_t p
        cdef intp_t p_prev

        is_constant = self.partitioner.sort_samples_and_feature_values(feature)

        if is_constant:
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            return 1

        n_missing = self.partitioner.n_missing
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
                self.partitioner.shift_missing_to_the_left()

            self.criterion.reset()

            p = start

            while p < end:
                self.partitioner.next_p(&p_prev, &p)
                if p == end:
                    continue

                # Reject if min_samples_leaf is not guaranteed
                n_left = p - start
                n_right = end - p
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                self.criterion.update(p)

                # Reject if monotonicity constraints are not satisfied
                if (
                    self.with_monotonic_cst and
                    self.monotonic_cst[feature] != 0 and
                    not self.criterion.check_monotonicity(
                        self.monotonic_cst[feature],
                        parent_record.lower_bound,
                        parent_record.upper_bound,
                    )
                ):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if (
                    self.criterion.weighted_n_left < min_weight_leaf
                    or self.criterion.weighted_n_right < min_weight_leaf
                ):
                    continue

                current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement[0]:
                    # write into best split:
                    best_proxy_improvement[0] = current_proxy_improvement
                    best_split.threshold = self.partitioner.pos_to_threshold(p_prev, p)
                    best_split.feature = feature
                    # if there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        best_split.missing_go_to_left = n_left > n_right
                    else:
                        best_split.missing_go_to_left = missing_go_to_left

        return 0


@final
cdef class RandomSplitter(Splitter):

    cdef int node_split_for_feature(
        self,
        intp_t feature,
        ParentInfo* parent_record,
        SplitRecord* best_split,
        float64_t* best_proxy_improvement,
    ) except -1 nogil:

        """Find the best split on node samples[start:end]

        Returns
        -------
        int
            -1 on failure, 1 if the feature is constant, 0 otherwise.
        """
        # Find the best split
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t n_missing = 0
        cdef bint has_missing = 0
        cdef intp_t n_left, n_right
        cdef bint missing_go_to_left

        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf

        cdef float64_t current_proxy_improvement
        cdef float64_t threshold
        cdef float32_t min_feature_value, max_feature_value
        cdef uint32_t* random_state = &self.rand_r_state

        cdef intp_t p

        # Find min, max as we will randomly select a threshold between them
        self.partitioner.find_min_max(
            feature, &min_feature_value, &max_feature_value
        )
        n_missing = self.partitioner.n_missing

        if (
            # All values for this feature are missing, or
            end - start == n_missing or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            (max_feature_value <= min_feature_value + FEATURE_THRESHOLD and n_missing == 0)
        ):
            # We consider this feature constant in this case.
            # Since finding a split with a constant feature is not valuable,
            # we do not consider this feature for splitting.
            return 1

        has_missing = n_missing != 0

        # Draw a random threshold
        threshold = rand_uniform(min_feature_value, max_feature_value, random_state)

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

        if threshold == max_feature_value:
            threshold = min_feature_value

        # Partition
        p = self.partitioner.partition_samples(threshold, missing_go_to_left)

        n_left = p - start
        n_right = end - p

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            return 0

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        self.criterion.reset()
        self.criterion.update(p)

        # Reject if min_weight_leaf is not satisfied
        if (
            self.criterion.weighted_n_left < min_weight_leaf
            or self.criterion.weighted_n_right < min_weight_leaf
        ):
            return 0

        # Reject if monotonicity constraints are not satisfied
        if (
            self.with_monotonic_cst and
            self.monotonic_cst[feature] != 0 and
            not self.criterion.check_monotonicity(
                self.monotonic_cst[feature],
                parent_record.lower_bound,
                parent_record.upper_bound,
            )
        ):
            return 0

        current_proxy_improvement = self.criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement[0]:
            # write into best split:
            best_proxy_improvement[0] = current_proxy_improvement
            best_split.threshold = threshold
            best_split.feature = feature
            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                best_split.missing_go_to_left = missing_go_to_left
            else:
                best_split.missing_go_to_left = n_left > n_right

        return 0
