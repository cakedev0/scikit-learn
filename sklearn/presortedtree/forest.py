# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numba import njit, prange, set_num_threads

from sklearn.presortedtree.dt import (
    DecisionTreeRegressor,
    _build_tree_numba,
    _compute_max_nodes,
)


@njit(parallel=True)
def _parallel_build_trees_numba(
    X, y, sorted_idx, cant_split, n_missing, max_depth, n_estimators, n_bootstrap
):
    """
    Build multiple trees in parallel using numba prange with sample weights.

    Instead of creating bootstrapped subsets and resorting, this function:
    1. Converts bootstrap sample indices to sample weights
    2. Uses the original sorted indices with weights
    3. Avoids expensive resorting for each tree

    Parameters
    ----------
    X : float32 2D array, shape (D, N)
    y : float64 1D array, shape (N,)
    sorted_idx_base : int32 2D array, shape (D, N)
        Base sorted indices (shared across all trees, not modified).
    cant_split_base : bool 2D array, shape (D, N)
        Base cant_split array (will be copied for each tree).
    n_missing : int32 1D array, shape (D,)
        Number of missing values per feature.
    max_depth : int
        Maximum depth of trees.
    n_estimators : int
        Number of trees to build.
    bootstrap_indices_array : int32 2D array, shape (n_estimators, max_bootstrap_size)
        Bootstrap sample indices for each tree (padded with -1).
    bootstrap_sizes : int32 1D array, shape (n_estimators,)
        Actual size of bootstrap sample for each tree.

    Returns
    -------
    all_node_features : int32 2D array
    all_node_thresholds : float32 2D array
    all_node_missing_go_left : bool 2D array
    all_node_values : float64 2D array
    all_left_children : int32 2D array
    all_right_children : int32 2D array
    all_node_counts : int32 1D array
    """
    n_features, n_samples = X.shape

    # Estimate max nodes per tree
    max_nodes_per_tree = _compute_max_nodes(n_samples, max_depth)

    # Pre-allocate arrays for all trees
    all_node_features = np.empty((n_estimators, max_nodes_per_tree), dtype=np.int32)
    all_node_thresholds = np.empty((n_estimators, max_nodes_per_tree), dtype=np.float32)
    all_node_missing_go_left = np.empty(
        (n_estimators, max_nodes_per_tree), dtype=np.bool_
    )
    all_node_values = np.empty((n_estimators, max_nodes_per_tree), dtype=y.dtype)
    all_left_children = np.empty((n_estimators, max_nodes_per_tree), dtype=np.int32)
    all_right_children = np.empty((n_estimators, max_nodes_per_tree), dtype=np.int32)
    all_node_counts = np.empty(n_estimators, dtype=np.int32)

    for tree_idx in prange(n_estimators):
        if n_bootstrap > 0:
            sample_weights = _perform_bootstrap(n_bootstrap, n_samples)
        else:
            sample_weights = None

        # Build tree using filtered data with sample weights
        (
            node_feature,
            node_threshold,
            node_missing_go_left,
            node_value,
            left_child,
            right_child,
            node_count,
        ) = _build_tree_numba(
            X,
            y,
            sample_weights,
            sorted_idx.copy(),
            cant_split.copy(),
            n_missing,
            max_depth,
        )

        # Store tree data
        all_node_counts[tree_idx] = node_count
        for i in range(node_count):
            all_node_features[tree_idx, i] = node_feature[i]
            all_node_thresholds[tree_idx, i] = node_threshold[i]
            all_node_missing_go_left[tree_idx, i] = node_missing_go_left[i]
            all_node_values[tree_idx, i] = node_value[i]
            all_left_children[tree_idx, i] = left_child[i]
            all_right_children[tree_idx, i] = right_child[i]

    return (
        all_node_features,
        all_node_thresholds,
        all_node_missing_go_left,
        all_node_values,
        all_left_children,
        all_right_children,
        all_node_counts,
    )


@njit
def _perform_bootstrap(
    n_bootstrap: int,
    n_samples: int,
):
    # Convert bootstrap indices to sample weights
    # Count occurrences of each sample index in the bootstrap sample
    sample_weights = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = np.random.choice(n_samples)
        sample_weights[idx] += 1.0

    return sample_weights


class Forest:
    """
    Random Forest Regressor using presorted features and numba parallelization.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        The maximum depth of the trees. If None, defaults to 100.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    max_samples : int or float, default=None
        The number of samples to draw for each tree:
        - If int, draw `max_samples` samples.
        - If float, draw `max_samples * n_samples` samples.
        - If None, draw `n_samples` samples (100%).
    random_state : int or None, default=None
        Random state for reproducibility.
    n_jobs : int or None, default=None
        The number of threads to use for parallel tree building:
        - If None or -1, use all available threads.
        - If int > 0, use that many threads.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        bootstrap=True,
        max_samples=None,
        random_state=None,
        n_jobs=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _get_n_samples_bootstrap(self, n_samples):
        """
        Get the number of samples in a bootstrap sample.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        n_samples_bootstrap : int
            The total number of samples to draw for the bootstrap sample.
        """
        if self.max_samples is None:
            return n_samples

        if isinstance(self.max_samples, int):
            return self.max_samples

        if isinstance(self.max_samples, float):
            return max(round(n_samples * self.max_samples), 1)

        raise TypeError()

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Preprocess data
        temp_tree = DecisionTreeRegressor(max_depth=self.max_depth)
        from time import perf_counter

        t = perf_counter()
        X, y, sorted_idx_base, n_missing, cant_split_base, _ = temp_tree.preprocess_Xy(
            X, y
        )
        print("presort time", perf_counter() - t)

        _, n_samples = X.shape

        # Generate bootstrap indices for each tree
        if self.bootstrap:
            n_samples_bootstrap = self._get_n_samples_bootstrap(n_samples)
        else:
            n_samples_bootstrap = 0

        # Set max_depth for numba function
        if self.max_depth is None:
            max_depth_int = -1
        else:
            max_depth_int = int(self.max_depth)

        # Determine number of threads
        n_threads = self.n_jobs
        if n_threads is None or n_threads == -1:
            import os

            n_threads = os.cpu_count()

        # Build trees in parallel using numba
        set_num_threads(n_threads)
        (
            all_node_features,
            all_node_thresholds,
            all_node_missing_go_left,
            all_node_values,
            all_left_children,
            all_right_children,
            all_node_counts,
        ) = _parallel_build_trees_numba(
            X,
            y,
            sorted_idx_base,
            cant_split_base,
            n_missing,
            max_depth_int,
            self.n_estimators,
            n_samples_bootstrap,
        )

        # Store tree data
        self.estimators_ = []
        for i in range(self.n_estimators):
            node_count = all_node_counts[i]

            # Create a tree object and set its attributes
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.node_feature = all_node_features[i, :node_count].copy()
            tree.node_threshold = all_node_thresholds[i, :node_count].copy()
            tree.node_missing_go_left = all_node_missing_go_left[i, :node_count].copy()
            tree.node_value = all_node_values[i, :node_count].copy()
            tree.left_child = all_left_children[i, :node_count].copy()
            tree.right_child = all_right_children[i, :node_count].copy()
            tree.n_nodes_ = node_count
            tree._fitted = True

            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = np.asarray(X, dtype=np.float32)

        # Accumulate predictions from all trees
        y_hat = np.zeros(X.shape[0], dtype=np.float64)

        for tree in self.estimators_:
            y_hat += tree.predict(X)

        y_hat /= len(self.estimators_)

        return y_hat
