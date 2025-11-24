# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numba import njit, prange

from sklearn.presortedtree.dt import DecisionTreeRegressor, _build_tree_numba

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Generate bootstrap sample indices.

    Parameters
    ----------
    random_state : int or RandomState
        Random state for reproducibility.
    n_samples : int
        Number of samples in the dataset.
    n_samples_bootstrap : int
        Number of samples to draw for the bootstrap sample.

    Returns
    -------
    sample_indices : ndarray of shape (n_samples_bootstrap,)
        The sampled indices (with replacement).
    """
    rng = np.random.RandomState(random_state)
    sample_indices = rng.randint(0, n_samples, n_samples_bootstrap, dtype=np.int32)
    return sample_indices


@njit(parallel=True)
def _parallel_build_trees_numba(
    X,
    y,
    sorted_idx_base,
    cant_split_base,
    n_missing,
    max_depth,
    n_estimators,
    bootstrap_indices_array,
    bootstrap_sizes,
):
    """
    Build multiple trees in parallel using numba prange.

    Parameters
    ----------
    X : float32 2D array, shape (D, N)
    y : float64 1D array, shape (N,)
    sorted_idx_base : int32 2D array, shape (D, N)
        Base sorted indices (will be copied for each tree).
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
    all_node_features : list of int32 arrays
    all_node_thresholds : list of float32 arrays
    all_node_missing_go_left : list of bool arrays
    all_node_values : list of float64 arrays
    all_left_children : list of int32 arrays
    all_right_children : list of int32 arrays
    all_node_counts : int32 1D array
    """
    n_features, n_samples = X.shape

    # Estimate max nodes per tree
    max_nodes_per_tree = 2 * n_samples - 1

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
        # Get bootstrap indices for this tree
        n_bootstrap = bootstrap_sizes[tree_idx]

        # Create bootstrapped dataset
        X_boot = np.empty((n_features, n_bootstrap), dtype=X.dtype)
        y_boot = np.empty(n_bootstrap, dtype=y.dtype)

        for i in range(n_bootstrap):
            idx = bootstrap_indices_array[tree_idx, i]
            y_boot[i] = y[idx]
            for f in range(n_features):
                X_boot[f, i] = X[f, idx]

        # Create sorted indices for bootstrapped data
        sorted_idx = np.empty((n_features, n_bootstrap), dtype=np.int32)
        cant_split = np.empty((n_features, n_bootstrap), dtype=np.bool_)

        for f in range(n_features):
            # Sort by feature values
            feature_values = X_boot[f, :]
            # Use argsort with kind='mergesort' for stability, then cast to int32
            temp_sorted = np.argsort(feature_values, kind="mergesort")
            for i in range(n_bootstrap):
                sorted_idx[f, i] = temp_sorted[i]

            # Mark positions where we can't split (duplicate values)
            for i in range(n_bootstrap - 1):
                idx_curr = sorted_idx[f, i]
                idx_next = sorted_idx[f, i + 1]
                cant_split[f, i] = feature_values[idx_curr] == feature_values[idx_next]
            cant_split[f, n_bootstrap - 1] = True

        # Count missing values in bootstrapped data
        n_missing_boot = np.zeros(n_features, dtype=np.int32)
        for f in range(n_features):
            for i in range(n_bootstrap):
                if np.isnan(X_boot[f, i]):
                    n_missing_boot[f] += 1

        # Build tree
        (
            node_feature,
            node_threshold,
            node_missing_go_left,
            node_value,
            left_child,
            right_child,
            node_count,
        ) = _build_tree_numba(
            X_boot, y_boot, sorted_idx, cant_split, n_missing_boot, max_depth
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
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        bootstrap=True,
        max_samples=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

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
            if self.max_samples > n_samples:
                raise ValueError(
                    f"`max_samples` must be <= n_samples={n_samples} "
                    f"but got value {self.max_samples}"
                )
            return self.max_samples

        if isinstance(self.max_samples, float):
            return max(round(n_samples * self.max_samples), 1)

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
        X, y, sorted_idx_base, n_missing, cant_split_base = temp_tree.preprocess_Xy(
            X, y
        )

        n_features, n_samples = X.shape

        # Setup random state
        if self.random_state is None:
            rng = np.random.RandomState()
        else:
            rng = np.random.RandomState(self.random_state)

        # Generate bootstrap indices for each tree
        if self.bootstrap:
            n_samples_bootstrap = self._get_n_samples_bootstrap(n_samples)
        else:
            n_samples_bootstrap = n_samples

        # Create a 2D array to hold all bootstrap indices (padded with -1 if needed)
        bootstrap_indices_array = np.empty(
            (self.n_estimators, n_samples_bootstrap), dtype=np.int32
        )
        bootstrap_sizes = np.full(
            self.n_estimators, n_samples_bootstrap, dtype=np.int32
        )

        for i in range(self.n_estimators):
            if self.bootstrap:
                # Use a different seed for each tree
                tree_seed = rng.randint(0, MAX_INT)
                indices = _generate_sample_indices(
                    tree_seed, n_samples, n_samples_bootstrap
                )
            else:
                # No bootstrap: use all samples for each tree
                indices = np.arange(n_samples, dtype=np.int32)

            bootstrap_indices_array[i, :] = indices

        # Set max_depth for numba function
        if self.max_depth is None:
            max_depth_int = -1
        else:
            max_depth_int = int(self.max_depth)

        # Build trees in parallel using numba
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
            bootstrap_indices_array,
            bootstrap_sizes,
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
