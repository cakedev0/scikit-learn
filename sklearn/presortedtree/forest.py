# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numba import njit, prange, set_num_threads

from sklearn.presortedtree.criterion import MSE, Gini
from sklearn.presortedtree.dt import DecisionTree, _build_tree_numba


@njit(parallel=True)
def _parallel_build_trees_numba(
    X,
    y,
    sorted_idx,
    cant_split,
    n_missing,
    max_depth,
    n_bootstrap,
    min_impurity_decrease,
    criteria,
    all_node_features,
    all_node_thresholds,
    all_node_missing_go_left,
    all_node_values,
    all_left_children,
    all_right_children,
    all_node_counts,
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
    sorted_idx : int32 2D array, shape (D, N)
        Base sorted indices (shared across all trees, not modified).
    cant_split : bool 2D array, shape (D, N)
        Base cant_split array (will be copied for each tree).
    n_missing : int32 1D array, shape (D,)
        Number of missing values per feature.
    max_depth : int
        Maximum depth of trees.
    n_bootstrap : int
        Number of bootstrap samples per tree (0 means no bootstrap).
    min_impurity_decrease : float
        Minimum impurity decrease required for a split.
    criteria : list of Criterion objects
        Pre-instantiated criterion objects, one per tree.
    all_node_features : int32 2D array (pre-allocated)
    all_node_thresholds : float32 2D array (pre-allocated)
    all_node_missing_go_left : bool 2D array (pre-allocated)
    all_node_values : float64 2D array (pre-allocated)
    all_left_children : int32 2D array (pre-allocated)
    all_right_children : int32 2D array (pre-allocated)
    all_node_counts : int32 1D array (pre-allocated)

    Returns
    -------
    None (modifies arrays in-place)
    """
    n_features, n_samples = X.shape
    n_estimators = all_node_features.shape[0]

    for tree_idx in prange(n_estimators):
        if n_bootstrap > 0:
            sample_weights = _perform_bootstrap(n_bootstrap, n_samples)
        else:
            sample_weights = None

        # Get pre-allocated arrays for this tree
        node_feature = all_node_features[tree_idx, :]
        node_threshold = all_node_thresholds[tree_idx, :]
        node_missing_go_left = all_node_missing_go_left[tree_idx, :]
        node_value = all_node_values[tree_idx, :]
        left_child = all_left_children[tree_idx, :]
        right_child = all_right_children[tree_idx, :]

        # Get pre-instantiated criterion for this tree
        criterion = criteria[tree_idx]

        # Build tree using filtered data with sample weights
        node_count = _build_tree_numba(
            X,
            y,
            sample_weights,
            sorted_idx.copy(),
            cant_split.copy(),
            n_missing,
            max_depth,
            criterion,
            min_impurity_decrease,
            node_feature,
            node_threshold,
            node_missing_go_left,
            left_child,
            right_child,
            node_value,
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
        min_impurity_decrease=0.0,
        criterion="squared_error",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion

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
        temp_tree = DecisionTree(max_depth=self.max_depth)

        X, y, sorted_idx_base, n_missing, cant_split_base, _ = temp_tree.preprocess_Xy(
            X, y
        )

        _, n_samples = X.shape
        max_nodes_per_tree = temp_tree.get_max_n_nodes(n_samples)

        # Determine if classification or regression
        is_clf = self.criterion == "gini"
        if is_clf:
            y = y.astype("uint8")
            n_classes = int(y.max()) + 1
            value_shape = (max_nodes_per_tree, n_classes)
        else:
            value_shape = (max_nodes_per_tree,)

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

        # Estimate max nodes per tree

        # Pre-allocate arrays for all trees
        shape = (self.n_estimators, max_nodes_per_tree)
        all_node_features = np.empty(shape, dtype=np.int32)
        all_node_thresholds = np.empty(shape, dtype=np.float32)
        all_node_missing_go_left = np.empty(shape, dtype=np.bool_)
        all_node_values = np.empty((self.n_estimators, *value_shape), dtype=np.float64)
        all_left_children = np.empty(shape, dtype=np.int32)
        all_right_children = np.empty(shape, dtype=np.int32)
        all_node_counts = np.empty(self.n_estimators, dtype=np.int32)

        # Initialize arrays
        all_node_features[:] = -1
        all_left_children[:] = -1
        all_right_children[:] = -1

        # Create criteria list - one per tree
        if is_clf:
            criteria = [Gini(n_classes) for _ in range(self.n_estimators)]
        else:
            criteria = [MSE() for _ in range(self.n_estimators)]

        # Build trees in parallel using numba
        set_num_threads(n_threads)
        _parallel_build_trees_numba(
            X,
            y,
            sorted_idx_base,
            cant_split_base,
            n_missing,
            max_depth_int,
            n_samples_bootstrap,
            self.min_impurity_decrease,
            criteria,
            all_node_features,
            all_node_thresholds,
            all_node_missing_go_left,
            all_node_values,
            all_left_children,
            all_right_children,
            all_node_counts,
        )

        # Store tree data
        self.estimators_ = []
        for i in range(self.n_estimators):
            node_count = all_node_counts[i]

            # Create a tree object and set its attributes
            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion)
            tree.node_feature = all_node_features[i, :node_count].copy()
            tree.node_threshold = all_node_thresholds[i, :node_count].copy()
            tree.node_missing_go_left = all_node_missing_go_left[i, :node_count].copy()
            tree.node_value = all_node_values[i, :node_count].copy()
            tree.left_child = all_left_children[i, :node_count].copy()
            tree.right_child = all_right_children[i, :node_count].copy()
            tree.n_nodes_ = node_count
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict target for X.

        For regression, the predicted target is computed as the mean of the
        predictions from all trees.
        For classification, the predicted class is the one with highest average
        probability across all trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values (regression) or class labels (classification).
        """
        if self.criterion == "gini":
            # For classification, use predict_proba and take argmax
            return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)
        else:
            # For regression, accumulate predictions
            X = np.asarray(X, dtype=np.float32)
            y_hat = np.zeros(X.shape[0], dtype=np.float64)
            for tree in self.estimators_:
                y_hat += tree.predict(X)
            y_hat /= len(self.estimators_)
            return y_hat

    def predict_proba(self, X):
        """Predict class probabilities for X.

        For classification, returns the mean predicted class probabilities
        of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the fitted attribute classes_.

        Raises
        ------
        ValueError
            If the forest was not fitted with a classification criterion.
        """
        if self.criterion != "gini":
            raise ValueError(
                "predict_proba is only available for classification. "
                "This forest was fitted with criterion='{}'".format(self.criterion)
            )

        X = np.asarray(X, dtype=np.float32)

        # Get first tree's prediction to determine n_classes
        first_proba = self.estimators_[0].predict_proba(X)
        n_classes = first_proba.shape[1]
        y_proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)

        # Accumulate probabilities from all trees
        for tree in self.estimators_:
            y_proba += tree.predict_proba(X)

        y_proba /= len(self.estimators_)
        return y_proba
