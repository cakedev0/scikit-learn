# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numba import njit

# ============================================================
# Binning utilities (non-numba, called once per tree)
# ============================================================

EPS = np.finfo("double").eps


def _preprocess_X_binning(X, n_bins=256):
    assert n_bins <= 256
    X = np.asarray(X, dtype=np.float32, order="F", copy=True).T
    n_features, n_samples = X.shape
    X_binned = np.empty(X.shape, dtype=np.uint8)
    constant_bins = np.zeros((n_features, n_bins), dtype=bool)
    n_bins_per_feature = np.empty(n_features, dtype=np.int32)

    for f in range(n_features):
        x = X[f, :]
        x_u, idx, counts = np.unique(x, return_inverse=True, return_counts=True)
        nu = x_u.size
        if nu <= n_bins:
            X_binned[f, :] = idx
            # Mark bins with single unique value as constant
            constant_bins[f, :nu] = True
            n_bins_per_feature[f] = x_u.size
            continue

        bin_edges = optimal_bin_edges(counts, n_samples, n_bins)
        assert bin_edges.size <= n_bins + 1
        X_binned[f, :] = np.searchsorted(bin_edges, idx, side="right") - 1
        nb = bin_edges.size - 1
        n_bins_per_feature[f] = nb
        constant_bins[f, :nb] = (bin_edges[:-1] + 1) == bin_edges[1:]

    return X, X_binned, n_bins_per_feature, constant_bins


def optimal_bin_edges(counts: np.ndarray, n_samples: int, n_bins: int):
    nu = counts.size
    sorted_indices = np.repeat(np.arange(nu), counts)
    min_step = max(10, n_samples // (n_bins * 10))
    max_step = (n_samples + n_bins - 1) // n_bins
    while min_step + 1 < max_step:
        step = (min_step + max_step) // 2
        nb = np.unique(sorted_indices[step::step]).size
        if sorted_indices[step] != 0:
            nb += 1
        if nb > n_bins:
            min_step = step
        else:
            max_step = step
    step = max_step
    return np.unique(np.r_[0, sorted_indices[step::step], nu])


# ============================================================
# Numba core: tree building with MSE (simplified, no indices)
# ============================================================


@njit
def _sort_by_bin_then_refine(
    x,
    x_binned,
    y,
    x_buf,
    y_buf,
    constant_bins,
    n_bins,
    s,
    e,
):
    # Count samples per bin
    bin_ptrs = np.zeros(n_bins + 1, dtype=np.int32)
    for i in range(s, e):
        bin_idx = x_binned[i]
        bin_ptrs[bin_idx + 1] += 1

    # Cumsum:
    pos = 0
    for b in range(1, n_bins + 1):
        pos += bin_ptrs[b]
        bin_ptrs[b] = pos

    # Place values into bins in buffers
    bin_positions = bin_ptrs.copy()
    for i in range(s, e):
        bin_idx = x_binned[i]
        dest = bin_positions[bin_idx]
        x_buf[dest] = x[i]
        y_buf[dest] = y[i]
        bin_positions[bin_idx] += 1

    # Now sort within each non-constant bin
    for b in range(n_bins):
        if constant_bins[b]:
            continue

        start = bin_ptrs[b]
        end = bin_ptrs[b + 1]

        if end - start <= 1:
            continue

        # Sort only this bin's range
        sorter = np.argsort(x_buf[start:end])
        x_buf[start:end] = x_buf[start:end][sorter]
        y_buf[start:end] = y_buf[start:end][sorter]


@njit
def _find_best_split_mse(
    X,
    X_binned,
    y,
    s,
    e,
    constant_bins,
    n_bins,
    x_buf,
    y_buf,
):
    """
    Find best split across all features using MSE criterion.
    Works on contiguous slice [s:e] of X, X_binned, y.
    """
    n_samples = e - s
    n_features = n_bins.size

    # Compute total statistics for the node
    sum_total = 0.0
    for i in range(s, e):
        sum_total += y[i]

    best_improvement = -np.inf
    best_feature = -1
    best_threshold = 0.0

    for f in range(n_features):
        # Sort buffers by this feature
        _sort_by_bin_then_refine(
            X[f, :],
            X_binned[f, :],
            y,
            x_buf,  # Will be overwritten
            y_buf,  # Will be overwritten
            constant_bins[f, :],
            n_bins[f],
            s,
            e,
        )

        # Scan for best split in sorted buffers
        sum_left = 0.0
        for j in range(n_samples - 1):
            sum_left += y_buf[j]

            # Check if we can split here (next value is different)
            if x_buf[j] == x_buf[j + 1]:
                continue

            # Compute improvement: sum_left^2/n_left + sum_right^2/n_right
            n_left = j + 1
            n_right = n_samples - n_left
            sum_right = sum_total - sum_left

            improvement = (sum_left * sum_left) / n_left + (
                sum_right * sum_right
            ) / n_right

            if improvement > best_improvement:
                best_improvement = improvement
                best_feature = f
                best_threshold = (x_buf[j] + x_buf[j + 1]) / 2.0

    return best_feature, best_threshold, best_improvement


@njit
def _partition_inplace(
    X,
    X_binned,
    y,
    s,
    e,
    split_feature,
    split_threshold,
):
    """
    Partition X, X_binned, and y in-place based on split threshold.
    Works on slice [s:e].

    Parameters
    ----------
    X : float32 2D array
        Original feature values (entire dataset) - shape (n_features, n_samples)
    X_binned : uint8 2D array
        Binned feature values (entire dataset) - shape (n_features, n_samples)
    y : float64 1D array
        Target values (entire dataset)
    s : int
        Start index
    e : int
        End index
    split_feature : int
        Feature to split on
    split_threshold : float
        Threshold value

    Returns
    -------
    n_left : int
        Number of samples going left
    """
    left = s
    right = e - 1

    while left <= right:
        val = X[split_feature, left]

        if val <= split_threshold:
            left += 1
        else:
            # Swap with right
            for f in range(X.shape[0]):
                X[f, left], X[f, right] = X[f, right], X[f, left]
                X_binned[f, left], X_binned[f, right] = (
                    X_binned[f, right],
                    X_binned[f, left],
                )
            y[left], y[right] = y[right], y[left]
            right -= 1

    return left - s


@njit
def _build_tree_numba(
    X: np.ndarray,
    X_binned: np.ndarray,
    y: np.ndarray,
    constant_bins: np.ndarray,
    n_bins: np.ndarray,
    max_depth: int,
    min_impurity_decrease: float,
    # out:
    node_feature,
    node_threshold,
    left_child,
    right_child,
    node_value,
):
    """

    Returns
    -------
    node_count : int
        Total number of nodes created
    """
    n_features, n_samples = X.shape

    # Buffers for sorting
    x_buf = np.empty(n_samples, dtype=X.dtype)
    y_buf = np.empty(n_samples, dtype=y.dtype)

    # Stack for DFS
    max_stack_size = max(max_depth + 1, 100)
    stack_node = np.empty(max_stack_size, dtype=np.int32)
    stack_depth = np.empty(max_stack_size, dtype=np.int32)
    stack_start = np.empty(max_stack_size, dtype=np.int32)
    stack_end = np.empty(max_stack_size, dtype=np.int32)
    stack_size = 0

    # Push root
    node_count = 1
    stack_node[0] = 0
    stack_depth[0] = 0
    stack_start[0] = 0
    stack_end[0] = n_samples
    stack_size = 1

    # Main loop
    while stack_size > 0:
        stack_size -= 1
        node = stack_node[stack_size]
        depth = stack_depth[stack_size]
        start = stack_start[stack_size]
        end = stack_end[stack_size]
        n = end - start

        if n == 0:
            node_feature[node] = -1
            node_value[node] = 0.0
            continue

        # Compute node mean and variance (contiguous access)
        sum_y = 0.0
        sum_y2 = 0.0
        for i in range(start, end):
            val = y[i]
            sum_y += val
            sum_y2 += val * val

        node_mean = sum_y / n
        node_variance = sum_y2 / n - node_mean * node_mean
        node_value[node] = node_mean

        # Check leaf conditions
        is_leaf = (
            (max_depth >= 0 and depth >= max_depth) or (node_variance < EPS) or (n <= 1)
        )

        if is_leaf:
            node_feature[node] = -1
            continue

        # Find best split (contiguous memory access)
        best_feat, best_thresh, best_imp = _find_best_split_mse(
            X,
            X_binned,
            y,
            start,
            end,
            constant_bins,
            n_bins,
            x_buf,
            y_buf,
        )

        normalized_imp = best_imp / n

        if best_feat == -1 or normalized_imp + EPS < min_impurity_decrease:
            node_feature[node] = -1
            continue

        # Partition data in-place
        n_left = _partition_inplace(
            X,
            X_binned,
            y,
            start,
            end,
            best_feat,
            best_thresh,
        )

        if n_left == 0 or n_left == n:
            # Degenerate split, make leaf
            node_feature[node] = -1
            continue

        # Create children
        left_id = node_count
        right_id = node_count + 1
        node_count += 2

        node_feature[node] = best_feat
        node_threshold[node] = best_thresh
        left_child[node] = left_id
        right_child[node] = right_id

        # Push children to stack (right then left for DFS)
        stack_node[stack_size] = right_id
        stack_depth[stack_size] = depth + 1
        stack_start[stack_size] = start + n_left
        stack_end[stack_size] = end
        stack_size += 1

        stack_node[stack_size] = left_id
        stack_depth[stack_size] = depth + 1
        stack_start[stack_size] = start
        stack_end[stack_size] = start + n_left
        stack_size += 1

    return node_count


# ============================================================
# Numba core: prediction
# ============================================================


@njit
def _predict_numba(
    X,
    node_feature,
    node_threshold,
    node_value,
    left_child,
    right_child,
):
    N, D = X.shape
    preds = np.empty(N, dtype=np.float32)

    for i in range(N):
        node = 0  # root
        while True:
            f = node_feature[node]
            if f == -1:  # leaf
                preds[i] = node_value[node]
                break
            v = X[i, f]

            if v <= node_threshold[node]:
                node = left_child[node]
            else:
                node = right_child[node]

    return preds


# ============================================================
# Python wrapper class
# ============================================================


class BinSortDecisionTree:
    def __init__(self, max_depth=None, min_impurity_decrease=0.0, n_bins=256):
        """
        Decision tree using binning and sorting approach (regression only, no missing values or weights).

        Parameters
        ----------
        max_depth : int or None
            Maximum depth of the tree. If None, defaults to 100.
        min_impurity_decrease : float
            Minimum impurity decrease required for a split.
        n_bins : int
            Number of bins for discretization (default 256)
        """
        self.max_depth = max_depth or 100
        self.min_impurity_decrease = min_impurity_decrease
        self.n_bins = n_bins

    def get_max_n_nodes(self, n_samples):
        if self.max_depth is None:
            return 2 * n_samples - 1

        max_nodes = 1
        n_nodes_at_d = 2
        for _ in range(self.max_depth):
            max_nodes += n_nodes_at_d
            n_nodes_at_d *= 2
        return min(2 * n_samples - 1, max_nodes)

    def preprocess_Xy(self, X, y, sample_weight=None):
        assert sample_weight is None

        X, X_binned, n_bins_per_feature, constant_bins = _preprocess_X_binning(X)
        y = np.asarray(y, dtype=np.float64, copy=True).ravel()
        return X, y, sample_weight, X_binned, n_bins_per_feature, constant_bins

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_binned=None,
        n_bins_per_feature=None,
        constant_bins=None,
    ):
        assert sample_weight is None
        # Preprocess
        if X_binned is None:
            X, y, sample_weight, X_binned, n_bins_per_feature, constant_bins = (
                self.preprocess_Xy(X, y)
            )

        n_samples = len(y)
        max_depth_int = -1 if self.max_depth is None else int(self.max_depth)
        max_n_nodes = self.get_max_n_nodes(n_samples)

        # Allocate tree arrays
        self.node_feature = -1 * np.ones(max_n_nodes, dtype=np.int32)
        self.node_threshold = np.zeros(max_n_nodes, dtype=np.float32)
        self.left_child = -1 * np.ones(max_n_nodes, dtype=np.int32)
        self.right_child = -1 * np.ones(max_n_nodes, dtype=np.int32)
        self.node_value = np.empty(max_n_nodes, dtype=np.float64)

        # Build tree (X, X_binned, y will be modified in-place)
        node_count = _build_tree_numba(
            X,
            X_binned,
            y,
            constant_bins,
            n_bins_per_feature,
            max_depth_int,
            self.min_impurity_decrease,
            self.node_feature,
            self.node_threshold,
            self.left_child,
            self.right_child,
            self.node_value,
        )

        # Truncate arrays
        self.node_feature = self.node_feature[:node_count]
        self.node_threshold = self.node_threshold[:node_count]
        self.node_value = self.node_value[:node_count]
        self.left_child = self.left_child[:node_count]
        self.right_child = self.right_child[:node_count]

        self.n_nodes_ = node_count
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return _predict_numba(
            X,
            self.node_feature,
            self.node_threshold,
            self.node_value,
            self.left_child,
            self.right_child,
        )
