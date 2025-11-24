# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numba import njit

# ============================================================
# Numba core: tree building
# ============================================================


@njit
def _build_tree_numba(
    X: np.ndarray,
    y: np.ndarray,
    sorted_idx: np.ndarray,
    cant_split: np.ndarray,
    n_missing_per_feature: np.ndarray,
    max_depth: int,
):
    """
    Build a regression tree minimizing MSE.

    Parameters
    ----------
    X : float32 2D array, shape (D, N)
    y : float32 1D array, shape (N,)
    sorted_idx : int32 2D array, shape (D, N)
        For each feature f, sorted_idx[f] is a permutation of [0..N-1]
        such that X[sorted_idx[f, :], f] is non-decreasing.
        This array WILL be modified in-place during tree building.
    max_depth : int
        -1 means "no limit". Otherwise, maximum depth of tree.

    Returns
    -------
    node_feature : int32 1D array (size <= 2*N)
    node_threshold : float32 1D array
    node_value : float32 1D array
    left_child : int32 1D array
    right_child : int32 1D array
    node_count : int
    """
    n_features, n_samples = X.shape
    if max_depth == -1:
        max_nodes = 2 * n_samples - 1
    else:
        max_nodes = 1
        n_nodes_at_d = 2
        for _ in range(max_depth):
            max_nodes += n_nodes_at_d
            n_nodes_at_d *= 2
            if n_nodes_at_d > 4 * n_samples:
                break
        max_nodes = min(2 * n_samples - 1, max_nodes)

    # Tree arrays
    node_feature = -1 * np.ones(max_nodes, dtype=np.int32)  # -1 => leaf
    node_threshold = np.zeros(max_nodes, dtype=np.float32)
    node_missing_go_left = np.zeros(max_nodes, dtype=np.bool_)
    node_value = np.zeros(max_nodes, dtype=y.dtype)
    left_child = -1 * np.ones(max_nodes, dtype=np.int32)
    right_child = -1 * np.ones(max_nodes, dtype=np.int32)

    # Stack for iterative DFS: node index + depth
    max_stack_size = max_depth + 1
    stack_node = np.empty(max_stack_size, dtype=np.int32)
    stack_depth = np.empty(max_stack_size, dtype=np.int32)
    stack_seg_start = np.empty(max_stack_size, dtype=np.int32)
    stack_seg_len = np.empty(max_stack_size, dtype=np.int32)
    stack_n_missing = np.empty((max_stack_size, n_features), dtype=np.int32)
    stack_size = 0

    # Temp buffers reused for all nodes/features
    idx_buf = np.empty(n_samples, dtype=sorted_idx.dtype)  # for stable partitioning
    ranks_buf = np.empty(n_samples, dtype=np.int32)  # for stable partitioning
    go_left = np.empty(n_samples, dtype=np.bool_)  # for stable partitioning
    ns_missing_left = np.empty(n_features, dtype=np.int32)

    # Push root to the stack
    node_count = 1
    root = 0  # Initialize root node (id = 0)
    stack_node[0] = root
    stack_depth[0] = 0
    stack_seg_start[0] = 0
    stack_seg_len[0] = n_samples
    stack_n_missing[0] = n_missing_per_feature

    stack_size = 1

    # Main loop
    while stack_size > 0:
        stack_size -= 1
        node = stack_node[stack_size]
        depth = stack_depth[stack_size]
        s = stack_seg_start[stack_size]
        n = stack_seg_len[stack_size]
        e = s + n
        ns_missing = stack_n_missing[stack_size]

        # Compute mean and variance of y in this node (using feature 0's segment)
        if n == 0:
            # This should not really happen, but just in case
            node_feature[node] = -1
            node_value[node] = 0.0
            continue

        sum_y = 0.0
        sum_y2 = 0.0
        for j in range(s, e):
            idx = sorted_idx[0, j]
            val = y[idx]
            sum_y += val
            sum_y2 += val * val
        mean_y = sum_y / n
        var_y = sum_y2 / n - mean_y * mean_y

        # Leaf conditions
        leaf = False
        if max_depth >= 0 and depth >= max_depth:
            leaf = True
        elif n <= 1:
            leaf = True
        elif var_y <= 1e-12:
            leaf = True

        if leaf:
            node_feature[node] = -1  # leaf
            node_value[node] = mean_y
            continue

        # --------------------------------------------------------
        # Find best split over all features
        # --------------------------------------------------------
        best_sse = np.inf
        best_feature = -1
        best_j = -1
        best_missing_go_left = False

        for f in range(n_features):
            n_missing = ns_missing[f]

            # Scan split positions with missing on the right:
            left_sum = 0.0
            right_sum = 0.0
            left_n = 0
            for j in range(s, e - max(1, n_missing)):
                idx = sorted_idx[f, j]
                left_sum += y[idx]  # sparse/random reads
                left_n += 1

                if cant_split[f, j]:
                    continue

                right_n = n - left_n
                right_sum = sum_y - left_sum

                # SSE = sum(y^2) - sum(y)^2 / n
                sse = (
                    sum_y2
                    - (left_sum * left_sum) / left_n
                    - (right_sum * right_sum) / right_n
                )

                if sse < best_sse:
                    best_sse = sse
                    best_feature = f
                    best_j = j
                    best_missing_go_left = False

            if n_missing == 0:
                continue

            # Scan split positions with missing on the left
            left_sum = right_sum
            left_n = n_missing
            for j in range(s, e - n_missing - 1):
                idx = sorted_idx[f, j]
                left_sum += y[idx]  # sparse/random reads
                left_n += 1

                if cant_split[f, j]:
                    continue

                right_n = n - left_n
                right_sum = sum_y - left_sum

                # SSE = sum(y^2) - sum(y)^2 / n
                sse = (
                    sum_y2
                    - (left_sum * left_sum) / left_n
                    - (right_sum * right_sum) / right_n
                )

                if sse < best_sse:
                    best_sse = sse
                    best_feature = f
                    best_j = j
                    best_missing_go_left = True

        # If no valid split found -> leaf
        if best_feature == -1:
            node_feature[node] = -1
            node_value[node] = mean_y
            continue

        # --------------------------------------------------------
        # Create children
        # --------------------------------------------------------
        left_id = node_count
        right_id = node_count + 1
        node_count += 2
        n_missing = ns_missing[best_feature]

        left_child[node] = left_id
        right_child[node] = right_id

        node_feature[node] = best_feature
        node_value[node] = mean_y  # not used for prediction, but can store parent mean
        node_missing_go_left[node] = best_missing_go_left

        if best_j == e - n_missing - 1:
            best_threshold = np.inf
        else:
            x_left = X[best_feature, sorted_idx[best_feature, best_j]]
            x_right = X[best_feature, sorted_idx[best_feature, best_j + 1]]
            best_threshold = (x_left + x_right) / 2
        node_threshold[node] = best_threshold

        # --------------------------------------------------------
        # Partition all feature segments into left/right children
        # Stable partition so children segments remain sorted by feature.
        # --------------------------------------------------------

        n_left = 0

        for j in range(s, e - n_missing):
            idx = sorted_idx[best_feature, j]
            gl = j <= best_j
            go_left[idx] = gl  # sparse/random writes
            n_left += gl
        for j in range(e - n_missing, e):
            idx = sorted_idx[best_feature, j]
            go_left[idx] = best_missing_go_left
        if best_missing_go_left:
            n_left += n_missing

        for f in range(n_features):
            # Stable partition using temporary buffer
            n_missing_left = 0
            end_non_missing = e - ns_missing[f]
            left_ptr = s
            right_ptr = s + n_left
            rank = 0
            for i in range(s, e):
                idx = sorted_idx[f, i]
                if go_left[idx]:  # sparse/random reads
                    sorted_idx[f, left_ptr] = idx
                    ranks_buf[left_ptr] = rank
                    left_ptr += 1
                    n_missing_left += i >= end_non_missing
                else:
                    idx_buf[right_ptr] = idx
                    ranks_buf[right_ptr] = rank
                    right_ptr += 1
                if not cant_split[f, i]:
                    rank += 1

            ns_missing_left[f] = n_missing_left

            # Copy back from buffer
            for i in range(s + n_left, e):
                sorted_idx[f, i] = idx_buf[i]

            for i in range(s, e - 1):
                cant_split[f, i] = ranks_buf[i] == ranks_buf[i + 1]

        # --------------------------------------------------------
        # Push children to stack (DFS)
        # --------------------------------------------------------
        # Right then left => left processed first (if you want)
        stack_node[stack_size] = right_id
        stack_depth[stack_size] = depth + 1
        stack_seg_start[stack_size] = s + n_left
        stack_seg_len[stack_size] = n - n_left
        stack_n_missing[stack_size, :] = ns_missing - ns_missing_left
        stack_size += 1

        stack_node[stack_size] = left_id
        stack_depth[stack_size] = depth + 1
        stack_seg_start[stack_size] = s
        stack_seg_len[stack_size] = n_left
        stack_n_missing[stack_size, :] = ns_missing_left
        stack_size += 1

    # Done, but leaf nodes don't all have node_value yet (we set mean at leaf creation).
    # Actually we already set node_value for leaves.

    return (
        node_feature,
        node_threshold,
        node_missing_go_left,
        node_value,
        left_child,
        right_child,
        node_count,
    )


# ============================================================
# Numba core: prediction
# ============================================================


@njit
def _predict_numba(
    X,
    node_feature,
    node_threshold,
    node_missing_go_left,
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
            if np.isnan(v):
                go_left = node_missing_go_left[node]
            else:
                go_left = v <= node_threshold[node]

            if go_left:
                node = left_child[node]
            else:
                node = right_child[node]

    return preds


# ============================================================
# Python wrapper class
# ============================================================


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        """
        Parameters
        ----------
        max_depth : int or None
            Maximum depth of the tree. If None, no depth limit.
        """
        self.max_depth = max_depth or 100
        self._fitted = False

    def preprocess_Xy(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        sorted_indices = np.argsort(y)
        y = y[sorted_indices]
        X = X[sorted_indices]

        # Global sorted indices per feature
        # sorted_idx[f] is sorted order of samples by X[:, f]
        X = np.asarray(X, dtype=np.float32, order="F").T  # shape (D, N)
        sorted_idx = np.empty_like(X, dtype=np.int32, order="C")
        cant_split = np.empty_like(sorted_idx, dtype=bool)

        for f, feature_values in enumerate(X):
            sorted_idx[f, :] = np.argsort(feature_values, stable=True)
            v_sorted = feature_values[sorted_idx[f, :]]
            cant_split[f, -1] = True
            cant_split[f, :-1] = np.diff(v_sorted) == 0

        n_missing = np.isnan(X).sum(axis=1)

        return X, y, sorted_idx, n_missing, cant_split

    def fit(self, X, y, sorted_idx=None, n_missing=None, cant_split=None):
        if sorted_idx is None:
            X, y, sorted_idx, n_missing, cant_split = self.preprocess_Xy(X, y)

        if self.max_depth is None:
            max_depth_int = -1
        else:
            max_depth_int = int(self.max_depth)

        (
            node_feature,
            node_threshold,
            node_missing_go_left,
            node_value,
            left_child,
            right_child,
            node_count,
        ) = _build_tree_numba(X, y, sorted_idx, cant_split, n_missing, max_depth_int)

        # Truncate to actual node_count to keep things clean
        self.node_feature = node_feature[:node_count]
        self.node_threshold = node_threshold[:node_count]
        self.node_missing_go_left = node_missing_go_left[:node_count]
        self.node_value = node_value[:node_count]
        self.left_child = left_child[:node_count]
        self.right_child = right_child[:node_count]

        self.n_nodes_ = node_count
        self._X_dtype = X.dtype
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("DecisionTreeRegressor is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        return _predict_numba(
            X,
            self.node_feature,
            self.node_threshold,
            self.node_missing_go_left,
            self.node_value,
            self.left_child,
            self.right_child,
        )
