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
    D, N = X.shape
    if max_depth == -1:
        max_nodes = 2 * N - 1
    else:
        max_nodes = 1
        n_nodes_at_d = 2
        for _ in range(max_depth):
            max_nodes += n_nodes_at_d
            n_nodes_at_d *= 2

    # Tree arrays
    node_feature = -1 * np.ones(max_nodes, dtype=np.int32)  # -1 => leaf
    node_threshold = np.zeros(max_nodes, dtype=np.float32)
    node_value = np.zeros(max_nodes, dtype=y.dtype)
    left_child = -1 * np.ones(max_nodes, dtype=np.int32)
    right_child = -1 * np.ones(max_nodes, dtype=np.int32)

    # For each node and feature: its segment [start, start+length) in sorted_idx[f]
    seg_start = np.zeros(max_nodes, dtype=np.int32)
    seg_len = np.zeros(max_nodes, dtype=np.int32)

    # Stack for iterative DFS: node index + depth
    stack_node = np.empty(max_nodes, dtype=np.int32)
    stack_depth = np.empty(max_nodes, dtype=np.int32)
    stack_size = 0

    # Temp buffers reused for all nodes/features
    idx_buf = np.empty(N, dtype=sorted_idx.dtype)  # for stable partitioning
    ranks_buf = np.empty(N, dtype=np.int32)  # for stable partitioning
    go_left = np.empty(N, dtype=np.bool_)  # for stable partitioning

    # Initialize root node (id = 0)
    node_count = 1
    root = 0
    seg_start[root] = 0
    seg_len[root] = N

    # Push root
    stack_node[0] = root
    stack_depth[0] = 0
    stack_size = 1

    # Main loop
    while stack_size > 0:
        stack_size -= 1
        node = stack_node[stack_size]
        depth = stack_depth[stack_size]

        # Use feature 0's segment to know how many samples are in this node
        s = seg_start[node]
        n = seg_len[node]

        # Compute mean and variance of y in this node (using feature 0's segment)
        if n == 0:
            # This should not really happen, but just in case
            node_feature[node] = -1
            node_value[node] = 0.0
            continue

        sum_y = 0.0
        sum_y2 = 0.0
        for i in range(n):
            idx = sorted_idx[0, s + i]
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
        best_i = -1

        for f in range(D):
            # Scan split positions
            left_sum = 0.0
            next_idx = sorted_idx[f, s]
            for i in range(n - 1):
                idx = next_idx
                left_sum += y[idx]

                j = s + i
                next_idx = sorted_idx[f, j + 1]
                if cant_split[f, j]:
                    continue

                left_n = i + 1
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
                    best_i = i

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

        node_feature[node] = best_feature
        node_value[node] = mean_y  # not used for prediction, but can store parent mean
        left_child[node] = left_id
        right_child[node] = right_id
        x_left = X[best_feature, sorted_idx[best_feature, best_i + s]]
        x_right = X[best_feature, sorted_idx[best_feature, best_i + s + 1]]
        best_threshold = (x_left + x_right) / 2
        node_threshold[node] = best_threshold

        # --------------------------------------------------------
        # Partition all feature segments into left/right children
        # Stable partition so children segments remain sorted by feature.
        # --------------------------------------------------------

        n_left = 0
        for i in range(n):
            idx = sorted_idx[best_feature, s + i]
            gl = i <= best_i
            go_left[idx] = gl
            n_left += gl

        # Child segments
        seg_start[left_id] = s
        seg_len[left_id] = n_left
        seg_start[right_id] = s + n_left
        seg_len[right_id] = n - n_left

        for g in range(D):
            # Stable partition using temporary buffer
            left_ptr = s
            right_ptr = s + n_left
            rank = 0
            for i in range(s, s + n):
                idx = sorted_idx[g, i]
                if go_left[idx]:
                    sorted_idx[g, left_ptr] = idx
                    ranks_buf[left_ptr] = rank
                    left_ptr += 1
                else:
                    idx_buf[right_ptr] = idx
                    ranks_buf[right_ptr] = rank
                    right_ptr += 1
                if not cant_split[g, i]:
                    rank += 1

            # Copy back from buffer
            for i in range(s + n_left, s + n):
                sorted_idx[g, i] = idx_buf[i]

            for i in range(s, s + n - 1):
                cant_split[g, i] = ranks_buf[i] == ranks_buf[i + 1]

        # --------------------------------------------------------
        # Push children to stack (DFS)
        # --------------------------------------------------------
        # Right then left => left processed first (if you want)
        stack_node[stack_size] = right_id
        stack_depth[stack_size] = depth + 1
        stack_size += 1

        stack_node[stack_size] = left_id
        stack_depth[stack_size] = depth + 1
        stack_size += 1

    # Done, but leaf nodes don't all have node_value yet (we set mean at leaf creation).
    # Actually we already set node_value for leaves.

    return node_feature, node_threshold, node_value, left_child, right_child, node_count


# ============================================================
# Numba core: prediction
# ============================================================


@njit
def _predict_numba(
    X, node_feature, node_threshold, node_value, left_child, right_child, node_count
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
            thr = node_threshold[node]
            if X[i, f] <= thr:
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
        self.max_depth = max_depth
        self._fitted = False

    def preprocess_Xy(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        # Global sorted indices per feature
        # sorted_idx[f] is sorted order of samples by X[:, f]
        X = np.asarray(X, dtype=np.float32, order="F").T  # shape (D, N)
        sorted_idx = np.empty_like(X, dtype=np.int32, order="C")
        cant_split = np.empty_like(sorted_idx, dtype=bool)

        for f, feature_values in enumerate(X):
            sorted_idx[f, :] = np.argsort(feature_values)
            v_sorted = feature_values[sorted_idx[f, :]]
            cant_split[f, -1] = True
            cant_split[f, :-1] = np.diff(v_sorted) == 0

        return X, y, sorted_idx, cant_split

    def fit(self, X, y, sorted_idx=None, cant_split=None):
        if sorted_idx is None:
            X, y, sorted_idx, cant_split = self.preprocess_Xy(X, y)

        if self.max_depth is None:
            max_depth_int = -1
        else:
            max_depth_int = int(self.max_depth)

        (
            node_feature,
            node_threshold,
            node_value,
            left_child,
            right_child,
            node_count,
        ) = _build_tree_numba(X, y, sorted_idx, cant_split, max_depth_int)

        # Truncate to actual node_count to keep things clean
        self.node_feature = node_feature[:node_count]
        self.node_threshold = node_threshold[:node_count]
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
            self.node_value,
            self.left_child,
            self.right_child,
            self.n_nodes_,
        )
