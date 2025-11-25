# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import Protocol, Union

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

# ============================================================================
# Base Criterion Protocol (for typing only)
# ============================================================================


class Criterion(Protocol):
    """Protocol defining the interface for split criteria.

    This is used for type hints only and is not actually inherited by
    the jitclass implementations (Numba doesn't support inheritance).

    All criterion classes should implement these methods to be compatible
    with tree building algorithms.
    """

    @property
    def sum_w(self) -> float: ...

    def init_node(
        self, y: np.ndarray, sample_weights: np.ndarray | None, indices: np.ndarray
    ) -> tuple[float, float | np.ndarray]:
        """Initialize criterion for a node and compute node impurity.

        Parameters
        ----------
        y : ndarray
            Target values (full array)
        sample_weights : ndarray or None
            Sample weights (full array), or None for uniform weights
        indices : ndarray
            Indices of samples in this node

        Returns
        -------
        float
            Node impurity (e.g., Gini impurity or SSE)
        """
        ...

    def reset(self) -> None:
        """Reset left child statistics to empty (all samples in right child)."""
        ...

    def reverse_reset(self) -> None:
        """Set left child to complement of current left child."""
        ...

    def update(self, value: Union[int, float], weight: float) -> None:
        """Add one sample to the left child.

        Parameters
        ----------
        value : int or float
            For classification: class index (int)
            For regression: target value (float)
        weight : float
            Sample weight
        """
        ...

    def impurity_improvement(self) -> float:
        """Compute proxy for impurity improvement of current split.

        Returns a value where higher indicates a better split.
        This is typically a proxy that avoids computing actual impurities
        for efficiency.

        Returns
        -------
        float
            Improvement proxy (higher = better split)
        """
        ...


# ============================================================================
# Gini Criterion (Classification)
# ============================================================================

gini_spec = [
    ("n_classes", int64),
    ("sum_total", float64[:]),
    ("sum_left", float64[:]),
    ("sum_w", float64),
]


@jitclass(gini_spec)
class Gini:
    """Gini criterion for incremental split evaluation.

    Designed to be used in a scan loop where we incrementally add samples
    to the left child and compute the Gini impurity improvement.
    """

    def __init__(self, n_classes):
        """Initialize Gini criterion.

        Parameters
        ----------
        n_classes : int
            Number of classes in the classification problem
        """
        self.n_classes = n_classes
        self.sum_total = np.zeros(n_classes, dtype=np.float64)
        self.sum_left = np.zeros(n_classes, dtype=np.float64)
        self.sum_w = 0.0

    def init_node(self, y, sample_weights, indices):
        """Initialize for a node by computing statistics from data.

        Parameters
        ----------
        y : array
            Target values (full array)
        sample_weights : array or None
            Sample weights (full array), or None for uniform weights
        indices : array
            Indices of samples in this node
        """
        self.sum_w = 0.0
        for c in range(self.n_classes):
            self.sum_total[c] = 0.0

        for idx in indices:
            w = 1.0
            if sample_weights is not None:
                w = sample_weights[idx]

            class_idx = int(y[idx])
            self.sum_total[class_idx] += w

        sq_sum = 0
        for c in range(self.n_classes):
            w_c = self.sum_total[c]
            self.sum_w += w_c
            sq_sum += w_c * w_c

        # Node impurity: Gini = 1 - sum(p_k^2) where p_k = w_k/sum_w
        # Returns: sum_w * Gini = sum_w - sum(w_k^2)/sum_w
        return (self.sum_w - sq_sum / self.sum_w, self.sum_total / self.sum_w)

    def reset(self):
        self.sum_left[:] = 0.0

    def reverse_reset(self):
        for c in range(self.n_classes):
            self.sum_left[c] = self.sum_total[c] - self.sum_left[c]

    def update(self, class_idx, weight):
        """Add one sample to the left child.

        Parameters
        ----------
        class_idx : int
            Class of the sample
        weight : float
            Weight of the sample
        """
        self.sum_left[class_idx] += weight

    def impurity_improvement(self):
        """Compute proxy impurity improvement for current split.

        Returns sum_left[k]^2/n_left + sum_right[k]^2/n_right, which is
        proportional to the reduction in Gini impurity. Higher values
        indicate better splits (purer children).

        Returns
        -------
        float
            Improvement proxy (higher = better split)
        """
        # Compute left impurity: 1 - sum(p_k^2)
        # But we only need the proxy: -n_left * impurity_left - n_right * impurity_right
        # Which simplifies to: sum_left[k]^2 / n_left + sum_right[k]^2 / n_right

        left_w = 0.0
        sq_left = 0.0
        sq_right = 0.0
        for c in range(self.n_classes):
            left_c = self.sum_left[c]
            left_w += left_c
            sq_left += left_c * left_c
            right_c = self.sum_total[c] - left_c
            sq_right += right_c * right_c

        right_w = self.sum_w - left_w

        # Proxy: higher is better
        return sq_left / left_w + sq_right / right_w


# ============================================================================
# MSE Criterion (Regression)
# ============================================================================

mse_spec = [
    ("sum_y", float64),
    ("sum_y2", float64),
    ("sum_w", float64),
    ("sum_left", float64),
    ("left_w", float64),
]


@jitclass(mse_spec)
class MSE:
    """MSE criterion for incremental split evaluation.

    Designed to be used in a scan loop where we incrementally add samples
    to the left child and compute the SSE (sum of squared errors).
    """

    def __init__(self):
        """Initialize MSE criterion."""
        self.sum_y = 0.0
        self.sum_left = 0.0
        self.sum_y2 = 0.0
        self.sum_w = 0.0
        self.left_w = 0.0

    def init_node(self, y, sample_weights, indices):
        """Initialize for a node by computing statistics from data.

        Parameters
        ----------
        y : array
            Target values (full array)
        sample_weights : array or None
            Sample weights (full array), or None for uniform weights
        indices : array
            Indices of samples in this node
        """
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_w = 0.0

        w = 1.0
        for idx in indices:
            w_val = val = y[idx]
            if sample_weights is not None:
                w = sample_weights[idx]
                w_val *= w

            self.sum_y += w_val
            self.sum_y2 += w_val * val
            self.sum_w += w

        # SSE = sum(y^2) - sum(y)^2 / n
        return (
            self.sum_y2 - self.sum_y * self.sum_y / self.sum_w,
            self.sum_y / self.sum_w,
        )

    def reset(self):
        self.sum_left = 0.0
        self.left_w = 0.0

    def reverse_reset(self):
        self.sum_left = self.sum_y - self.sum_left
        self.left_w = self.sum_w - self.left_w

    def update(self, y_value, weight):
        """Add one sample to the left child.

        Parameters
        ----------
        y_value : float
            Target value of the sample (unweighted)
        weight : float
            Weight of the sample (method will multiply y_value * weight internally)
        """
        self.sum_left += y_value * weight
        self.left_w += weight

    def impurity_improvement(self):
        """Compute proxy impurity improvement for current split.

        Returns sum_left^2/n_left + sum_right^2/n_right, which is proportional
        to the reduction in SSE. Higher values indicate better splits.

        Returns
        -------
        float
            Improvement proxy (higher = better split)
        """
        right_w = self.sum_w - self.left_w
        sum_right = self.sum_y - self.sum_left

        # SSE = sum(y^2) - sum_left^2/n_left - sum_right^2/n_right
        # We return: sum_left^2/n_left + sum_right^2/n_right
        return (self.sum_left * self.sum_left) / self.left_w + (
            sum_right * sum_right
        ) / right_w
