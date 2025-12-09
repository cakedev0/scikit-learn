"""Verify BinSortDecisionTree produces correct splits"""

import numpy as np

from sklearn.presortedtree.binsortdt import BinSortDecisionTree

# Test 1: Simple 1D data where we know the optimal split
print("Test 1: Known optimal split")
X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]], dtype=np.float32)
y = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=np.float64)
tree = BinSortDecisionTree(max_depth=1, n_bins=16)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Predictions: {preds}")
print("  Expected: [1, 1, 1, 10, 10, 10] (approximately)")
print(f"  MSE: {np.mean((y - preds) ** 2):.6f}")
print(f"  Split threshold: {tree.node_threshold[0]:.2f} (should be between 3 and 10)")

# Test 2: Perfect separation
print("\nTest 2: Perfect separation")
X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
tree = BinSortDecisionTree(max_depth=10, n_bins=16)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Predictions: {preds}")
print(f"  Target:      {y}")
print(f"  MSE: {np.mean((y - preds) ** 2):.6f} (should be ~0)")

# Test 3: Multi-dimensional
print("\nTest 3: Multi-dimensional")
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)  # XOR-like pattern
tree = BinSortDecisionTree(max_depth=3, n_bins=16)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Predictions: {preds}")
print(f"  Target:      {y}")
print(f"  MSE: {np.mean((y - preds) ** 2):.6f}")

# Test 4: Verify partitioning maintains data integrity
print("\nTest 4: Data integrity after partitioning")
np.random.seed(42)
X = np.random.randn(20, 3).astype(np.float32)
y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(20) * 0.1
tree = BinSortDecisionTree(max_depth=4, n_bins=32)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Training MSE: {np.mean((y - preds) ** 2):.6f}")
print(f"  Nodes: {tree.n_nodes_}")

print("\n✓ All verification tests completed!")
