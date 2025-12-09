"""Simple test for BinSortDecisionTree (simplified: MSE only, no missing values, no weights)"""

import numpy as np

from sklearn.datasets import make_regression
from sklearn.presortedtree.binsortdt import BinSortDecisionTree

# Test regression
print("Testing regression...")
X, y = make_regression(n_samples=100, n_features=5, random_state=42)
tree = BinSortDecisionTree(max_depth=5, n_bins=32)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Nodes: {tree.n_nodes_}")
print(f"  MSE: {np.mean((y - preds) ** 2):.4f}")

# Test with different depth
print("\nTesting with max_depth=3...")
X, y = make_regression(n_samples=200, n_features=10, random_state=42)
tree = BinSortDecisionTree(max_depth=3, n_bins=64)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Nodes: {tree.n_nodes_}")
print(f"  MSE: {np.mean((y - preds) ** 2):.4f}")

# Test with single feature
print("\nTesting with single feature...")
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel()
tree = BinSortDecisionTree(max_depth=4, n_bins=16)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Nodes: {tree.n_nodes_}")
print(f"  MSE: {np.mean((y - preds) ** 2):.4f}")

# Test overfitting check (no depth limit)
print("\nTesting overfitting (deep tree)...")
X, y = make_regression(n_samples=50, n_features=3, random_state=42, noise=0.1)
tree = BinSortDecisionTree(max_depth=None, n_bins=32)
tree.fit(X, y)
preds = tree.predict(X)
print(f"  Nodes: {tree.n_nodes_}")
print(f"  MSE: {np.mean((y - preds) ** 2):.6f}")

print("\n✓ All tests completed!")
