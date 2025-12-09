"""Debug test for BinSortDecisionTree"""

import numpy as np

from sklearn.presortedtree.binsortdt import BinSortDecisionTree

# Simple test with missing values
print("Testing with missing values...")
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [np.nan, 8.0]], dtype=np.float32)
y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Missing values: {np.isnan(X).sum()}")

tree = BinSortDecisionTree(max_depth=2, criterion="squared_error", n_bins=32)
try:
    tree.fit(X, y)
    print("✓ Fit succeeded")
    preds = tree.predict(X)
    print(f"  Predictions: {preds}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
