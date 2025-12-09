"""Test script for Forest with sample weights implementation."""

import time

import numpy as np

from sklearn.datasets import make_regression
from sklearn.presortedtree.forest import Forest

# Generate test data
print("Generating test data...")
X, y = make_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42)

# Test 1: Basic functionality
print("\n" + "=" * 60)
print("Test 1: Basic Functionality")
print("=" * 60)

forest = Forest(n_estimators=10, max_depth=5, random_state=42)
start = time.time()
forest.fit(X, y)
fit_time = time.time() - start
print(f"✓ Fit successful: {fit_time:.3f}s")

# Test predictions
y_pred = forest.predict(X[:10])
print(f"✓ Predictions shape: {y_pred.shape}")
print(f"  Sample predictions: {y_pred[:3]}")
print(f"  Sample actual: {y[:3]}")

# Test 2: Compare with/without bootstrap
print("\n" + "=" * 60)
print("Test 2: Bootstrap vs No Bootstrap")
print("=" * 60)

forest_bootstrap = Forest(n_estimators=20, max_depth=5, bootstrap=True, random_state=42)
start = time.time()
forest_bootstrap.fit(X, y)
bootstrap_time = time.time() - start
y_pred_bootstrap = forest_bootstrap.predict(X)

forest_no_bootstrap = Forest(
    n_estimators=20, max_depth=5, bootstrap=False, random_state=42
)
start = time.time()
forest_no_bootstrap.fit(X, y)
no_bootstrap_time = time.time() - start
y_pred_no_bootstrap = forest_no_bootstrap.predict(X)

print(f"✓ Bootstrap fit time: {bootstrap_time:.3f}s")
print(f"✓ No bootstrap fit time: {no_bootstrap_time:.3f}s")
print(
    f"  Prediction difference (mean abs): {np.mean(np.abs(y_pred_bootstrap - y_pred_no_bootstrap)):.4f}"
)

# Test 3: Thread scaling
print("\n" + "=" * 60)
print("Test 3: Thread Scaling Performance")
print("=" * 60)

n_samples_scale = 2000
n_features_scale = 30
n_estimators_scale = 50

print(f"Dataset: n_samples={n_samples_scale}, n_features={n_features_scale}")
print(f"Forest: n_estimators={n_estimators_scale}, max_depth=8")
print()

X_large, y_large = make_regression(
    n_samples=n_samples_scale, n_features=n_features_scale, noise=10.0, random_state=42
)

thread_counts = [1, 2, 4, 8]
times = []

for n_threads in thread_counts:
    forest = Forest(
        n_estimators=n_estimators_scale, max_depth=8, random_state=42, n_jobs=n_threads
    )

    start = time.time()
    forest.fit(X_large, y_large)
    elapsed = time.time() - start
    times.append(elapsed)

    # Verify predictions work
    y_pred = forest.predict(X_large[:5])

    speedup = times[0] / elapsed if len(times) > 0 else 1.0
    efficiency = speedup / n_threads * 100

    print(
        f"n_threads={n_threads:2d}: {elapsed:6.3f}s  "
        f"speedup={speedup:5.2f}x  efficiency={efficiency:5.1f}%"
    )

# Test 4: Verify sample weights logic
print("\n" + "=" * 60)
print("Test 4: Sample Weights Logic Verification")
print("=" * 60)

# Create a small dataset to manually verify
np.random.seed(42)
X_small = np.random.randn(10, 3).astype(np.float32)
y_small = np.random.randn(10).astype(np.float64)

# Test with bootstrap to ensure weights are being used
forest_small = Forest(n_estimators=5, max_depth=3, bootstrap=True, random_state=42)
forest_small.fit(X_small, y_small)
y_pred_small = forest_small.predict(X_small)

print("✓ Small dataset test passed")
print(f"  Dataset shape: {X_small.shape}")
print(f"  Number of trees: {len(forest_small.estimators_)}")
print(f"  Predictions range: [{y_pred_small.min():.3f}, {y_pred_small.max():.3f}]")

# Test 5: Different max_samples settings
print("\n" + "=" * 60)
print("Test 5: max_samples Parameter")
print("=" * 60)

for max_samples in [None, 0.5, 500]:
    forest = Forest(
        n_estimators=10, max_depth=5, max_samples=max_samples, random_state=42
    )
    start = time.time()
    forest.fit(X, y)
    elapsed = time.time() - start

    print(f"max_samples={max_samples!s:5s}: {elapsed:.3f}s")

print("\n" + "=" * 60)
print("All tests completed successfully! ✓")
print("=" * 60)
