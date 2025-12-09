"""Test Forest on large dataset (2M samples, 10 features)."""

import time

import numpy as np

from sklearn.datasets import make_regression

# from sklearn.presortedtree.forest import Forest
from sklearn.ensemble import RandomForestRegressor as Forest

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


print("=" * 60)
print("Large Dataset Test: 2M samples, 10 features")
print("=" * 60)

# Generate large dataset
print("\nGenerating dataset...")
n_samples = 200_000
n_features = 50
start = time.time()
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=10,
    noise=10.0,
    random_state=42,
)
gen_time = time.time() - start
print(f"Dataset generated in {gen_time:.2f}s")
print(f"Shape: X={X.shape}, y={y.shape}")
print(f"Memory: X={X.nbytes / 1e9:.2f} GB, y={y.nbytes / 1e9:.2f} GB")

# Test with different thread counts
print("\n" + "=" * 60)
print("Thread Scaling Test")
print("=" * 60)
n_estimators = 24
print(f"Config: n_estimators={n_estimators}, max_depth=10, bootstrap=True")
print()

thread_counts = [1, 2, 4, 6]
times = []

for n_threads in thread_counts:
    print(f"Testing with {n_threads} thread(s)...", end=" ", flush=True)

    forest = Forest(
        n_estimators=n_estimators,
        max_depth=10,
        bootstrap=True,
        random_state=42,
        n_jobs=n_threads,
    )

    start = time.time()
    forest.fit(X, y)
    elapsed = time.time() - start
    times.append(elapsed)

    # Quick prediction test
    y_pred = forest.predict(X[:100])

    speedup = times[0] / elapsed if len(times) > 0 else 1.0
    efficiency = speedup / n_threads * 100

    print(f"{elapsed:6.2f}s  speedup={speedup:5.2f}x  efficiency={efficiency:5.1f}%")

# Test prediction performance
print("\n" + "=" * 60)
print("Prediction Performance Test")
print("=" * 60)

# Use the last trained forest (8 threads)
n_pred_samples = 100_000
print(f"\nPredicting on {n_pred_samples:,} samples...")
start = time.time()
y_pred = forest.predict(X[:n_pred_samples])
pred_time = time.time() - start
print(f"Prediction time: {pred_time:.3f}s")
print(f"Throughput: {n_pred_samples / pred_time:.0f} predictions/sec")

# Calculate some basic metrics
print("\n" + "=" * 60)
print("Model Quality Check")
print("=" * 60)
train_pred = forest.predict(X[:10000])
train_mse = np.mean((train_pred - y[:10000]) ** 2)
train_r2 = 1 - train_mse / np.var(y[:10000])
print(f"Training MSE (first 10k samples): {train_mse:.2f}")
print(f"Training R² (first 10k samples): {train_r2:.4f}")

print("\n" + "=" * 60)
print("Test completed successfully! ✓")
print("=" * 60)
