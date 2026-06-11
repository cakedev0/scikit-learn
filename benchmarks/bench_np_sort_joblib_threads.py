"""Benchmark np.sort inside joblib threading workers."""

import argparse
import time

import numpy as np
from joblib import Parallel, delayed


def sort_one(seed, n_samples):
    rng = np.random.default_rng(seed)
    x = rng.random(n_samples)
    return np.sort(x)


def approx_quantile(arr):
    n = arr.size
    return np.sort(arr)[:: n // 100]


def time_sort(n_jobs, *, n_tasks, n_samples, repeat):
    durations = []
    for repeat_idx in range(repeat):
        rng = np.random.default_rng(repeat_idx)
        X = rng.random((n_samples, n_tasks))
        tic = time.perf_counter()
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(approx_quantile)(x) for x in X.T
        )
        durations.append(time.perf_counter() - tic)
    return durations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark np.sort under joblib's threading backend."
    )
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-tasks", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"np.sort joblib threading benchmark: n_tasks={args.n_tasks}, "
        f"n_samples={args.n_samples}, repeat={args.repeat}",
        flush=True,
    )
    for n_jobs in args.n_jobs:
        durations = time_sort(
            n_jobs,
            n_tasks=args.n_tasks,
            n_samples=args.n_samples,
            repeat=args.repeat,
        )
        print(
            f"n_jobs={n_jobs:>2}: "
            f"mean={np.mean(durations):.3f}s "
            f"min={np.min(durations):.3f}s "
            f"max={np.max(durations):.3f}s "
            f"runs={[round(duration, 3) for duration in durations]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
