"""Profile _BinMapper.fit on a few synthetic data distributions.

Examples
--------
Run timings directly:

    python benchmarks/profile_bin_mapper_fit.py

Record a Python-only py-spy flamegraph:

    py-spy record -o bin_mapper_fit.svg -- python benchmarks/profile_bin_mapper_fit.py

Record each distribution separately:

    py-spy record -o bin_mapper_fit_uniform.svg -- \
        python benchmarks/profile_bin_mapper_fit.py --dataset uniform
    py-spy record -o bin_mapper_fit_long_tail.svg -- \
        python benchmarks/profile_bin_mapper_fit.py --dataset long-tail
    py-spy record -o bin_mapper_fit_binary.svg -- \
        python benchmarks/profile_bin_mapper_fit.py --dataset binary
"""

import argparse
import gc
import time

import numpy as np

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

DATASETS = ("uniform", "long-tail", "binary")


def make_data(kind, *, n_samples, n_features, rng, order):
    shape = (n_samples, n_features)

    if kind == "uniform":
        X = rng.random(shape, dtype=np.float64)
    elif kind == "long-tail":
        X = rng.lognormal(mean=0.0, sigma=3.0, size=shape)
    elif kind == "binary":
        X = rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.float64)
    else:
        raise ValueError(f"Unknown dataset kind: {kind!r}")

    return np.asarray(X, dtype=np.float64, order=order)


def make_sample_weight(n_samples, rng):
    return rng.random(n_samples, dtype=np.float64) * 10


def inject_missing_values(X, *, missing_fraction, rng):
    if missing_fraction == 0:
        return

    mask = rng.random(X.shape) < missing_fraction
    X[mask] = np.nan


def fit_bin_mapper(X, *, sample_weight, n_bins, n_threads, random_state):
    bin_mapper = _BinMapper(
        n_bins=n_bins,
        random_state=random_state,
        n_threads=n_threads,
    )
    return bin_mapper.fit(X, sample_weight=sample_weight)


def run_one_dataset(kind, args):
    rng = np.random.default_rng(args.random_seed)
    X = make_data(
        kind,
        n_samples=args.n_samples,
        n_features=args.n_features,
        rng=rng,
        order=args.order,
    )
    inject_missing_values(
        X,
        missing_fraction=args.missing_fraction,
        rng=rng,
    )

    if args.random_sample_weights:
        sample_weight = make_sample_weight(args.n_samples, rng)
    else:
        sample_weight = None

    print(
        f"{kind}: X.shape={X.shape}, dtype={X.dtype}, "
        f"order={'F' if X.flags.f_contiguous else 'C'}, "
        f"missing_fraction={args.missing_fraction}, "
        f"sample_weight={sample_weight is not None}",
        flush=True,
    )

    for repeat_idx in range(args.warmup):
        fit_bin_mapper(
            X,
            sample_weight=sample_weight,
            n_bins=args.n_bins,
            n_threads=args.n_threads,
            random_state=repeat_idx,
        )

    durations = []
    n_thresholds = None
    for repeat_idx in range(args.repeat):
        tic = time.perf_counter()
        bin_mapper = fit_bin_mapper(
            X,
            sample_weight=sample_weight,
            n_bins=args.n_bins,
            n_threads=args.n_threads,
            random_state=repeat_idx,
        )
        duration = time.perf_counter() - tic
        durations.append(duration)
        n_thresholds = sum(
            thresholds.shape[0] for thresholds in bin_mapper.bin_thresholds_
        )
        print(f"  repeat {repeat_idx + 1:02d}: {duration:.3f}s", flush=True)

    print(
        f"  mean={np.mean(durations):.3f}s, min={np.min(durations):.3f}s, "
        f"max={np.max(durations):.3f}s, thresholds={n_thresholds}",
        flush=True,
    )

    del X
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run _BinMapper.fit on uniform, long-tail, and binary data."
    )
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--missing-fraction", type=float, default=0)
    parser.add_argument(
        "--random-sample-weights",
        action="store_true",
        help="Generate random sample weights and pass them to _BinMapper.fit.",
    )
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--dataset", choices=DATASETS, default="uniform")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        "_BinMapper.fit profile workload: "
        f"n_samples={args.n_samples}, n_features={args.n_features}, "
        f"n_bins={args.n_bins}, "
        f"n_threads={args.n_threads}, repeat={args.repeat}, "
        f"missing_fraction={args.missing_fraction}, "
        f"sample_weight={args.random_sample_weights}",
        flush=True,
    )

    run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()
