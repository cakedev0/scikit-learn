"""Benchmark RandomForest fit time against scikit-learn-intelex.

The suite is intentionally compact: every retained case should run in less than
10 seconds per estimator/backend timing, and the full suite should run in less
than 3 minutes on the development machine used for this investigation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

N_ESTIMATORS = 20
DEFAULT_N_JOBS = 1
MAX_CASE_SECONDS = 10.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    n_samples: int
    n_features: int
    dtype: str
    signal: str
    n_classes: int = 2


@dataclass(frozen=True)
class CaseSpec:
    name: str
    dataset: DatasetSpec
    params: dict


DATASETS = {
    "reg_1f_deep_f32": DatasetSpec(
        "reg_1f_deep_f32", "regression", 60_000, 1, "float32", "deep"
    ),
    "reg_12f_signal_f32": DatasetSpec(
        "reg_12f_signal_f32", "regression", 24_000, 12, "float32", "signal"
    ),
    "reg_80f_wide_f32": DatasetSpec(
        "reg_80f_wide_f32", "regression", 9_000, 80, "float32", "wide"
    ),
    "reg_12f_signal_f64": DatasetSpec(
        "reg_12f_signal_f64", "regression", 16_000, 12, "float64", "signal"
    ),
    "clf_12f_signal_f32": DatasetSpec(
        "clf_12f_signal_f32", "classification", 28_000, 12, "float32", "signal", 3
    ),
    "clf_96f_sqrt_f32": DatasetSpec(
        "clf_96f_sqrt_f32", "classification", 10_000, 96, "float32", "wide", 4
    ),
    "reg_24f_low_card_f32": DatasetSpec(
        "reg_24f_low_card_f32", "regression", 30_000, 24, "float32", "low_card"
    ),
    "clf_24f_low_card_f32": DatasetSpec(
        "clf_24f_low_card_f32", "classification", 30_000, 24, "float32", "low_card", 4
    ),
}


RETAINED_CASES = [
    CaseSpec(
        "reg_1f_deep_full",
        DATASETS["reg_1f_deep_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
    CaseSpec(
        "reg_12f_full_deep",
        DATASETS["reg_12f_signal_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
    CaseSpec(
        "reg_12f_shallow_bootstrap",
        DATASETS["reg_12f_signal_f32"],
        {"bootstrap": True, "max_features": 1.0, "max_depth": 10},
    ),
    CaseSpec(
        "reg_80f_sqrt_leaf8",
        DATASETS["reg_80f_wide_f32"],
        {
            "bootstrap": False,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": 8,
        },
    ),
    CaseSpec(
        "reg_12f_full_f64",
        DATASETS["reg_12f_signal_f64"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
    CaseSpec(
        "clf_12f_full_deep",
        DATASETS["clf_12f_signal_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
    CaseSpec(
        "clf_12f_shallow_bootstrap",
        DATASETS["clf_12f_signal_f32"],
        {"bootstrap": True, "max_features": 1.0, "max_depth": 10},
    ),
    CaseSpec(
        "clf_96f_sqrt_leaf8",
        DATASETS["clf_96f_sqrt_f32"],
        {
            "bootstrap": False,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": 8,
        },
    ),
    CaseSpec(
        "reg_24f_low_card",
        DATASETS["reg_24f_low_card_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
    CaseSpec(
        "clf_24f_low_card",
        DATASETS["clf_24f_low_card_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": None},
    ),
]


PILOT_CASES = [
    *RETAINED_CASES,
    CaseSpec(
        "reg_80f_full_depth12",
        DATASETS["reg_80f_wide_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": 12},
    ),
    CaseSpec(
        "clf_96f_full_depth12",
        DATASETS["clf_96f_sqrt_f32"],
        {"bootstrap": False, "max_features": 1.0, "max_depth": 12},
    ),
]


def make_data(spec: DatasetSpec):
    rng = np.random.default_rng(stable_seed(spec.name))
    X = rng.standard_normal((spec.n_samples, spec.n_features), dtype=np.float64)

    if spec.signal == "deep":
        x0 = X[:, 0]
        base = np.sin(9 * x0) + 0.5 * np.sign(x0) * np.sqrt(np.abs(x0) + 0.01)
        base += 0.15 * rng.standard_normal(spec.n_samples)
    elif spec.signal == "wide":
        informative = min(8, spec.n_features)
        coefs = np.linspace(1.5, -0.6, informative)
        base = X[:, :informative] @ coefs
        base += 0.35 * (X[:, 0] > 0) * X[:, 1]
        base += 0.15 * rng.standard_normal(spec.n_samples)
    elif spec.signal == "low_card":
        bins = np.array([3, 4, 5, 7, 9, 11], dtype=np.float64)
        for feature_idx in range(spec.n_features):
            n_bins = bins[feature_idx % bins.size]
            X[:, feature_idx] = np.floor((X[:, feature_idx] + 3.0) * n_bins / 6.0)
            X[:, feature_idx] = np.clip(X[:, feature_idx], 0, n_bins - 1)
        informative = min(8, spec.n_features)
        coefs = np.linspace(1.0, -0.7, informative)
        base = X[:, :informative] @ coefs
        base += 0.4 * (X[:, 0] == X[:, 1])
        base += 0.05 * rng.standard_normal(spec.n_samples)
    else:
        informative = min(6, spec.n_features)
        coefs = np.linspace(2.0, -1.0, informative)
        base = X[:, :informative] @ coefs
        base += 0.5 * np.sin(X[:, 0] * X[:, 1])
        base += 0.15 * rng.standard_normal(spec.n_samples)

    X = X.astype(spec.dtype, copy=False)
    if spec.task == "regression":
        return X, base.astype(np.float64, copy=False)

    quantiles = np.linspace(0, 1, spec.n_classes + 1)[1:-1]
    thresholds = np.quantile(base, quantiles)
    y = np.searchsorted(thresholds, base).astype(np.int64, copy=False)
    return X, y


def stable_seed(text: str) -> int:
    seed = 0x5EED
    for char in text:
        seed = ((seed * 131) + ord(char)) % (2**32)
    return seed


def get_case(case_name: str, suite: str = "retained") -> CaseSpec:
    cases = PILOT_CASES if suite == "pilot" else RETAINED_CASES
    for case in cases:
        if case.name == case_name:
            return case
    valid = ", ".join(case.name for case in cases)
    raise SystemExit(f"Unknown case {case_name!r}. Valid cases: {valid}")


def make_estimator(case: CaseSpec, backend: str, n_jobs: int):
    params = {
        "n_estimators": N_ESTIMATORS,
        "n_jobs": n_jobs,
        "random_state": stable_seed(case.name),
        **case.params,
    }
    if backend == "sklearn":
        if case.dataset.task == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**params)
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)

    params["max_bins"] = case.dataset.n_samples
    if case.dataset.task == "regression":
        from sklearnex.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    from sklearnex.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params)


def sklearnex_support(estimator, X, y):
    status = estimator._onedal_cpu_supported("fit", X, y, None)
    return status.get_status(), list(getattr(status, "messages", ()))


def summarize_forest(estimator):
    summary = {}
    try:
        trees = estimator.estimators_
    except Exception as exc:
        # sklearnex lazy conversion can fail on unsupported cases.
        summary["tree_summary_error"] = repr(exc)
        return summary

    depths = np.array([tree.tree_.max_depth for tree in trees], dtype=np.int64)
    leaves = np.array([tree.tree_.n_leaves for tree in trees], dtype=np.int64)
    summary.update(
        {
            "depth_median": float(np.median(depths)),
            "depth_max": int(depths.max()),
            "leaves_median": float(np.median(leaves)),
            "leaves_max": int(leaves.max()),
        }
    )
    return summary


def time_fit(case: CaseSpec, backend: str, n_jobs: int, repeat: int):
    X, y = make_data(case.dataset)
    estimator = make_estimator(case, backend, n_jobs)

    support_ok = True
    support_messages = []
    if backend == "sklearnex":
        support_ok, support_messages = sklearnex_support(estimator, X, y)
        if not support_ok:
            return {
                "case": case.name,
                "backend": backend,
                "repeat": repeat,
                "included": False,
                "reason": "sklearnex fallback preflight",
                "support_messages": support_messages,
                **case_metadata(case, n_jobs),
            }

    with threadpool_limits(limits=n_jobs):
        start = time.perf_counter()
        estimator.fit(X, y)
        fit_time = time.perf_counter() - start

    included = True
    reason = ""
    if backend == "sklearnex" and not hasattr(estimator, "_onedal_estimator"):
        included = False
        reason = "sklearnex fitted without _onedal_estimator"

    result = {
        "case": case.name,
        "backend": backend,
        "repeat": repeat,
        "fit_time": fit_time,
        "included": included,
        "reason": reason,
        "support_messages": support_messages,
        **case_metadata(case, n_jobs),
    }
    result.update(summarize_forest(estimator))
    return result


def case_metadata(case: CaseSpec, n_jobs: int):
    return {
        "task": case.dataset.task,
        "dataset": case.dataset.name,
        "n_samples": case.dataset.n_samples,
        "n_features": case.dataset.n_features,
        "dtype": case.dataset.dtype,
        "signal": case.dataset.signal,
        "n_classes": case.dataset.n_classes,
        "n_estimators": N_ESTIMATORS,
        "n_jobs": n_jobs,
        "params": json.dumps(case.params, sort_keys=True),
        "max_bins_sklearnex": case.dataset.n_samples,
    }


def environment_metadata():
    import sklearnex

    import sklearn

    try:
        import onedal

        onedal_version = getattr(onedal, "__version__", None)
    except ImportError:
        onedal_version = None

    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "sklearn": sklearn.__version__,
        "sklearnex": getattr(sklearnex, "__version__", None),
        "onedal": onedal_version,
        "threadpool_info": threadpool_info(),
        "n_estimators": N_ESTIMATORS,
        "max_case_seconds": MAX_CASE_SECONDS,
    }


def run_suite(args):
    cases = PILOT_CASES if args.suite == "pilot" else RETAINED_CASES
    backends = ["sklearn", "sklearnex"]
    rows = []
    start = time.perf_counter()
    for case in cases:
        for backend in backends:
            for repeat in range(args.repeats):
                row = time_fit(case, backend, args.n_jobs, repeat)
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - start
    write_outputs(rows, args.output_dir, args.suite, elapsed)
    print(f"elapsed={elapsed:.3f}s", file=sys.stderr)
    return 0 if elapsed <= args.max_suite_seconds or args.suite == "pilot" else 2


def write_outputs(rows, output_dir: Path, suite: str, elapsed: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{suite}_raw.jsonl"
    csv_path = output_dir / f"{suite}_summary.csv"
    env_path = output_dir / "environment.json"

    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    env = environment_metadata()
    env["suite_elapsed"] = elapsed
    env_path.write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")

    summaries = summarize_rows(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case",
            "backend",
            "included",
            "n_samples",
            "n_features",
            "dtype",
            "task",
            "n_jobs",
            "median_fit_time",
            "min_fit_time",
            "max_fit_time",
            "repeats",
            "max_case_ok",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["backend"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (case, backend), items in grouped.items():
        included_items = [item for item in items if item.get("included")]
        times = [item["fit_time"] for item in included_items if "fit_time" in item]
        first = items[0]
        reason = "; ".join(
            sorted({item.get("reason", "") for item in items if item.get("reason")})
        )
        summaries.append(
            {
                "case": case,
                "backend": backend,
                "included": bool(times),
                "n_samples": first["n_samples"],
                "n_features": first["n_features"],
                "dtype": first["dtype"],
                "task": first["task"],
                "n_jobs": first["n_jobs"],
                "median_fit_time": median(times) if times else "",
                "min_fit_time": min(times) if times else "",
                "max_fit_time": max(times) if times else "",
                "repeats": len(times),
                "max_case_ok": max(times) <= MAX_CASE_SECONDS if times else False,
                "reason": reason,
            }
        )
    return summaries


def run_one(args):
    case = get_case(args.case, args.suite)
    for repeat in range(args.repeats):
        row = time_fit(case, args.backend, args.n_jobs, repeat)
        print(json.dumps(row, sort_keys=True), flush=True)


def preflight(args):
    failures = []
    for case in RETAINED_CASES:
        X, y = make_data(case.dataset)
        estimator = make_estimator(case, "sklearnex", args.n_jobs)
        support_ok, messages = sklearnex_support(estimator, X, y)
        row = {
            "case": case.name,
            "support_ok": support_ok,
            "messages": messages,
            **case_metadata(case, args.n_jobs),
        }
        print(json.dumps(row, sort_keys=True))
        if not support_ok:
            failures.append(row)
    return 1 if failures else 0


def run_unsupported_checks(args):
    from scipy import sparse

    checks = []
    case = RETAINED_CASES[0]
    X, y = make_data(case.dataset)

    unsupported = [
        ("sparse_input", sparse.csr_matrix(X), y, {}),
        ("warm_start", X, y, {"warm_start": True}),
        ("multi_output", X, np.column_stack([y, y]), {}),
    ]
    for name, Xi, yi, extra_params in unsupported:
        probe = CaseSpec(name, case.dataset, {**case.params, **extra_params})
        estimator = make_estimator(probe, "sklearnex", args.n_jobs)
        support_ok, messages = sklearnex_support(estimator, Xi, yi)
        row = {"check": name, "support_ok": support_ok, "messages": messages}
        checks.append(row)
        print(json.dumps(row, sort_keys=True))

    return 0 if all(not row["support_ok"] for row in checks) else 1


def profile_command(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        __file__,
        "one",
        "--case",
        args.case,
        "--backend",
        args.backend,
        "--suite",
        args.suite,
        "--n-jobs",
        str(args.n_jobs),
        "--repeats",
        "1",
    ]
    py_spy_cmd = [
        "py-spy",
        "record",
        "--rate",
        str(args.rate),
        "--format",
        args.format,
        "--output",
        str(output),
        "--",
        *cmd,
    ]
    if args.native:
        py_spy_cmd.insert(2, "--native")
    print(" ".join(py_spy_cmd))
    completed = subprocess.run(py_spy_cmd, check=False)
    return completed.returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--suite", choices=["retained", "pilot"], default="retained"
    )
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    run_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/rf_intelex/results")
    )
    run_parser.add_argument("--max-suite-seconds", type=float, default=180.0)
    run_parser.set_defaults(func=run_suite)

    one_parser = subparsers.add_parser("one")
    one_parser.add_argument("--case", required=True)
    one_parser.add_argument(
        "--backend", choices=["sklearn", "sklearnex"], required=True
    )
    one_parser.add_argument(
        "--suite", choices=["retained", "pilot"], default="retained"
    )
    one_parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    one_parser.add_argument("--repeats", type=int, default=1)
    one_parser.set_defaults(func=run_one)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    preflight_parser.set_defaults(func=preflight)

    unsupported_parser = subparsers.add_parser("unsupported-checks")
    unsupported_parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    unsupported_parser.set_defaults(func=run_unsupported_checks)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--case", required=True)
    profile_parser.add_argument(
        "--backend", choices=["sklearn", "sklearnex"], required=True
    )
    profile_parser.add_argument(
        "--suite", choices=["retained", "pilot"], default="retained"
    )
    profile_parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    profile_parser.add_argument("--output", required=True)
    profile_parser.add_argument(
        "--format", choices=["flamegraph", "speedscope"], default="speedscope"
    )
    profile_parser.add_argument("--rate", type=int, default=200)
    profile_parser.add_argument("--native", action="store_true")
    profile_parser.set_defaults(func=profile_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
