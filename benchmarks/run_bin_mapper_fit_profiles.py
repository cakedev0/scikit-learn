"""Run _BinMapper.fit timings and py-spy profiles for one commit."""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("uniform", "long-tail", "binary")
DATASETS = ("uniform",)
NAN_CASES = (False, True)
SAMPLE_WEIGHT_CASES = (False, True)
# SAMPLE_WEIGHT_CASES = (True,)

N_SAMPLES = 100_000
N_FEATURES = 32
N_BINS = 256
N_THREADS = 1
REPEAT = 7
WARMUP = 1
RANDOM_SEED = 0
MISSING_FRACTION = 0.1


def git_output(args, *, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def checkout_commit(commit, *, cwd):
    subprocess.run(["git", "checkout", "--detach", commit], cwd=cwd, check=True)
    return git_output(["rev-parse", "HEAD"], cwd=cwd)


def resolve_current_commit(*, cwd):
    return git_output(["rev-parse", "HEAD"], cwd=cwd)


def make_case_name(commit_short, dataset, has_nan, has_sample_weight):
    dataset_label = dataset.replace("-", "_")
    nan_label = "nan" if has_nan else "no_nan"
    weight_label = "weighted" if has_sample_weight else "unweighted"
    return f"bin_mapper_fit_{dataset_label}_{nan_label}_{weight_label}_{commit_short}"


def make_workload_command(
    workload_script, dataset, has_nan, has_sample_weight, n_threads
):
    missing_fraction = MISSING_FRACTION if has_nan else 0
    command = [
        sys.executable,
        str(workload_script),
        "--dataset",
        dataset,
        "--n-samples",
        str(N_SAMPLES),
        "--n-features",
        str(N_FEATURES),
        "--n-bins",
        str(N_BINS),
        "--n-threads",
        str(n_threads),
        "--repeat",
        str(REPEAT),
        "--warmup",
        str(WARMUP),
        "--random-seed",
        str(RANDOM_SEED),
        "--missing-fraction",
        str(missing_fraction),
    ]
    if has_sample_weight:
        command.append("--random-sample-weights")
    return command


def run_case(
    *,
    repo_root,
    workload_script,
    commit_short,
    dataset,
    has_nan,
    has_sample_weight,
    timing_only,
    n_threads,
):
    case_name = make_case_name(commit_short, dataset, has_nan, has_sample_weight)
    workload_command = make_workload_command(
        workload_script,
        dataset,
        has_nan,
        has_sample_weight,
        n_threads,
    )

    print(f"[timing] {case_name}", flush=True)
    tic = time.perf_counter()
    subprocess.run(workload_command, cwd=repo_root, check=True)
    elapsed = time.perf_counter() - tic

    if not timing_only:
        svg_path = repo_root / f"{case_name}.svg"
        print(f"[profile] {case_name}", flush=True)
        completed = subprocess.run(
            ["py-spy", "record", "-o", str(svg_path), "--", *workload_command],
            cwd=repo_root,
        )
        if completed.returncode != 0 and not svg_path.exists():
            completed.check_returncode()

    return elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run _BinMapper.fit timings and py-spy profiles for one commit."
    )
    parser.add_argument(
        "commit",
        nargs="?",
        default=None,
        help="Commit digest or ref to check out. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--timing-only",
        "--no-profile",
        action="store_true",
        help="Run timings only and skip py-spy SVG profiles.",
    )
    parser.add_argument("--n-threads", type=int, default=N_THREADS)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(git_output(["rev-parse", "--show-toplevel"], cwd=Path.cwd()))
    if args.commit is None:
        commit_hash = resolve_current_commit(cwd=repo_root)
    else:
        commit_hash = checkout_commit(args.commit, cwd=repo_root)
    commit_short = commit_hash[:4]
    print(f"commit: {commit_hash}", flush=True)
    workload_script = repo_root / "benchmarks" / "profile_bin_mapper_fit.py"
    if not workload_script.exists():
        raise RuntimeError(f"Missing workload script: {workload_script}")

    timings = []
    for dataset, has_nan, has_sample_weight in itertools.product(
        DATASETS,
        NAN_CASES,
        SAMPLE_WEIGHT_CASES,
    ):
        elapsed = run_case(
            repo_root=repo_root,
            workload_script=workload_script,
            commit_short=commit_short,
            dataset=dataset,
            has_nan=has_nan,
            has_sample_weight=has_sample_weight,
            timing_only=args.timing_only,
            n_threads=args.n_threads,
        )
        case_name = make_case_name(commit_short, dataset, has_nan, has_sample_weight)
        timings.append((case_name, elapsed))


if __name__ == "__main__":
    main()
