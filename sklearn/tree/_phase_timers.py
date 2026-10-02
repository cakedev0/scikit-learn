"""PHASE TIMERS (benchmarking only, not for merging).

Sum the per-tree phase times recorded by `_fit_validated` over a fitted tree,
forest or gradient boosting estimator.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time
from functools import cache

import numpy as np

from sklearn.tree._splitter import _py_timer_ticks

PHASES = ("build", "sort", "search", "final", "node_reset", "n_sorts")
TIMED_PHASES = ("build", "sort", "search", "final", "node_reset")


@cache
def ticks_per_second():
    """Calibrate the timer ticks (CPU timestamp counter) against perf_counter."""
    tic, tic_ticks = time.perf_counter(), _py_timer_ticks()
    time.sleep(0.2)
    return (_py_timer_ticks() - tic_ticks) / (time.perf_counter() - tic)


def phase_times(estimator):
    """Sum of the phase times over all the trees of `estimator`.

    Times are in seconds (timer ticks converted with ticks_per_second), summed
    over trees: for a forest fitted in
    parallel, divide by the number of threads to compare with the fit time.
    "other" is the part of the build time outside the timed phases (tree
    builder, node values, ...).
    """
    if hasattr(estimator, "_phase_times"):
        trees = [estimator]
    else:
        trees = list(np.ravel(estimator.estimators_))
    totals = {phase: sum(tree._phase_times[phase] for tree in trees) for phase in PHASES}
    for phase in TIMED_PHASES:
        totals[phase] /= ticks_per_second()
    totals["other"] = totals["build"] - sum(
        totals[phase] for phase in ("sort", "search", "final", "node_reset")
    )
    totals["n_trees"] = len(trees)
    return totals


BUCKET_PHASES = ("sort", "search", "final", "node_reset", "node_total")


def phase_times_by_size(estimator):
    """Phase times by node size, summed over all the trees of `estimator`.

    Nodes are bucketed by size: bucket b holds the nodes with
    2**b <= n_node_samples < 2**(b + 1). Returns a list of dicts, one per
    non-empty bucket, with the size range, the number of nodes and of split
    nodes, the average depth of the nodes, and the wall time (in seconds,
    summed over trees) of each phase. "node_total" is the whole time spent on
    the node by the tree builder, and "other" is node_total minus the timed
    phases. Only best splits time the sort, search and final phases.
    """
    if hasattr(estimator, "_phase_times"):
        trees = [estimator]
    else:
        trees = list(np.ravel(estimator.estimators_))
    times = sum(tree._phase_times["bucket_times"] for tree in trees) / ticks_per_second()
    counts = sum(tree._phase_times["bucket_counts"] for tree in trees)
    rows = []
    for b in np.flatnonzero(counts[:, 0]):
        row = {
            "size_min": 2**b,
            "size_max": 2 ** (b + 1) - 1,
            "n_nodes": int(counts[b, 0]),
            "n_split": int(counts[b, 1]),
            "avg_depth": counts[b, 2] / counts[b, 0],
        }
        row.update(zip(BUCKET_PHASES, times[b]))
        row["other"] = row["node_total"] - sum(
            row[phase] for phase in ("sort", "search", "final", "node_reset")
        )
        rows.append(row)
    return rows


def format_phase_times_by_size(estimator):
    """Table of phase_times_by_size, in ms per tree."""
    n_trees = 1 if hasattr(estimator, "_phase_times") else np.size(estimator.estimators_)
    columns = ("sort", "search", "final", "node_reset", "other", "node_total")
    lines = [
        f"{'node size':>19} {'n_nodes':>10} {'n_split':>10} {'depth':>6} | ms per tree: "
        + " ".join(f"{c:>10}" for c in columns)
    ]
    for row in phase_times_by_size(estimator):
        size = f"[{row['size_min']}, {row['size_max']}]"
        lines.append(
            f"{size:>19} {row['n_nodes'] / n_trees:10.0f} {row['n_split'] / n_trees:10.0f} "
            f"{row['avg_depth']:6.1f} |              "
            + " ".join(f"{1e3 * row[c] / n_trees:10.1f}" for c in columns)
        )
    return "\n".join(lines)
