"""PHASE TIMERS (benchmarking only, not for merging).

Sum the per-tree phase times recorded by `_fit_validated` over a fitted tree,
forest or gradient boosting estimator.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

PHASES = ("build", "sort", "search", "final", "node_reset", "n_sorts")


def phase_times(estimator):
    """Sum of the phase times over all the trees of `estimator`.

    Times are wall times in seconds, summed over trees: for a forest fitted in
    parallel, divide by the number of threads to compare with the fit time.
    "other" is the part of the build time outside the timed phases (tree
    builder, node values, ...).
    """
    if hasattr(estimator, "_phase_times"):
        trees = [estimator]
    else:
        trees = list(np.ravel(estimator.estimators_))
    totals = {phase: sum(tree._phase_times[phase] for tree in trees) for phase in PHASES}
    totals["other"] = totals["build"] - sum(
        totals[phase] for phase in ("sort", "search", "final", "node_reset")
    )
    totals["n_trees"] = len(trees)
    return totals
