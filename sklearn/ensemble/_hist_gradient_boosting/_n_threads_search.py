# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Prototype: empirical search for the fastest OpenMP thread count.

Some workloads (e.g. small/medium datasets, machines with noisy-neighbour or
oversubscription issues) can end up *slower* when using many OpenMP threads
than when using few, or even a single thread. This module implements a
small experiment to avoid such pathological cases: instead of blindly using
the default ``n_threads`` for every boosting iteration, we empirically probe
a handful of candidate thread counts early during ``fit`` and keep whichever
was fastest for the remaining iterations.

This is deliberately *not* wired into the OpenMP runtime (no global
``omp_set_num_threads`` call, no change to any ``prange`` signature):
``n_threads`` is simply the same plain Python int that already flows down to
the ``num_threads=n_threads`` argument of the ``prange`` calls used while
building trees (see ``histogram.pyx``, ``splitting.pyx``, ``grower.py``);
here we just vary its value between boosting iterations.
"""

from dataclasses import dataclass, field


@dataclass
class _NThreadsTrial:
    n_threads: int
    times: list = field(default_factory=list)

    @property
    def time(self):
        # Wall-clock iteration timings are noisy, and external hiccups (GC,
        # OS scheduling, page faults...) can only ever inflate a
        # measurement, never make it artificially faster. The minimum
        # observed time is therefore the best available estimate of the
        # "true" cost for this thread count.
        return min(self.times) if self.times else float("inf")


class NThreadsSearch:
    """Greedy empirical search for the fastest OpenMP thread count.

    Starting from `max_n_threads`, candidates are probed in decreasing
    order (`max_n_threads`, `// 2`, `// 4`, ..., down to `1`), each timed
    over `n_iter_per_candidate` boosting iterations. As soon as a candidate
    is more than `tol` slower than the best candidate found so far, the
    search stops and `best_n_threads` is used for all the remaining
    iterations.

    The first boosting iteration run at *any* candidate (including the very
    first one) is treated as an unmeasured "priming" iteration and excluded
    from the comparisons (see the `record` method): switching thread count
    triggers a transient slowdown lasting a few iterations (most likely a
    CPU frequency-governor ramp-up, since it persists regardless of the
    OpenMP runtime's thread wait/spin policy), and skipping just the first
    post-switch iteration measurably reduces -- without fully eliminating --
    the odds of the search mistaking a slow-to-ramp-up candidate for a
    genuinely slow one.

    Parameters
    ----------
    max_n_threads : int
        Thread count to start the search from (typically
        `_openmp_effective_n_threads()`). If <= 1 there is nothing to
        search for and the search is a no-op.
    n_iter_per_candidate : int, default=2
        Number of (post-priming) boosting iterations used to time each
        candidate.
    tol : float, default=0.08
        Relative tolerance used to decide that a candidate is "clearly"
        slower than the current best, so as to not react to noise in the
        (few) iteration timings that are available.

    Attributes
    ----------
    history : list of dict
        One entry per completed iteration (in call order to `record`),
        with keys "iteration", "n_threads", "time" and "warmup" (`True` for
        the unmeasured priming iteration of each candidate). Useful to
        inspect which thread counts were tried and how long they took.
    best_n_threads : int
        Best thread count found so far (`max_n_threads` before anything
        has been measured).
    best_time : float
        Timing (in seconds) associated with `best_n_threads`.
    done : bool
        Whether the search has concluded. Once `True`, `current_n_threads`
        stays equal to `best_n_threads` for the rest of the fit.
    """

    def __init__(self, max_n_threads, n_iter_per_candidate=2, tol=0.08):
        self.max_n_threads = max_n_threads
        self.n_iter_per_candidate = n_iter_per_candidate
        self.tol = tol

        self.candidates = self._make_candidates(max_n_threads)
        self.best_n_threads = max_n_threads
        self.best_time = float("inf")
        # Nothing to search for with 0 or 1 thread.
        self.done = max_n_threads <= 1

        self.history = []
        self._n_iterations = 0
        self._candidate_idx = 0
        self._current_trial = None if self.done else _NThreadsTrial(self.candidates[0])
        # Whether the current candidate's unmeasured priming iteration has
        # already run.
        self._primed = False

    @staticmethod
    def _make_candidates(max_n_threads):
        """Build the [max_n_threads, .../2, .../4, ..., 1] candidate list."""
        candidates = []
        n = max_n_threads
        while n > 1:
            candidates.append(n)
            n //= 2
        candidates.append(1)
        return candidates

    @property
    def current_n_threads(self):
        """Thread count that should be used for the next boosting iteration."""
        if self.done:
            return self.best_n_threads
        return self._current_trial.n_threads

    def record(self, elapsed):
        """Record the wall-clock time of the iteration that just ran.

        Must be called exactly once per boosting iteration, right after
        timing it, with the thread count that was used being whatever
        `current_n_threads` returned before running that iteration.
        """
        n_threads = self.current_n_threads
        if self.done:
            self.history.append(
                {
                    "iteration": self._n_iterations,
                    "n_threads": n_threads,
                    "time": elapsed,
                    "warmup": False,
                }
            )
            self._n_iterations += 1
            return

        is_warmup = not self._primed
        self.history.append(
            {
                "iteration": self._n_iterations,
                "n_threads": n_threads,
                "time": elapsed,
                "warmup": is_warmup,
            }
        )
        self._n_iterations += 1

        if is_warmup:
            # First iteration at this candidate: let the new thread count
            # settle before using it in comparisons.
            self._primed = True
            return

        trial = self._current_trial
        trial.times.append(elapsed)
        if len(trial.times) < self.n_iter_per_candidate:
            return  # Keep measuring the current candidate.

        candidate_time = trial.time
        is_clearly_worse = candidate_time > self.best_time * (1 + self.tol)
        if candidate_time < self.best_time:
            self.best_time = candidate_time
            self.best_n_threads = trial.n_threads

        self._candidate_idx += 1
        if is_clearly_worse or self._candidate_idx >= len(self.candidates):
            # Either downsizing just got clearly worse, or we ran out of
            # candidates (reached 1 thread): stop probing and settle on the
            # best thread count found so far.
            self.done = True
        else:
            self._current_trial = _NThreadsTrial(self.candidates[self._candidate_idx])
            self._primed = False
