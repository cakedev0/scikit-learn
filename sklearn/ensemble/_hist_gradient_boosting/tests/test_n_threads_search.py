import pytest

from sklearn.ensemble._hist_gradient_boosting._n_threads_search import NThreadsSearch


@pytest.mark.parametrize(
    "max_n_threads, expected_candidates",
    [
        (1, [1]),
        (2, [2, 1]),
        (3, [3, 1]),
        (5, [5, 2, 1]),
        (8, [8, 4, 2, 1]),
    ],
)
def test_make_candidates(max_n_threads, expected_candidates):
    search = NThreadsSearch(max_n_threads=max_n_threads)
    assert search.candidates == expected_candidates


@pytest.mark.parametrize("max_n_threads", [0, 1])
def test_no_search_below_two_threads(max_n_threads):
    # Nothing to probe with 0 or 1 thread: the search is a no-op from the
    # start and always recommends max_n_threads.
    search = NThreadsSearch(max_n_threads=max_n_threads)
    assert search.done
    assert search.current_n_threads == max_n_threads

    search.record(1234.0)
    assert search.current_n_threads == max_n_threads
    assert search.done


def test_first_iteration_excluded_from_measurements():
    # The very first (priming) iteration must never affect the comparisons,
    # however slow it is.
    search = NThreadsSearch(max_n_threads=4, n_iter_per_candidate=2)
    assert search.current_n_threads == 4

    search.record(1_000.0)  # priming, ignored
    assert search.best_time == float("inf")
    assert not search.done
    assert search.current_n_threads == 4
    assert search.history[0]["warmup"] is True


def test_priming_iteration_excluded_on_every_candidate_switch():
    # Not just the very first iteration: switching to *any* new candidate
    # triggers one unmeasured priming iteration before it starts counting.
    search = NThreadsSearch(max_n_threads=4, n_iter_per_candidate=2, tol=0.08)

    search.record(1.0)  # priming for candidate 4
    search.record(2.0)
    search.record(2.0)  # candidate 4: best_time = 2.0

    assert search.current_n_threads == 2
    search.record(1_000.0)  # priming for candidate 2: wildly slow, but ignored
    assert not search.done
    assert search.best_time == pytest.approx(2.0)  # unaffected by the spike

    search.record(1.0)
    search.record(1.0)  # candidate 2: genuinely faster, becomes the new best

    assert search.best_n_threads == 2
    assert search.best_time == pytest.approx(1.0)

    warmup_flags = [entry["warmup"] for entry in search.history]
    assert warmup_flags == [True, False, False, True, False, False]


def test_search_settles_on_fastest_thread_count():
    search = NThreadsSearch(max_n_threads=8, n_iter_per_candidate=2, tol=0.08)

    # candidate 8 threads: priming, then measured.
    assert search.current_n_threads == 8
    search.record(10.0)  # priming, ignored
    search.record(5.0)
    search.record(5.2)

    # candidate 4 threads: priming, then measured -- faster, new best.
    assert search.current_n_threads == 4
    search.record(3.6)  # priming, ignored
    search.record(3.0)
    search.record(3.1)

    # candidate 2 threads: priming, then measured -- faster still.
    assert search.current_n_threads == 2
    search.record(2.8)  # priming, ignored
    search.record(2.5)
    search.record(2.6)

    # candidate 1 thread: priming, then measured -- much slower
    # (contention-free work needs threads here), clearly worse than the
    # current best -> search stops there.
    assert search.current_n_threads == 1
    search.record(7.0)  # priming, ignored
    search.record(6.0)
    search.record(6.5)

    assert search.done
    assert search.best_n_threads == 2
    assert search.best_time == pytest.approx(2.5)
    assert search.current_n_threads == 2

    # 4 candidates * (1 priming + 2 measured) iterations each = 12.
    assert len(search.history) == 12
    assert [entry["n_threads"] for entry in search.history] == (
        [8, 8, 8] + [4, 4, 4] + [2, 2, 2] + [1, 1, 1]
    )
    assert [entry["warmup"] for entry in search.history] == ([True, False, False] * 4)


def test_one_thread_can_be_selected_as_fastest():
    # The primary motivation of the search: pathological over-threading
    # slow-downs must be escapable, all the way down to a single thread.
    search = NThreadsSearch(max_n_threads=4, n_iter_per_candidate=2, tol=0.08)

    search.record(1.0)  # priming with 4 threads

    assert search.current_n_threads == 4
    search.record(9.0)
    search.record(9.5)  # 4 threads is pathologically slow

    assert search.current_n_threads == 2
    search.record(8.5)  # priming, ignored
    search.record(8.0)
    search.record(8.2)  # still bad

    assert search.current_n_threads == 1
    search.record(0.6)  # priming, ignored
    search.record(0.5)
    search.record(0.5)  # single-threaded is much faster

    assert search.done
    assert search.best_n_threads == 1
    assert search.current_n_threads == 1


def test_tolerance_avoids_reacting_to_noise():
    # A candidate that is only marginally worse (within `tol`) than the
    # current best must not stop the search: it could just be measurement
    # noise, and a much better candidate may still be found further down.
    search = NThreadsSearch(max_n_threads=8, n_iter_per_candidate=2, tol=0.1)

    search.record(10.0)  # priming

    assert search.current_n_threads == 8
    search.record(5.0)
    search.record(5.0)  # candidate 8: best_time = 5.0

    assert search.current_n_threads == 4
    search.record(5.3)  # priming, ignored
    search.record(5.2)
    search.record(5.2)  # candidate 4: 4% worse than best, within 10% tol

    # Search must *not* have stopped: it keeps exploring smaller counts.
    assert not search.done
    assert search.best_n_threads == 8  # unchanged, candidate 4 wasn't better

    assert search.current_n_threads == 2
    search.record(3.2)  # priming, ignored
    search.record(3.0)
    search.record(3.0)  # candidate 2: much better, becomes the new best

    assert search.best_n_threads == 2
    assert search.best_time == pytest.approx(3.0)

    assert search.current_n_threads == 1
    search.record(3.6)  # priming, ignored
    search.record(3.5)
    search.record(3.5)  # clearly worse than 3.0 * 1.1 -> stop

    assert search.done
    assert search.best_n_threads == 2


def test_search_stops_when_out_of_candidates():
    # If every candidate down to 1 thread is within tolerance of the best,
    # the search must still terminate once 1 thread has been tried.
    search = NThreadsSearch(max_n_threads=2, n_iter_per_candidate=1, tol=0.5)

    search.record(1.0)  # priming

    assert search.current_n_threads == 2
    search.record(1.0)  # candidate 2

    assert search.current_n_threads == 1
    search.record(0.9)  # priming, ignored
    search.record(1.01)  # candidate 1, within tolerance

    assert search.done
    assert search.best_n_threads == 2
