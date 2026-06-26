from sklearn.utils._openmp_helpers import _openmp_calibrated_n_threads_for_testing


def test_openmp_calibrated_n_threads_uses_sequential_for_small_work():
    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=100.0,
            max_threads=8,
            seconds_per_work_unit=1.0,
            startup_seconds_per_thread=100.0,
        )
        == 1
    )


def test_openmp_calibrated_n_threads_clamps_to_max_threads():
    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=10_000.0,
            max_threads=8,
            seconds_per_work_unit=1.0,
            startup_seconds_per_thread=1.0,
        )
        == 8
    )


def test_openmp_calibrated_n_threads_falls_back_for_invalid_inputs():
    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=10_000.0,
            max_threads=8,
            seconds_per_work_unit=-1.0,
            startup_seconds_per_thread=1.0,
        )
        == 8
    )
    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=10_000.0,
            max_threads=8,
            seconds_per_work_unit=1.0,
            startup_seconds_per_thread=-1.0,
        )
        == 8
    )
    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=10_000.0,
            max_threads=1,
            seconds_per_work_unit=1.0,
            startup_seconds_per_thread=1.0,
        )
        == 1
    )


def test_openmp_calibrated_n_threads_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SKLEARN_HGB_OPENMP_ADAPTIVE_THREADS", "0")

    assert (
        _openmp_calibrated_n_threads_for_testing(
            work_units=100.0,
            max_threads=8,
            seconds_per_work_unit=1.0,
            startup_seconds_per_thread=100.0,
        )
        == 8
    )
