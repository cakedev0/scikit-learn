import os
from cython.parallel import prange
from joblib import cpu_count
from libc.math cimport round
from libc.math cimport sqrt
from libc.stdlib cimport free
from libc.stdlib cimport malloc


# Module level cache for cpu_count as we do not expect this to change during
# the lifecycle of a Python program. This dictionary is keyed by
# only_physical_cores.
_CPU_COUNTS = {}

cdef bint _HGB_THREAD_CALIBRATION_DONE = False
cdef int _HGB_THREAD_CALIBRATION_MAX_N_THREADS = 0
cdef double _HGB_THREAD_SECONDS_PER_WORK_UNIT = -1.0
cdef double _HGB_THREAD_STARTUP_SECONDS_PER_THREAD = -1.0
cdef bint _HGB_THREAD_CONFIG_READ = False
cdef bint _HGB_ADAPTIVE_THREADS_DISABLED = False


cdef bint _hgb_adaptive_threads_disabled():
    global _HGB_THREAD_CONFIG_READ
    global _HGB_ADAPTIVE_THREADS_DISABLED

    if not _HGB_THREAD_CONFIG_READ:
        _HGB_ADAPTIVE_THREADS_DISABLED = (
            os.getenv("SKLEARN_HGB_OPENMP_ADAPTIVE_THREADS", "1") == "0"
        )
        _HGB_THREAD_CONFIG_READ = True
    return _HGB_ADAPTIVE_THREADS_DISABLED


cdef double _measure_axpy_seconds_per_item() noexcept nogil:
    cdef:
        Py_ssize_t n_items = 32768
        Py_ssize_t n_repeats = 256
        Py_ssize_t i
        double* x = <double*> malloc(n_items * sizeof(double))
        double* y = <double*> malloc(n_items * sizeof(double))
        double start
        double elapsed
        double sink = 0.0

    if x == NULL or y == NULL:
        free(x)
        free(y)
        return -1.0

    for i in range(n_items):
        x[i] = 1.0 / (i + 1)
        y[i] = 0.0

    start = omp_get_wtime()
    for _ in range(n_repeats):
        for i in range(n_items):
            y[i] += 2.0 * x[i]
    elapsed = omp_get_wtime() - start

    for i in range(8):
        sink += y[i]

    free(x)
    free(y)
    if sink < 0.0:
        return -1.0
    return elapsed / (n_items * n_repeats)


cdef double _measure_prange_startup_seconds_per_thread(int n_threads) noexcept nogil:
    cdef:
        Py_ssize_t n_repeats = 256
        int thread_idx
        double start
        double elapsed
        double sink = 0.0
        double* scratch = <double*> malloc(n_threads * sizeof(double))

    if scratch == NULL:
        return -1.0

    for thread_idx in range(n_threads):
        scratch[thread_idx] = 0.0

    start = omp_get_wtime()
    for _ in range(n_repeats):
        for thread_idx in prange(n_threads, schedule="static", chunksize=1,
                                 num_threads=n_threads):
            scratch[thread_idx] += 1.0
    elapsed = omp_get_wtime() - start

    for thread_idx in range(n_threads):
        sink += scratch[thread_idx]

    free(scratch)
    if sink < 0.0:
        return -1.0
    return elapsed / (n_repeats * n_threads)


cdef void _openmp_ensure_hgb_thread_calibration(int max_n_threads):
    cdef:
        int candidate
        int n_valid_candidates = 0
        double seconds_per_thread
        double startup_sum = 0.0
        int candidates[3]

    global _HGB_THREAD_CALIBRATION_DONE
    global _HGB_THREAD_CALIBRATION_MAX_N_THREADS
    global _HGB_THREAD_SECONDS_PER_WORK_UNIT
    global _HGB_THREAD_STARTUP_SECONDS_PER_THREAD

    if max_n_threads < 2:
        return

    if _hgb_adaptive_threads_disabled():
        return

    if (
        _HGB_THREAD_CALIBRATION_DONE
        and max_n_threads <= _HGB_THREAD_CALIBRATION_MAX_N_THREADS
    ):
        return

    _HGB_THREAD_CALIBRATION_DONE = True
    _HGB_THREAD_CALIBRATION_MAX_N_THREADS = max_n_threads

    if not SKLEARN_OPENMP_PARALLELISM_ENABLED:
        _HGB_THREAD_SECONDS_PER_WORK_UNIT = -1.0
        _HGB_THREAD_STARTUP_SECONDS_PER_THREAD = -1.0
        return

    candidates[0] = 2
    candidates[1] = 4
    candidates[2] = 16

    with nogil:
        _HGB_THREAD_SECONDS_PER_WORK_UNIT = _measure_axpy_seconds_per_item()
        for candidate in candidates:
            if candidate > max_n_threads:
                continue
            seconds_per_thread = _measure_prange_startup_seconds_per_thread(candidate)
            if seconds_per_thread > 0.0:
                startup_sum += seconds_per_thread
                n_valid_candidates += 1

    if n_valid_candidates > 0:
        _HGB_THREAD_STARTUP_SECONDS_PER_THREAD = startup_sum / n_valid_candidates
    else:
        _HGB_THREAD_STARTUP_SECONDS_PER_THREAD = -1.0


cdef int _openmp_calibrated_n_threads_from_constants(
    double work_units,
    int max_threads,
    double seconds_per_work_unit,
    double startup_seconds_per_thread,
    bint adaptive_threads_disabled,
):
    cdef:
        double optimal_n_threads
        int n_threads

    if max_threads <= 1 or work_units <= 0.0:
        return 1

    if adaptive_threads_disabled:
        return max_threads

    if seconds_per_work_unit <= 0.0 or startup_seconds_per_thread <= 0.0:
        return max_threads

    optimal_n_threads = sqrt(work_units * seconds_per_work_unit
                             / startup_seconds_per_thread)
    if optimal_n_threads < 2.0:
        return 1

    n_threads = <int> round(optimal_n_threads)
    return min(n_threads, max_threads)


cdef int _openmp_calibrated_n_threads(double work_units, int max_threads):
    if max_threads <= 1 or work_units <= 0.0:
        return 1

    if _hgb_adaptive_threads_disabled():
        return max_threads

    _openmp_ensure_hgb_thread_calibration(max_threads)
    return _openmp_calibrated_n_threads_from_constants(
        work_units,
        max_threads,
        _HGB_THREAD_SECONDS_PER_WORK_UNIT,
        _HGB_THREAD_STARTUP_SECONDS_PER_THREAD,
        False,
    )


cpdef int _openmp_calibrated_n_threads_for_testing(
    double work_units,
    int max_threads,
    double seconds_per_work_unit,
    double startup_seconds_per_thread,
):
    return _openmp_calibrated_n_threads_from_constants(
        work_units,
        max_threads,
        seconds_per_work_unit,
        startup_seconds_per_thread,
        os.getenv("SKLEARN_HGB_OPENMP_ADAPTIVE_THREADS", "1") == "0",
    )


cpdef tuple _openmp_get_hgb_thread_calibration():
    _openmp_ensure_hgb_thread_calibration(omp_get_max_threads())
    return (
        _HGB_THREAD_SECONDS_PER_WORK_UNIT,
        _HGB_THREAD_STARTUP_SECONDS_PER_THREAD,
    )


def _openmp_parallelism_enabled():
    """Determines whether scikit-learn has been built with OpenMP

    It allows to retrieve at runtime the information gathered at compile time.
    """
    # SKLEARN_OPENMP_PARALLELISM_ENABLED is resolved at compile time and defined
    # in _openmp_helpers.pxd as a boolean. This function exposes it to Python.
    return SKLEARN_OPENMP_PARALLELISM_ENABLED


cpdef _openmp_effective_n_threads(n_threads=None, only_physical_cores=True):
    """Determine the effective number of threads to be used for OpenMP calls

    - For ``n_threads = None``,
      - if the ``OMP_NUM_THREADS`` environment variable is set, return
        ``openmp.omp_get_max_threads()``
      - otherwise, return the minimum between ``openmp.omp_get_max_threads()``
        and the number of cpus, taking cgroups quotas into account. Cgroups
        quotas can typically be set by tools such as Docker.
      The result of ``omp_get_max_threads`` can be influenced by environment
      variable ``OMP_NUM_THREADS`` or at runtime by ``omp_set_num_threads``.

    - For ``n_threads > 0``, return this as the maximal number of threads for
      parallel OpenMP calls.

    - For ``n_threads < 0``, return the maximal number of threads minus
      ``|n_threads + 1|``. In particular ``n_threads = -1`` will use as many
      threads as there are available cores on the machine.

    - Raise a ValueError for ``n_threads = 0``.

    Passing the `only_physical_cores=False` flag makes it possible to use extra
    threads for SMT/HyperThreading logical cores. It has been empirically
    observed that using as many threads as available SMT cores can slightly
    improve the performance in some cases, but can severely degrade
    performance other times. Therefore it is recommended to use
    `only_physical_cores=True` unless an empirical study has been conducted to
    assess the impact of SMT on a case-by-case basis (using various input data
    shapes, in particular small data shapes).

    If scikit-learn is built without OpenMP support, always return 1.
    """
    if n_threads == 0:
        raise ValueError("n_threads = 0 is invalid")

    if not SKLEARN_OPENMP_PARALLELISM_ENABLED:
        # OpenMP disabled at build-time => sequential mode
        return 1

    if os.getenv("OMP_NUM_THREADS"):
        # Fall back to user provided number of threads making it possible
        # to exceed the number of cpus.
        max_n_threads = omp_get_max_threads()
    else:
        try:
            n_cpus = _CPU_COUNTS[only_physical_cores]
        except KeyError:
            n_cpus = cpu_count(only_physical_cores=only_physical_cores)
            _CPU_COUNTS[only_physical_cores] = n_cpus
        max_n_threads = min(omp_get_max_threads(), n_cpus)

    if n_threads is None:
        return max_n_threads
    elif n_threads < 0:
        return max(1, max_n_threads + n_threads + 1)

    return n_threads
