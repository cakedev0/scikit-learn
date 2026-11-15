# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""A team of threads that stays in one OpenMP parallel region.

EXPERIMENT. Growing a tree opens ~6 short parallel regions per split, with
Python code in between. The cost of entering and leaving a region depends a
lot on how the OpenMP runtime makes its idle threads wait (wake-up of
sleeping threads, barrier at the end of each region).

`run_in_team(func, n_threads)` opens a single region. Its main thread calls
`func()`, with the GIL. The other threads loop on the tasks that the main
thread posts with `team_run`: they claim chunks of items with an atomic
counter, and the main thread waits until all the items are done, not until
all the threads are back. Idle threads spin shortly, then yield their CPU in
a loop.

Only GCC/Clang atomics and POSIX `sched_yield` here: not portable as is.
"""

from cython.parallel cimport parallel
from cpython.ref cimport PyObject

import os


cdef extern from *:
    """
    #include <stdint.h>
    #include <sched.h>
    #include <pthread.h>
    #ifdef _OPENMP
    #include <omp.h>
    static int hgb_team_thread_num(void) { return omp_get_thread_num(); }
    static int hgb_team_num_threads(void) { return omp_get_num_threads(); }
    #else
    static int hgb_team_thread_num(void) { return 0; }
    static int hgb_team_num_threads(void) { return 1; }
    #endif

    #if defined(__x86_64__) || defined(__i386__)
    #define hgb_team_relax() __builtin_ia32_pause()
    #elif defined(__aarch64__)
    #define hgb_team_relax() __asm__ __volatile__("yield")
    #else
    #define hgb_team_relax() ((void) 0)
    #endif

    typedef void (*hgb_team_fn)(void *, Py_ssize_t, Py_ssize_t);

    /* `word` = (generation << 32) | next item to claim.
       The generation is odd while the main thread writes the fields of a new
       task (like a seqlock) and even once the task is posted. A claim is a
       compare-and-swap on `word`, which fails if the generation has changed:
       a late thread can never claim an item of another task, nor run a task
       whose fields it read while they were being written. */
    typedef struct {
        _Alignas(64) uint64_t word;
        _Alignas(64) int64_t n_done;
        _Alignas(64) hgb_team_fn fn;
        void *ctx;
        int64_t n_items;
        int64_t chunk_size;
        int stop;
        int in_use;
        int active;
        int n_spins;
        int is_static;
        int n_threads;
        uint64_t start_gen;
        pthread_t owner;
    } hgb_team_t;

    static hgb_team_t hgb_team = {0};

    /* With the static schedule, each thread has its own range of items, as
       with OpenMP's schedule(static), so that a thread works on the same
       features from one task to the next. `word` is as in hgb_team_t, for
       the range of one thread. A thread that is done with its range takes
       items of the other ranges. */
    #define HGB_TEAM_MAX_THREADS 1024
    typedef struct { _Alignas(64) uint64_t word; } hgb_slot_t;
    static hgb_slot_t hgb_slots[HGB_TEAM_MAX_THREADS];

    #define HGB_LOAD(p) __atomic_load_n((p), __ATOMIC_SEQ_CST)
    #define HGB_STORE(p, v) __atomic_store_n((p), (v), __ATOMIC_SEQ_CST)

    static inline void hgb_team_pause(int *n_waits) {
        if (*n_waits < hgb_team.n_spins) {
            (*n_waits)++;
            hgb_team_relax();
        } else {
            sched_yield();
        }
    }

    /* Claim and run chunks of the range of thread `t` until none is left. */
    static void hgb_team_work_range(uint64_t gen, int t, hgb_team_fn fn,
                                    void *ctx, int64_t n_items,
                                    int64_t chunk_size, int n_threads) {
        int64_t end = (t + 1) * n_items / n_threads;
        uint64_t word = HGB_LOAD(&hgb_slots[t].word);
        for (;;) {
            int64_t start = (int64_t) (word & 0xffffffffu);
            if ((word >> 32) != gen || start >= end) return;
            if (!__atomic_compare_exchange_n(
                    &hgb_slots[t].word, &word, word + (uint64_t) chunk_size, 0,
                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
                continue;
            int64_t stop = start + chunk_size;
            if (stop > end) stop = end;
            fn(ctx, (Py_ssize_t) start, (Py_ssize_t) stop);
            __atomic_fetch_add(&hgb_team.n_done, stop - start, __ATOMIC_SEQ_CST);
            word = HGB_LOAD(&hgb_slots[t].word);
        }
    }

    /* Claim and run chunks of the task of generation `gen` until none is
       left. */
    static void hgb_team_work(uint64_t gen, int thread_num) {
        /* Atomic loads, as the stores in hgb_team_run: with plain accesses,
           the stores of a new task could be seen before the store of its odd
           generation (e.g. on ARM), and the check below would let through
           fields of the next task. */
        hgb_team_fn fn = HGB_LOAD(&hgb_team.fn);
        void *ctx = HGB_LOAD(&hgb_team.ctx);
        int64_t n_items = HGB_LOAD(&hgb_team.n_items);
        int64_t chunk_size = HGB_LOAD(&hgb_team.chunk_size);
        int is_static = HGB_LOAD(&hgb_team.is_static);
        int n_threads = HGB_LOAD(&hgb_team.n_threads);
        uint64_t word = HGB_LOAD(&hgb_team.word);
        /* The fields above are those of `gen` only if no other task was
           being posted while reading them. */
        if ((word >> 32) != gen) return;
        if (is_static) {
            for (int i = 0; i < n_threads; i++)
                hgb_team_work_range(gen, (thread_num + i) % n_threads, fn, ctx,
                                    n_items, chunk_size, n_threads);
            return;
        }
        for (;;) {
            int64_t start = (int64_t) (word & 0xffffffffu);
            if ((word >> 32) != gen || start >= n_items) return;
            if (!__atomic_compare_exchange_n(
                    &hgb_team.word, &word, word + (uint64_t) chunk_size, 0,
                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
                continue;
            int64_t stop = start + chunk_size;
            if (stop > n_items) stop = n_items;
            fn(ctx, (Py_ssize_t) start, (Py_ssize_t) stop);
            __atomic_fetch_add(&hgb_team.n_done, stop - start, __ATOMIC_SEQ_CST);
            word = HGB_LOAD(&hgb_team.word);
        }
    }

    static void hgb_team_worker(void) {
        int thread_num = hgb_team_thread_num();
        /* The task posted before the team started belongs to a previous
           team: its arguments are gone, it must not be looked at. */
        uint64_t seen = HGB_LOAD(&hgb_team.start_gen);
        int n_waits = 0;
        while (!HGB_LOAD(&hgb_team.stop)) {
            uint64_t gen = HGB_LOAD(&hgb_team.word) >> 32;
            if (gen != seen && !(gen & 1)) {
                seen = gen;
                hgb_team_work(gen, thread_num);
                n_waits = 0;
            } else {
                hgb_team_pause(&n_waits);
            }
        }
    }

    static void hgb_team_run(hgb_team_fn fn, void *ctx, Py_ssize_t n_items,
                             Py_ssize_t chunk_size) {
        if (chunk_size < 1) chunk_size = 1;
        if (n_items <= chunk_size) {
            /* A single chunk: not worth telling the others. */
            if (n_items > 0) fn(ctx, 0, n_items);
            return;
        }
        uint64_t gen = HGB_LOAD(&hgb_team.word) >> 32;
        HGB_STORE(&hgb_team.word, (gen + 1) << 32);
        HGB_STORE(&hgb_team.fn, fn);
        HGB_STORE(&hgb_team.ctx, ctx);
        HGB_STORE(&hgb_team.n_items, (int64_t) n_items);
        HGB_STORE(&hgb_team.chunk_size, (int64_t) chunk_size);
        HGB_STORE(&hgb_team.n_done, 0);
        if (hgb_team.is_static)
            for (int t = 0; t < hgb_team.n_threads; t++)
                HGB_STORE(&hgb_slots[t].word,
                          ((gen + 2) << 32)
                          | (uint64_t) (t * (int64_t) n_items / hgb_team.n_threads));
        HGB_STORE(&hgb_team.word, (gen + 2) << 32);
        hgb_team_work(gen + 2, 0);
        int n_waits = 0;
        while (HGB_LOAD(&hgb_team.n_done) < n_items) hgb_team_pause(&n_waits);
    }

    static int hgb_team_is_active(void) {
        return hgb_team.active && pthread_equal(hgb_team.owner, pthread_self());
    }

    static int hgb_team_acquire(int n_spins, int is_static) {
        int expected = 0;
        if (!__atomic_compare_exchange_n(&hgb_team.in_use, &expected, 1, 0,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return 0;
        hgb_team.n_spins = n_spins;
        HGB_STORE(&hgb_team.is_static, is_static);
        hgb_team.owner = pthread_self();
        HGB_STORE(&hgb_team.start_gen, HGB_LOAD(&hgb_team.word) >> 32);
        HGB_STORE(&hgb_team.stop, 0);
        return 1;
    }

    static void hgb_team_set_active(int active) {
        HGB_STORE(&hgb_team.n_threads, hgb_team_num_threads());
        if (hgb_team.n_threads > HGB_TEAM_MAX_THREADS)
            HGB_STORE(&hgb_team.is_static, 0);
        hgb_team.active = active;
    }
    static void hgb_team_stop(void) { HGB_STORE(&hgb_team.stop, 1); }
    static void hgb_team_release(void) { HGB_STORE(&hgb_team.in_use, 0); }
    """
    int hgb_team_thread_num() noexcept nogil
    void hgb_team_worker() noexcept nogil
    void hgb_team_run(task_fn fn, void *ctx, intp_t n_items,
                      intp_t chunk_size) noexcept nogil
    bint hgb_team_is_active() noexcept nogil
    bint hgb_team_acquire(int n_spins, int is_static) noexcept nogil
    void hgb_team_set_active(int active) noexcept nogil
    void hgb_team_stop() noexcept nogil
    void hgb_team_release() noexcept nogil


cdef bint team_active() noexcept nogil:
    """Whether the calling thread is the main thread of a running team."""
    return hgb_team_is_active()


cdef void team_run(
    task_fn fn, void *ctx, intp_t n_items, intp_t chunk_size
) noexcept nogil:
    """Run `fn` on `n_items` items, by chunks, on the threads of the team.

    Only to be called when `team_active()`.
    """
    hgb_team_run(fn, ctx, n_items, chunk_size)


cdef void _call(PyObject *func, PyObject *out) noexcept nogil:
    with gil:
        try:
            (<list> out).append((True, (<object> func)()))
        except BaseException as exc:
            (<list> out).append((False, exc))


def run_in_team(func, int n_threads):
    """Return `func()`, called in a parallel region of `n_threads` threads.

    Inside `func`, the kernels that support it share their work with the
    other threads of the region with `team_run`. Without a free team (a team
    is already running in the process) or with one thread, `func` is just
    called.
    """
    cdef:
        list out = []
        int n_spins = int(os.environ.get("SKLEARN_HGB_TEAM_SPINS", "100"))
        bint is_static = os.environ.get("SKLEARN_HGB_TEAM_SCHED", "static") == "static"

    if n_threads <= 1 or not hgb_team_acquire(n_spins, is_static):
        return func()

    try:
        with nogil, parallel(num_threads=n_threads):
            if hgb_team_thread_num() == 0:
                hgb_team_set_active(1)
                _call(<PyObject*> func, <PyObject*> out)
                hgb_team_set_active(0)
                hgb_team_stop()
            else:
                hgb_team_worker()
    finally:
        hgb_team_release()

    ok, value = out[0]
    if not ok:
        raise value
    return value


def team_is_active():
    return bool(hgb_team_is_active())
