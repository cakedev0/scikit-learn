# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._typedefs cimport intp_t

# A task processes the items `start <= i < stop`.
ctypedef void (*task_fn)(void *ctx, intp_t start, intp_t stop) noexcept nogil

cdef bint team_active() noexcept nogil

cdef void team_run(
    task_fn fn, void *ctx, intp_t n_items, intp_t chunk_size
) noexcept nogil
