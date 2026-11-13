from sklearn.utils._typedefs cimport intp_t

from cython cimport floating


cpdef enum SortPartitioning:
    # See simultaneous_sort for when to use each of them.
    TWO_WAY
    MIXED


cdef void simultaneous_sort(
    floating* values,
    intp_t* indices,
    intp_t n,
    SortPartitioning partitioning=*,
) noexcept nogil
