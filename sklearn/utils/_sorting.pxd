from sklearn.utils._typedefs cimport intp_t, uint8_t, uint16_t, uint32_t

from cython cimport floating

cdef void simultaneous_sort(
    floating* values,
    intp_t* indices,
    intp_t n,
    bint use_three_way_partition=*,
) noexcept nogil

ctypedef fused radix_t:
    uint8_t
    uint16_t
    uint32_t

cdef enum:
    # Size of the `counts` buffer that `radix_sort` needs.
    RADIX_SORT_COUNTS_SIZE = (1 << 16) + 1

cdef void radix_sort(
    radix_t* values,
    intp_t* indices,
    intp_t n,
    radix_t max_value,
    radix_t* values_buffer,
    intp_t* indices_buffer,
    intp_t* counts,
) noexcept nogil
