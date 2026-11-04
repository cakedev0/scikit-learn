cimport cython

from sklearn.ensemble._hist_gradient_boosting.common cimport hist_struct
from sklearn.ensemble._hist_gradient_boosting.common cimport X_BINNED_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common cimport G_H_DTYPE_C
from sklearn.utils._typedefs cimport uint8_t


@cython.final
cdef class HistogramBuilder:
    cdef public:
        const X_BINNED_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        G_H_DTYPE_C [::1] gradients
        G_H_DTYPE_C [::1] hessians
        G_H_DTYPE_C [::1] ordered_gradients
        G_H_DTYPE_C [::1] ordered_hessians
        uint8_t hessians_are_constant
        int n_threads

    cdef void _compute_histogram_brute_single_feature(
        HistogramBuilder self,
        const int feature_idx,
        const unsigned int [::1] sample_indices,
        hist_struct [:, ::1] histograms) noexcept nogil


cpdef void _subtract_histograms(
    const int feature_idx,
    unsigned int n_bins,
    hist_struct [:, ::1] hist_a,
    hist_struct [:, ::1] hist_b,
) noexcept nogil
