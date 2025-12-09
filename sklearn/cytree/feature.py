import numpy as np
from numba import njit


class SortFeature:
    def __init__(self, x: np.ndarray, x_buf: np.ndarray):
        self.x = x
        self.x_buf = x_buf

    def sort(
        self, y, sample_weight, start, end, y_out, sample_weight_out, split_indices_out
    ):
        sorter = np.argsort(self.x[start:end])
        sorter += start
        n = end - start
        self.x_buf[:n] = self.x[sorter]
        y_out[:n] = y[sorter]
        if sample_weight is not None:
            sample_weight_out[:n] = sample_weight[sorter]

        split_indices = np.flatnonzero(self.x_buf[: n - 1] < self.x_buf[1:n])
        # TODO: compute n_missings
        n_splits = split_indices.size
        split_indices_out[:n_splits] = split_indices
        self.split_indices = split_indices
        return n_splits

    def get_split_threshold(self, split_num):
        split_idx = self.split_indices[split_num]
        return (self.x_buf[split_idx] + self.x_buf[split_idx + 1]) / 2

    def goes_left(self, threshold, missing_go_left, idx):
        val = self.x[idx]
        if missing_go_left and np.isnan(val):
            return True
        return val <= threshold

    def swap(self, i, j):
        self.x[i], self.x[j] = self.x[j], self.x[i]


class BinnedFeature:
    def __init__(self, x_binned: np.ndarray, bin_thresholds: np.ndarray):
        # if there are no missing, I expect: bin_thresholds[-1] = np.inf
        # if there are missing, I expect: bin_thresholds[-2] = np.inf, bin_thresholds[-1] = np.nan
        self.x_binned = x_binned
        self.bin_thresholds = bin_thresholds
        self.n_bins = bin_thresholds.size
        self.has_missing = np.isnan(bin_thresholds[-1])

    def sort(
        self, y, sample_weight, start, end, y_out, sample_weight_out, split_indices_out
    ):
        bin_ptrs = bin_sort(
            self.x_binned,
            y,
            sample_weight,
            y_out,
            sample_weight_out,
            self.n_bins,
            start,
            end,
        )
        if self.has_missing:
            n_splits = self.n_bins - 2
            n_missing = bin_ptrs[-1] - bin_ptrs[-2]
            split_indices_out[:n_splits] = bin_ptrs[1:-2]
        else:
            n_splits = self.n_bins - 1
            split_indices_out[:n_splits] = bin_ptrs[1:-1]
            n_missing = 0

        return n_splits, n_missing

    def get_split_threshold(self, split_num):
        return self.bin_thresholds[split_num]

    def goes_left(self, idx, threshold, missing_go_left):
        bin = self.x_binned[idx]
        bin_right_val = self.bin_thresholds[bin]
        if missing_go_left and np.isnan(bin_right_val):
            return True
        return bin_right_val <= threshold

    def swap(self, i, j):
        self.x_binned[i], self.x_binned[j] = self.x_binned[j], self.x_binned[i]


@njit
def bin_sort(
    x_binned,
    y,
    y_buf,
    n_bins,
    s,
    e,
):
    # Count samples per bin
    bin_ptrs = np.zeros(n_bins + 1, dtype=np.int32)
    for i in range(s, e):
        bin_idx = x_binned[i]
        bin_ptrs[bin_idx + 1] += 1

    # Cumsum:
    pos = 0
    for b in range(1, n_bins + 1):
        pos += bin_ptrs[b]
        bin_ptrs[b] = pos

    # Place values into bins in buffers
    bin_positions = bin_ptrs.copy()
    for i in range(s, e):
        bin_idx = x_binned[i]
        dest = bin_positions[bin_idx]
        y_buf[dest] = y[i]
        bin_positions[bin_idx] += 1

    return bin_ptrs


class SparseFeature:
    def __init__(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        buf_neg: np.ndarray,
        buf_pos: np.ndarray,
    ):
        self.indices = indices
        self.values = values

    def sort(
        self, y, sample_weight, start, end, y_out, sample_weight_out, split_indices_out
    ):
        n = end - start

        # TODO: dichotomie to find those:
        indices_start = 0
        indices_end = self.indices.size

        indices = self.indices[indices_start:indices_end]
        values = self.indices[indices_start:indices_end]
        n_neg = (values < 0).sum()
        n_pos = values.size - n_neg
        neg_ptr = 0
        pos_ptr = n - n_pos
        zero_ptr = n_neg
        ind_ptr = 0

        for i in range(start, end):
            if i == indices[ind_ptr]:
                if values[ind_ptr] < 0:
                    y_out[neg_ptr] = y[i]
                    if sample_weight is not None:
                        sample_weight_out[neg_ptr] = sample_weight[i]
                    neg_ptr += 1
                else:
                    y_out[pos_ptr] = y[i]
                    if sample_weight is not None:
                        sample_weight_out[pos_ptr] = sample_weight[i]
                    pos_ptr += 1
                ind_ptr += 1
            else:
                y_out[zero_ptr] = y[i]
                if sample_weight is not None:
                    sample_weight_out[zero_ptr] = sample_weight[i]
            zero_ptr += 1
        is_neg = values < 0
        y_out[:n_neg] = y_out[np.argsort(values[is_neg])]
        y_out[n - n_pos :] = y_out[np.argsort(values[~is_neg]) + (n - n_pos)]

    def get_split_threshold(self, split_num):
        # to do
        # might require some initialization?
        ...

    def goes_left(self, idx, threshold, missing_go_left):
        # to do
        ...
