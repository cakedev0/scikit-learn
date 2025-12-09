"""WIP: Unused for now, it's just to collect my thoughts"""

import numpy as np


class PresortedIndicesFeature:
    def __init__(self, sorted_indices, cant_split, ranks_buf, idx_buf):
        self.indices = sorted_indices
        self.cant_split = cant_split
        self.ranks_buf = ranks_buf
        self.idx_buf = idx_buf

    def sort_indices(self, s, e):
        return -1

    def partition(self, go_left, n_left, s, e, n_missing):
        # Stable partition using temporary buffer
        n_missing_left = 0
        end_non_missing = e - n_missing
        left_ptr = s
        right_ptr = s + n_left
        rank = 0
        for i in range(s, e):
            idx = self.indices[i]
            if go_left[idx]:  # sparse/random reads
                self.indices[left_ptr] = idx
                self.ranks_buf[left_ptr] = rank
                left_ptr += 1
                n_missing_left += i >= end_non_missing
            else:
                self.idx_buf[right_ptr] = idx
                self.ranks_buf[right_ptr] = rank
                right_ptr += 1
            if not self.cant_split[i]:
                rank += 1

        # Copy back from buffer
        for i in range(s + n_left, e):
            self.indices[i] = self.idx_buf[i]

        for i in range(s, e - 1):
            self.cant_split[i] = self.ranks_buf[i] == self.ranks_buf[i + 1]

        return n_missing_left, n_missing - n_missing_left


class SortIndicesFeature:
    def __init__(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        cant_split: np.ndarray,
        x_buf: np.ndarray,
    ):
        """indices should be initially arange, and shared with the other features"""
        self.x = x
        self.indices = indices
        self.x_buf = x_buf
        self.cant_split = cant_split

    def sort_indices(self, s, e):
        n_missing = 0
        for i in range(s, e):
            val = self.x[self.indices[i]]
            self.x_buf[i] = val
            n_missing += np.isnan(val)

        sorter = np.argsort(self.x_buf[s:e])
        sorter += s
        x_sorted = self.x_buf[sorter]
        self.cant_split[s : e - 1] = x_sorted[:-1] == x_sorted[1:]
        self.indices[s:e] = self.indices[sorter]
        return n_missing

    def partition(self, go_left, n_left, s, e, n_missing):
        while s < e:
            if go_left[s]:
                s += 1
            else:
                self.indices[s], self.indices[e] = self.indices[e], self.indices[s]
                e -= 1

        return -1, -1
