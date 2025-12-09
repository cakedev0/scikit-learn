# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Optimized sorting algorithms for paired arrays.

This module implements an introsort (introspective sort) algorithm that sorts
two arrays simultaneously, maintaining correspondence between elements. The
implementation is optimized with:

- Introsort: Hybrid algorithm combining quicksort, heapsort, and insertion sort
- 2-way partitioning: Standard quicksort partitioning scheme
- Median-of-3 pivot selection: Better performance on partially sorted data
- Insertion sort for small partitions (n <= 15): Reduced overhead
- Heapsort fallback: Prevents worst-case O(n²) when max depth is exceeded
- Numba JIT compilation: Near-C performance

Algorithm: Based on Musser's Introsort (SP&E, 1997) and the implementation
in sklearn/tree/_partitioner.pyx.
"""

import numpy as np
from numba import njit

# Threshold for switching to insertion sort
INSERTION_SORT_THRESHOLD = 15


@njit
def insertion_sort(x, y, left, right):
    """Sort arrays using insertion sort.

    Efficient for small arrays (n <= 15).

    Parameters
    ----------
    x : array
        Primary array to sort
    y : array
        Secondary array to sort alongside x
    left : int
        Left boundary
    right : int
        Right boundary
    """
    for i in range(left + 1, right + 1):
        x_key = x[i]
        y_key = y[i]
        j = i - 1

        while j >= left and x[j] > x_key:
            x[j + 1] = x[j]
            y[j + 1] = y[j]
            j -= 1

        x[j + 1] = x_key
        y[j + 1] = y_key


@njit
def median_of_three(x, left, mid, right):
    """Select median of three values as pivot.

    Parameters
    ----------
    x : array
        Array containing values
    left : int
        Index of left element
    mid : int
        Index of middle element
    right : int
        Index of right element

    Returns
    -------
    float
        Median value among the three
    """
    a, b, c = x[left], x[mid], x[right]

    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


@njit
def sift_down(x, y, start, end):
    """Restore heap order by moving the max element to start.

    Parameters
    ----------
    x : array
        Primary array
    y : array
        Secondary array
    start : int
        Start of heap
    end : int
        End of heap
    """
    root = start
    while True:
        child = root * 2 + 1

        # Find max of root, left child, right child
        maxind = root
        if child < end and x[maxind] < x[child]:
            maxind = child
        if child + 1 < end and x[maxind] < x[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            x[root], x[maxind] = x[maxind], x[root]
            y[root], y[maxind] = y[maxind], y[root]
            root = maxind


@njit
def heapsort(x, y, left, right):
    """Sort arrays using heapsort.

    Used as fallback when max depth is exceeded in introsort.

    Parameters
    ----------
    x : array
        Primary array to sort
    y : array
        Secondary array to sort alongside x
    left : int
        Left boundary
    right : int
        Right boundary
    """
    n = right - left + 1

    # Heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(x[left:], y[left:], start, end)
        if start == 0:
            break
        start -= 1

    # Sort by shrinking the heap
    end = n - 1
    while end > 0:
        x[left:][0], x[left:][end] = x[left:][end], x[left:][0]
        y[left:][0], y[left:][end] = y[left:][end], y[left:][0]
        sift_down(x[left:], y[left:], 0, end)
        end -= 1


@njit
def introsort(x, y, left, right, maxd):
    """Introsort: quicksort with heapsort fallback.

    Uses 2-way partitioning with median-of-3 pivot selection.
    Switches to heapsort when max depth is exceeded.

    Parameters
    ----------
    x : array
        Primary array to sort
    y : array
        Secondary array to sort alongside x
    left : int
        Left boundary
    right : int
        Right boundary
    maxd : int
        Maximum recursion depth before switching to heapsort
    """
    while right - left > INSERTION_SORT_THRESHOLD:
        if maxd <= 0:
            # Max depth exceeded, switch to heapsort
            heapsort(x, y, left, right)
            return
        maxd -= 1

        # Median-of-3 pivot selection
        n = right - left + 1
        mid = left + n // 2
        pivot = median_of_three(x, left, mid, right)

        # 2-way partition inline
        i = left
        j = right

        while i <= j:
            while i <= j and x[i] <= pivot:
                i += 1
            while i <= j and x[j] > pivot:
                j -= 1
            if i < j:
                x[i], x[j] = x[j], x[i]
                y[i], y[j] = y[j], y[i]
                i += 1
                j -= 1

        # Recursively sort left partition, then iterate on right
        introsort(x, y, left, j, maxd)
        left = j + 1

    # Use insertion sort for small partitions
    if right - left > 0:
        insertion_sort(x, y, left, right)


@njit
def sort(x, y):
    """Sort both x and y arrays by ascending x values using introsort.

    This is an optimized introsort (introspective sort) implementation:
    - Uses quicksort with 2-way partitioning and median-of-3 pivot selection
    - Switches to heapsort when max depth is exceeded (prevents worst-case O(n²))
    - Uses insertion sort for small partitions (n <= 15) for efficiency

    Both arrays are sorted in-place, maintaining correspondence between
    x and y elements.

    Parameters
    ----------
    x : ndarray
        Primary array to sort by (modified in-place)
    y : ndarray
        Secondary array to sort alongside x (modified in-place)

    Examples
    --------
    >>> x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> y = np.array([30, 10, 40, 11, 50])
    >>> sort(x, y)
    >>> print(x)  # [1.0, 1.0, 3.0, 4.0, 5.0]
    >>> print(y)  # [10, 11, 30, 40, 50]
    """
    n = len(x)
    if n <= 1:
        return

    if n <= INSERTION_SORT_THRESHOLD:
        insertion_sort(x, y, 0, n - 1)
    else:
        # Max depth: 2 * log2(n)
        maxd = 2 * int(np.log2(n))
        introsort(x, y, 0, n - 1, maxd)


@njit
def simple_sort(x, y):
    sorter = np.argsort(x)
    x[:] = x[sorter]
    y[:] = y[sorter]
