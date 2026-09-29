# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.utils._sorting import _py_radix_sort, _py_simultaneous_sort


@pytest.mark.parametrize("kind", ["2-way", "3-way"])
def test_simultaneous_sort_correctness(kind):
    rng = np.random.default_rng(0)
    for x in [
        rng.uniform(size=3),
        rng.uniform(size=10),
        rng.uniform(size=1000),
        # with duplicates:
        rng.geometric(0.2, size=100).astype("float32"),
        rng.integers(0, 2, size=1000).astype("float32"),
    ]:
        n = x.size
        ind = np.arange(n, dtype=np.intp)
        x_sorted = x.copy()
        _py_simultaneous_sort(x_sorted, ind, n, use_three_way_partition=kind == "3-way")
        assert (x_sorted[:-1] <= x_sorted[1:]).all()
        assert_array_equal(x[ind], x_sorted)
        assert_array_equal(np.sort(ind), np.arange(n, dtype=np.intp))


@pytest.mark.parametrize("kind", ["2-way", "3-way"])
def test_simultaneous_sort_no_stackoverflow(kind):
    """Check that worst case inputs do not exceed the recursion stack limit."""
    n = 1_000_000
    # worst case pattern (i.e. triggers the quadratic path)
    # for naive 2-way partitioning quicksort:
    values = np.zeros(n)
    indices = np.arange(n, dtype=np.intp)
    _py_simultaneous_sort(
        values, indices, values.shape[0], use_three_way_partition=kind == "3-way"
    )

    # worst case pattern for the better (numpy-style) 2-way partitioning:
    values = np.roll(np.arange(n), -1).astype(np.float32)
    indices = np.arange(n, dtype=np.intp)
    _py_simultaneous_sort(
        values, indices, values.shape[0], use_three_way_partition=kind == "3-way"
    )

    # worst case pattern for the 3-way partitioning quicksort
    # with median-of-3 pivot:
    k = n // 2
    values = np.array(
        [i if i % 2 == 1 else k + i - 1 for i in range(1, k + 1)]
        + [i for i in range(1, 2 * k + 1) if i % 2 == 0]
    ).astype(np.float64)
    # (very unlikely in real-world non-adversarial data)
    indices = np.arange(n, dtype=np.intp)
    assert values.size == indices.size
    _py_simultaneous_sort(
        values, indices, values.shape[0], use_three_way_partition=kind == "3-way"
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 63, 64, 65, 100, 1000, 100_000])
def test_radix_sort_correctness_and_stability(dtype, n):
    """Cover both sides of the insertion sort threshold and several widths."""
    rng = np.random.default_rng(0)
    # Many duplicated values, to check stability.
    values = rng.integers(0, 11, size=n).astype(dtype)
    indices = np.arange(n, dtype=np.intp)
    values_sorted = values.copy()

    _py_radix_sort(values_sorted, indices)

    assert_array_equal(values_sorted, np.sort(values))
    assert_array_equal(indices, np.argsort(values, kind="stable"))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_radix_sort_full_range(dtype):
    """Check correctness using the full value range of each dtype."""
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    values = rng.integers(info.min, info.max, size=5000, endpoint=True).astype(dtype)
    indices = np.arange(values.shape[0], dtype=np.intp)
    values_sorted = values.copy()

    _py_radix_sort(values_sorted, indices)

    assert_array_equal(values_sorted, np.sort(values))
    assert_array_equal(indices, np.argsort(values, kind="stable"))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("max_value", [0, 1, 9, 200])
@pytest.mark.parametrize("n", [0, 1, 2, 63, 64, 65, 1000])
def test_radix_sort_max_value(dtype, max_value, n):
    """A tight `max_value` upper bound still gives a correct, stable sort."""
    rng = np.random.default_rng(0)
    values = rng.integers(0, max_value, size=n, endpoint=True).astype(dtype)
    indices = np.arange(n, dtype=np.intp)
    values_sorted = values.copy()

    _py_radix_sort(values_sorted, indices, max_value)

    assert_array_equal(values_sorted, np.sort(values))
    assert_array_equal(indices, np.argsort(values, kind="stable"))
