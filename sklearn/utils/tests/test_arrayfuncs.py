import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.utils import arrayfuncs
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import (
    _all_with_any_reduction_axis_1,
    _ascontiguousarray,
    _asfortranarray,
    min_pos,
)


def test_min_pos(global_random_seed):
    # Check that min_pos returns a positive value and that it's consistent
    # between float and double
    X = np.random.RandomState(global_random_seed).randn(100)

    min_double = min_pos(X)
    min_float = min_pos(X.astype(np.float32))

    assert_allclose(min_double, min_float)
    assert min_double >= 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_pos_no_positive(dtype):
    # Check that the return value of min_pos is the maximum representable
    # value of the input dtype when all input elements are <= 0 (#19328)
    X = np.full(100, -1.0).astype(dtype, copy=False)

    assert min_pos(X) == np.finfo(dtype).max


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("value", [0, 1.5, -1])
def test_all_with_any_reduction_axis_1(dtype, value):
    # Check that return value is False when there is no row equal to `value`
    X = np.arange(12, dtype=dtype).reshape(3, 4)
    assert not _all_with_any_reduction_axis_1(X, value=value)

    # Make a row equal to `value`
    X[1, :] = value
    assert _all_with_any_reduction_axis_1(X, value=value)


@pytest.mark.parametrize(
    "convert, numpy_convert, order",
    [
        (_asfortranarray, np.asfortranarray, "F"),
        (_ascontiguousarray, np.ascontiguousarray, "C"),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_threads", [1, 3, 8])
# Shapes smaller than a tile, not multiple of the tile size, with one row or
# one column, empty
@pytest.mark.parametrize(
    "shape", [(5, 3), (100, 70), (33, 1000), (1, 50), (50, 1), (0, 5)]
)
def test_change_memory_order(
    monkeypatch, convert, numpy_convert, order, dtype, n_threads, shape
):
    """Check that _asfortranarray and _ascontiguousarray match numpy."""
    # Parallel conversion even for these small inputs
    monkeypatch.setattr(arrayfuncs, "_PARALLEL_MIN_SIZE", 0)
    monkeypatch.setattr(arrayfuncs, "_MIN_SIZE_PER_THREAD", 1)
    rng = np.random.RandomState(0)
    other_order = "C" if order == "F" else "F"
    X = np.asarray(rng.uniform(size=shape), dtype=dtype, order=other_order)

    X_out = convert(X, n_threads)

    assert X_out.dtype == dtype
    assert X_out.flags[f"{order}_CONTIGUOUS"]
    assert_array_equal(X_out, X)
    # Same object returned as numpy, without copy when X is in both orders
    assert (X_out is X) == (numpy_convert(X) is X)

    # Already in the requested order: no copy
    assert convert(X_out, n_threads) is X_out


@pytest.mark.parametrize("convert", [_asfortranarray, _ascontiguousarray])
def test_change_memory_order_numpy_fallback(monkeypatch, convert):
    """Check the inputs the parallel copy doesn't handle: they are left to numpy."""
    monkeypatch.setattr(arrayfuncs, "_PARALLEL_MIN_SIZE", 0)
    monkeypatch.setattr(arrayfuncs, "_MIN_SIZE_PER_THREAD", 1)
    rng = np.random.RandomState(0)
    # Strided view, integer dtype, 3D array
    for X in [rng.uniform(size=(100, 70))[::2, ::3], np.arange(600).reshape(30, 20)]:
        assert_array_equal(convert(X, 4), X)
        assert_array_equal(convert(X.T, 4), X.T)
    X = rng.uniform(size=(5, 6, 7))
    assert_array_equal(convert(X, 4), X)


@pytest.mark.parametrize(
    "convert, order", [(_asfortranarray, "C"), (_ascontiguousarray, "F")]
)
def test_change_memory_order_small_input(monkeypatch, convert, order):
    """Check that small inputs are converted by numpy, and use fewer threads."""
    monkeypatch.setattr(arrayfuncs, "_PARALLEL_MIN_SIZE", 200)
    monkeypatch.setattr(arrayfuncs, "_MIN_SIZE_PER_THREAD", 50)
    calls = []
    kernel = arrayfuncs._transpose_copy

    def spy_kernel(*args):
        calls.append(args[-1])
        kernel(*args)

    monkeypatch.setattr(arrayfuncs, "_transpose_copy", spy_kernel)
    rng = np.random.RandomState(0)

    # 190 values: below _PARALLEL_MIN_SIZE, even with 3 threads of 50 values
    X = np.asarray(rng.uniform(size=(19, 10)), order=order)
    assert_array_equal(convert(X, 8), X)
    assert calls == []

    # 250 values: 5 threads of 50 values
    X = np.asarray(rng.uniform(size=(25, 10)), order=order)
    assert_array_equal(convert(X, 8), X)
    assert calls == [5]
