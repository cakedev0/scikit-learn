import numpy as np
import pytest

from ...tree import DecisionTreeRegressor
from ..dt import DecisionTreeRegressor as DTRegressor


def generate_data(size, d, duplication_level, ratio_missing):
    n = size // d

    if duplication_level == 0:
        X = np.random.rand(n, d)
    elif duplication_level == 1:
        X = np.random.geometric(0.05, size=(n * 2, d))
    elif duplication_level == 2:
        X = np.random.geometric(1 / 3, size=(n * 4, d))
    else:
        raise ValueError()

    X = X.astype(np.float32)
    y = X[:, ::3].sum(axis=1)
    y += np.random.rand(X.shape[0])

    if ratio_missing > 0:
        X[np.random.rand(*X.shape) < ratio_missing] = np.nan

    return X, y


@pytest.mark.parametrize("missing", ["x", "with_missing"])
@pytest.mark.parametrize("d", [1, 2, 3, 5, 10, 20, 100])
@pytest.mark.parametrize("duplication_level", [0, 1, 2])
@pytest.mark.parametrize("max_depth", [1, 2])
def test_same_split_sklearn(missing, d, duplication_level, max_depth):
    X, y = generate_data(
        size=10_000,
        d=d,
        duplication_level=duplication_level,
        ratio_missing=0 if missing == "x" else 0.1,
    )
    actual = DTRegressor(max_depth=max_depth).fit(X, y).predict(X)
    expected = DecisionTreeRegressor(max_depth=max_depth).fit(X, y).predict(X)
    assert np.allclose(actual, expected)
