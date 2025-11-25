import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.presortedtree.dt import DecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def generate_data(size, d, duplication_level, ratio_missing, weights, clf):
    n = size // d

    if duplication_level == 0:
        X = np.random.rand(n, d)
    elif duplication_level == 1:
        X = np.random.geometric(0.05, size=(n, d))
    elif duplication_level == 2:
        X = np.random.geometric(1 / 3, size=(n, d))
    else:
        raise ValueError()

    X = X.astype(np.float32)
    y = X[:, ::3].sum(axis=1)
    y += np.random.rand(n)

    if ratio_missing > 0:
        X[np.random.rand(*X.shape) < ratio_missing] = np.nan

    if clf:
        y = y < np.median(y)

    return {"X": X, "y": y, "sample_weight": np.random.rand(n) if weights else None}


@pytest.mark.parametrize("criterion", ["squared_error", "gini"])
@pytest.mark.parametrize("weights", ["x", "with_weights"])
@pytest.mark.parametrize("missing", ["x", "with_missing"])
@pytest.mark.parametrize("d", [1, 2, 3, 5, 10, 20])
@pytest.mark.parametrize("duplication_level", [0, 1, 2])
@pytest.mark.parametrize("max_depth", [1, 2])
def test_same_split_sklearn(
    weights, missing, d, duplication_level, max_depth, criterion
):
    clf = criterion == "gini"
    kwargs = generate_data(
        size=np.random.randint(5_000, 20_000),
        d=d,
        duplication_level=duplication_level,
        ratio_missing=0 if missing == "x" else 0.1,
        weights=weights != "x",
        clf=clf,
    )
    actual = (
        DecisionTree(max_depth=max_depth, criterion=criterion)
        .fit(**kwargs)
        .predict(kwargs["X"])
    )

    Tree = DecisionTreeClassifier if clf else DecisionTreeRegressor
    expected = (
        Tree(max_depth=max_depth, criterion=criterion)
        .fit(**kwargs)
        .predict(kwargs["X"])
    )
    assert_allclose(actual, expected)
