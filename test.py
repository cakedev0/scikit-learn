import numpy as np

from bench import benchmark_tree, kwargs_product
from dt import DecisionTreeRegressor as CustomDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor


def test_r2_vs_sklearn():
    for kwargs in kwargs_product(
        size=[100_000],
        d=[1, 5, 20],
        duplication_level=[0, 1, 2],
        max_depth=[1, 2, 3, 6, 9, 12, None],
    ):
        expected = benchmark_tree(DecisionTreeRegressor, **kwargs, n_fit=5)["score"]
        actual = benchmark_tree(CustomDecisionTreeRegressor, **kwargs, n_fit=5)["score"]
        print(kwargs, expected, actual)
        assert np.isclose(expected, actual, atol=0.05), (kwargs, expected, actual)


if __name__ == "__main__":
    test_r2_vs_sklearn()
