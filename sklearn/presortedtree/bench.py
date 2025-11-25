# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from time import perf_counter

import numpy as np

from sklearn.metrics import r2_score
from sklearn.presortedtree.dt import DecisionTree as CustomDecisionTreeRegressor
from sklearn.presortedtree.forest import Forest


def benchmark_tree(
    cls,
    size: int = 3_000_000,  # usually takes around ~1s per fit
    d: int = 20,
    duplication_level: int = 0,
    n_fit: int = 11,
    n_skip: int = 1,
    is_clf: bool = False,
    **kwargs,
):
    n = size // d // (kwargs.get("max_depth") or 20)
    dts = []
    scores = []
    for _ in range(n_fit):
        if duplication_level == 0:
            X = np.random.rand(n, d)
        elif duplication_level == 1:
            X = np.random.geometric(0.05, size=(n * 2, d))
        elif duplication_level == 2:
            X = np.random.geometric(1 / 3, size=(n * 4, d))
        else:
            raise ValueError()
        y = np.random.rand(X.shape[0]) + X.sum(axis=1)
        X = X.astype(np.float32)
        # X[np.random.rand(*X.shape) < 0.05] = np.nan
        if is_clf:
            raise NotImplementedError()
        tree = cls(**kwargs)
        if isinstance(tree, CustomDecisionTreeRegressor):
            args = tree.preprocess_Xy(X, y)
        else:
            args = (X, y)
        t = perf_counter()
        tree.fit(*args)
        dts.append(perf_counter() - t)
        y_pred = tree.predict(X)
        scores.append(r2_score(y, y_pred))
    dts = dts[n_skip:]
    return {
        "cls": cls,
        **kwargs,
        "dt": np.mean(dts),
        "dt_min": np.min(dts),
        "dt_max": np.max(dts),
        "score": np.mean(scores),
    }


def kwargs_product(**kwargs):
    for args in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), args))


if __name__ == "__main__":
    for kwargs in kwargs_product(
        duplication_level=[0],
        max_depth=[12],  # [1, 2, 3, 6, 9, 12, None],
        cls=[Forest],  # [partial(RandomForestRegressor, bootstrap=False), Forest],
        n_jobs=[1, 2, 4, 8, 12],
    ):
        kwargs["n_estimators"] = kwargs["n_jobs"] * 20
        out = benchmark_tree(**kwargs, size=10_000_000, d=10, n_fit=4, n_skip=1)
        dt, dt_min, dt_max = out["dt"], out["dt_min"], out["dt_max"]
        cls = out["cls"]
        name = (
            "pre-sorted"
            if cls in (CustomDecisionTreeRegressor, Forest)
            else "scikit-learn"
        )
        score = out["score"]
        print(name, f"{dt:.2f} [{dt_min:.2f} - {dt_max:.2f}] (score: {score:.2f})")
