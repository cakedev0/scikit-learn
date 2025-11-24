# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import threading

import numpy as np

from dt import DecisionTreeRegressor
from sklearn.utils.parallel import Parallel, delayed

MAX_INT = np.iinfo(np.int32).max


def _parallel_build_trees(
    tree,
    X,
    sorted_idx,
    cant_split,
    y,
):
    """Private function used to fit a single tree in parallel."""

    tree.fit(
        X,
        y,
        sorted_idx=sorted_idx,
        cant_split=cant_split,
    )

    return tree


class Forest:
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        n_jobs=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs or 1

    def fit(self, X, y):
        trees = [
            DecisionTreeRegressor(max_depth=self.max_depth)
            for i in range(self.n_estimators)
        ]

        # Parallel loop: we prefer the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading more efficient than multiprocessing in
        # that case. However, for joblib 0.12+ we respect any
        # parallel_backend contexts set at a higher level,
        # since correctness does not rely on using threads.
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=0,
            prefer="threads",
        )(
            delayed(_parallel_build_trees)(
                t,
                X,
                y,
                i,
                len(trees),
            )
            for i, t in enumerate(trees)
        )

        return self

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """

        # Assign chunk of trees to jobs

        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=self.n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]
