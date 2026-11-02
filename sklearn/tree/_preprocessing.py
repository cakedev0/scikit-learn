"""Input validation and categorical encoding shared by tree-based estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import narwhals.stable.v2 as nw
import numpy as np
from scipy.sparse import issparse

from sklearn.base import is_classifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _check_n_features,
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    validate_data,
)


def _check_categorical_features(X, categorical_features):
    """Check and validate categorical features in X

    Parameters
    ----------
    X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
        Input data.

    categorical_features : array-like of {bool, int, str} of shape (n_features) \
            or shape (n_categorical_features,), default='from_dtype'
        Indicates the categorical features in `X`.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.
        - str array-like: names of categorical features (assuming the training
          data has feature names).
        - `"from_dtype"`: dataframe columns with dtype "Categorical" and "Enum" are
          considered to be categorical features. The input must be a dataframe
          supported by narwhals, meaning :func:`narwhals.from_native` must work.
          This is the case, for instance, for pandas and polars DataFrames.

    Return
    ------
    is_categorical : ndarray of shape (n_features,) or None, dtype=bool
        Indicates whether a feature is categorical. If no feature is
        categorical, this is None.
    """
    if nw.dependencies.is_into_dataframe(X):
        X = nw.from_native(X)
        dtypes = X.schema.dtypes()
        X_is_dataframe = True
        categorical_columns_mask = np.asarray(
            [d in (nw.Categorical, nw.Enum) for d in dtypes]
        )
    else:
        X_is_dataframe = False
        categorical_columns_mask = None

    categorical_by_dtype = (
        isinstance(categorical_features, str) and categorical_features == "from_dtype"
    )
    no_categorical_dtype = categorical_features is None or (
        categorical_by_dtype and not X_is_dataframe
    )

    if no_categorical_dtype:
        return None

    if categorical_by_dtype and X_is_dataframe:
        categorical_features = categorical_columns_mask
    else:
        categorical_features = np.asarray(categorical_features)

    if categorical_features.size == 0:
        return None

    if categorical_features.dtype.kind not in ("i", "b", "U", "O"):
        raise ValueError(
            "categorical_features must be an array-like of bool, int or "
            f"str, got: {categorical_features.dtype.name}."
        )

    if categorical_features.dtype.kind == "O":
        types = set(type(f) for f in categorical_features)
        if types != {str}:
            raise ValueError(
                "categorical_features must be an array-like of bool, int or "
                f"str, got: {', '.join(sorted(t.__name__ for t in types))}."
            )

    n_features = X.shape[1]
    # At this point `validate_data` was not called yet because we use the original
    # dtypes to discover the categorical features. Thus `feature_names_in_`
    # is not defined yet.
    feature_names_in_ = getattr(X, "columns", None)

    if categorical_features.dtype.kind in ("U", "O"):
        # check for feature names
        if feature_names_in_ is None:
            raise ValueError(
                "categorical_features should be passed as an array of "
                "integers or as a boolean mask when the model is fitted "
                "on data without feature names."
            )
        is_categorical = np.zeros(n_features, dtype=bool)
        feature_names = list(feature_names_in_)
        for feature_name in categorical_features:
            try:
                is_categorical[feature_names.index(feature_name)] = True
            except ValueError as e:
                raise ValueError(
                    f"categorical_features has an item value '{feature_name}' "
                    "which is not a valid feature name of the training "
                    f"data. Observed feature names: {feature_names}"
                ) from e
    elif categorical_features.dtype.kind == "i":
        # check for categorical features as indices
        if (
            np.max(categorical_features) >= n_features
            or np.min(categorical_features) < 0
        ):
            raise ValueError(
                "categorical_features set as integer "
                "indices must be in [0, n_features - 1]"
            )
        is_categorical = np.zeros(n_features, dtype=bool)
        is_categorical[categorical_features] = True
    else:
        if categorical_features.shape[0] != n_features:
            raise ValueError(
                "categorical_features set as a boolean mask "
                "must have shape (n_features,), got: "
                f"{categorical_features.shape}"
            )
        is_categorical = categorical_features

    if not np.any(is_categorical):
        return None
    return is_categorical


class _CategoricalEncodingMixin:
    """Input validation and categorical encoding shared by tree-based estimators.

    Estimators with a ``categorical_features`` parameter get their categorical
    features ordinal-encoded: fitting sets ``is_categorical_`` and
    ``_categorical_encoder``.
    """

    # Set at fit time if X has categorical features. Trees fitted by an
    # ensemble get already encoded X through `_fit_validated` and keep None.
    _categorical_encoder = None

    def _encode_categorical_features(self, X, *, reset, dtype):
        """Ordinal-encode the categorical features of X, keeping column order.

        At fit time (``reset=True``), also detect the categorical features and
        fit the encoder, stored in ``self._categorical_encoder`` (None if there
        is no categorical feature). X is returned unchanged if there is no
        categorical feature. Otherwise, a Fortran-ordered array of dtype
        ``dtype`` is returned.
        """
        if reset:
            self._categorical_encoder = None
            # RandomTreesEmbedding and gradient boosting have no
            # `categorical_features` parameter.
            if hasattr(self, "categorical_features"):
                self.is_categorical_ = _check_categorical_features(
                    X, self.categorical_features
                )
            if getattr(self, "is_categorical_", None) is not None:
                self._categorical_encoder = OrdinalEncoder(
                    dtype=dtype,
                    categories="auto",
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                )

        encoder = self._categorical_encoder
        if encoder is None:
            return X

        if issparse(X):
            raise NotImplementedError(
                "Categorical features not supported with sparse inputs"
            )

        if not (hasattr(X, "__array__") or hasattr(X, "__dataframe__")):
            # Lists of lists do not support column indexing.
            X = check_array(X, dtype=object, ensure_all_finite=False)
        if not reset:
            _check_n_features(self, X, reset=False)

        is_categorical = self.is_categorical_
        X_out = np.empty((_num_samples(X), is_categorical.shape[0]), dtype, order="F")

        X_cat = _safe_indexing(X, is_categorical, axis=1)
        X_out[:, is_categorical] = (
            encoder.fit_transform(X_cat) if reset else encoder.transform(X_cat)
        )
        if not is_categorical.all():
            X_out[:, ~is_categorical] = check_array(
                _safe_indexing(X, ~is_categorical, axis=1),
                input_name="X",
                estimator=self,
                dtype=dtype,
                ensure_all_finite=False,
            )
        return X_out

    def _validate_X(
        self,
        X,
        y="no_validation",
        *,
        reset,
        dtype,
        accept_sparse=False,
        ensure_all_finite=True,
        check_input=True,
    ):
        """Validate X and ordinal-encode its categorical features.

        Used at fit time (``reset=True``) and predict time (``reset=False``).
        With ``check_input=False``, X is assumed to be already validated: only
        categorical features are encoded and the number of features is checked.

        ``y`` is only passed to :func:`validate_data` to raise an informative
        error if it is None at fit time. It is neither validated nor returned.
        """
        if check_input:
            # Check feature names on the original input before categorical
            # encoding converts it to a NumPy array and drops dataframe metadata.
            # The number of features is checked after check_array
            # (ensure_2d=False here), so that 1D inputs get check_array's
            # informative error.
            validate_data(
                self, X, y, reset=reset, skip_check_array=True, ensure_2d=False
            )

        X = self._encode_categorical_features(X, reset=reset, dtype=dtype)

        if check_input:
            X = check_array(
                X,
                input_name="X",
                estimator=self,
                dtype=dtype,
                accept_sparse=accept_sparse,
                ensure_all_finite=ensure_all_finite,
            )
        _check_n_features(self, X, reset=reset)
        return X

    def _get_n_categories(self):
        """Number of categories of each feature, -1 for numerical features.

        Missing values are not counted as a category.
        """
        n_categories = np.full(self.n_features_in_, -1, dtype=np.intp)
        encoder = self._categorical_encoder
        if encoder is not None:
            for idx, categories in zip(
                np.flatnonzero(self.is_categorical_), encoder.categories_
            ):
                # OrdinalEncoder places np.nan last if missing values reach fit.
                has_nan = len(categories) > 0 and is_scalar_nan(categories[-1])
                n_categories[idx] = len(categories) - has_nan
        return n_categories


class _TreeInputValidationMixin(_CategoricalEncodingMixin):
    """Input validation shared by decision trees, forests and gradient boosting.

    Requires a ``criterion`` parameter.
    """

    def _validate_and_preprocess_input(
        self, X, y=None, sample_weight=None, *, reset, check_input
    ):
        """Validate the inputs and encode the categorical features of X.

        Used at fit time (``reset=True``) and predict time (``reset=False``), by
        trees, forests and gradient boosting alike. With ``check_input=False``,
        the inputs are assumed to be already validated: only categorical features
        are encoded and the number of features is checked.

        Returns
        -------
        X : {ndarray, sparse matrix}
            Validated float32 X, with categorical features encoded. Returned
            alone at predict time.

        y : ndarray
            Validated y. Only returned at fit time.

        sample_weight : ndarray of shape (n_samples,) or None
            Validated sample weights. Only returned at fit time.

        fit_kwargs : dict
            The `missing_values_in_feature_mask` and `categorical_counts`
            arguments of `_fit_validated`. Only returned at fit time.
        """
        if check_input and reset and issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X = self._validate_X(
            X,
            # At fit time, validate_data raises an informative error if y is None.
            y if reset else "no_validation",
            reset=reset,
            check_input=check_input,
            dtype=np.float32,
            # Trees are built from CSC matrices and predict from CSR matrices.
            accept_sparse="csc" if reset else "csr",
            ensure_all_finite=(
                "allow-nan"
                if not issparse(X) and self.__sklearn_tags__().input_tags.allow_nan
                else True
            ),
        )
        if check_input and issparse(X):
            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError("No support for np.int64 index based sparse matrices")
            if reset:
                # Sort once for all trees of an ensemble.
                X.sort_indices()

        if not reset:
            return X

        if check_input:
            # We can't use validate_data(multi_output=True) because that would
            # allow y to be csr.
            y = check_array(
                y, input_name="y", estimator=self, ensure_2d=False, dtype=None
            )
            check_consistent_length(X, y)
            if is_classifier(self):
                check_classification_targets(y)
            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X)

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is "
                        "not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not strictly positive which "
                        "is necessary for Poisson regression."
                    )

        fit_kwargs = {
            "missing_values_in_feature_mask": np.isnan(
                np.asarray(X.sum(axis=0)).ravel()
            ),
            "categorical_counts": self._get_n_categories(),
        }
        return X, y, sample_weight, fit_kwargs
