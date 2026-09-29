"""Input validation and encoding shared by tree-based estimators.

Used by decision trees, forests, gradient boosting and histogram-based gradient
boosting.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.sparse import issparse

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_categorical_features,
    _check_n_features,
    _num_samples,
    check_array,
    validate_data,
)


def _encode_categorical_features(estimator, X, *, reset, dtype):
    """Ordinal-encode the categorical features of X, keeping the column order.

    At fit time (``reset=True``), also detect the categorical features and fit
    the encoder, stored in ``estimator._categorical_encoder`` (None if there is
    no categorical feature). X is returned unchanged if there is no categorical
    feature. Otherwise, a Fortran-ordered array of dtype ``dtype`` is returned.
    """
    if reset:
        estimator._categorical_encoder = None
        # RandomTreesEmbedding has no `categorical_features` parameter.
        if hasattr(estimator, "categorical_features"):
            estimator.is_categorical_ = _check_categorical_features(
                X, estimator.categorical_features
            )
        if getattr(estimator, "is_categorical_", None) is not None:
            estimator._categorical_encoder = OrdinalEncoder(
                dtype=dtype,
                categories="auto",
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )

    encoder = estimator._categorical_encoder
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
        _check_n_features(estimator, X, reset=False)

    is_categorical = estimator.is_categorical_
    X_out = np.empty((_num_samples(X), is_categorical.shape[0]), dtype, order="F")

    X_cat = _safe_indexing(X, is_categorical, axis=1)
    X_out[:, is_categorical] = (
        encoder.fit_transform(X_cat) if reset else encoder.transform(X_cat)
    )
    if not is_categorical.all():
        X_out[:, ~is_categorical] = check_array(
            _safe_indexing(X, ~is_categorical, axis=1),
            input_name="X",
            estimator=estimator,
            dtype=dtype,
            ensure_all_finite=False,
        )
    return X_out


def _validate_X(
    estimator,
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

    Used at fit time (``reset=True``) and predict time (``reset=False``). With
    ``check_input=False``, X is assumed to be already validated: only categorical
    features are encoded and the number of features is checked.

    ``y`` is only passed to :func:`validate_data` to raise an informative error
    if it is None at fit time. It is neither validated nor returned.
    """
    if check_input:
        # Check feature names on the original input before categorical
        # encoding converts it to a NumPy array and drops dataframe metadata.
        # The number of features is checked after check_array
        # (ensure_2d=False here), so that 1D inputs get check_array's
        # informative error.
        validate_data(
            estimator, X, y, reset=reset, skip_check_array=True, ensure_2d=False
        )

    X = _encode_categorical_features(estimator, X, reset=reset, dtype=dtype)

    if check_input:
        X = check_array(
            X,
            input_name="X",
            estimator=estimator,
            dtype=dtype,
            accept_sparse=accept_sparse,
            ensure_all_finite=ensure_all_finite,
        )
    _check_n_features(estimator, X, reset=reset)
    return X


def _get_n_categories(estimator):
    """Number of categories of each feature, -1 for numerical features.

    Missing values are not counted as a category.
    """
    n_categories = np.full(estimator.n_features_in_, -1, dtype=np.intp)
    encoder = estimator._categorical_encoder
    if encoder is not None:
        for idx, categories in zip(
            np.flatnonzero(estimator.is_categorical_), encoder.categories_
        ):
            # OrdinalEncoder places np.nan last if missing values reach fit.
            has_nan = len(categories) > 0 and is_scalar_nan(categories[-1])
            n_categories[idx] = len(categories) - has_nan
    return n_categories


class _RankEncoding:
    """Rank encoding of the numerical features of a dense X.

    Each numerical feature is encoded into the integer codes of its sorted
    unique values (NaN, if any, being the last one). Trees use it to sort
    samples by radix sort on the codes instead of comparison sort on X.

    Codes are stored in the narrowest of uint8, uint16 and uint32, with one
    Fortran-ordered array per dtype.

    Attributes
    ----------
    codes_uint8, codes_uint16, codes_uint32 : ndarray of shape (n_samples, n)
        Codes of the features encoded with each dtype.

    code_width : ndarray of shape (n_features,), dtype=uint8
        Number of bytes of the codes of each feature: 1, 2 or 4, or 0 for
        features that are not encoded (categorical features).

    code_column : ndarray of shape (n_features,), dtype=intp
        Column of each feature in the codes array of its dtype.

    max_code : ndarray of shape (n_features,), dtype=intp
        Largest code of each feature, i.e. its number of unique values minus 1.

    uniques : ndarray of shape (n_uniques,), dtype=float32
        Sorted unique values of all features, concatenated: the unique values
        of feature `j` are `uniques[uniques_offset[j]:uniques_offset[j + 1]]`.

    uniques_offset : ndarray of shape (n_features + 1,), dtype=intp
    """

    def __init__(
        self, codes, code_width, code_column, max_code, uniques, uniques_offset
    ):
        self.codes_uint8, self.codes_uint16, self.codes_uint32 = codes
        self.code_width = code_width
        self.code_column = code_column
        self.max_code = max_code
        self.uniques = uniques
        self.uniques_offset = uniques_offset

    def take(self, indices):
        """Rank encoding of the samples at `indices`.

        The unique values are kept: the codes stay valid even if some values
        are not present anymore.
        """
        codes = tuple(
            np.asfortranarray(codes.take(indices, axis=0))
            for codes in (self.codes_uint8, self.codes_uint16, self.codes_uint32)
        )
        return _RankEncoding(
            codes,
            self.code_width,
            self.code_column,
            self.max_code,
            self.uniques,
            self.uniques_offset,
        )


_CODE_DTYPES = (np.uint8, np.uint16, np.uint32)

# Below this number of values to encode, encoding features in parallel threads
# is not worth its overhead.
_MIN_VALUES_FOR_PARALLEL_ENCODING = 1_000_000


def _rank_encode_feature(values):
    """Sorted unique values of a feature and the codes of its values."""
    # NaNs are sorted last and collapsed into a single unique value.
    uniques, codes = np.unique(values, return_inverse=True)
    dtype = next(
        dtype for dtype in _CODE_DTYPES if uniques.shape[0] - 1 <= np.iinfo(dtype).max
    )
    return uniques, codes.reshape(-1).astype(dtype)


def _rank_encode(X, n_categories):
    """Rank-encode the numerical features of a dense X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=float32
        Validated training data.

    n_categories : ndarray of shape (n_features,)
        Number of categories of each feature, -1 for numerical features. Only
        numerical features are encoded.

    Returns
    -------
    rank_encoding : _RankEncoding
    """
    n_samples, n_features = X.shape
    code_width = np.zeros(n_features, dtype=np.uint8)
    code_column = np.zeros(n_features, dtype=np.intp)
    max_code = np.full(n_features, -1, dtype=np.intp)
    uniques_offset = np.zeros(n_features + 1, dtype=np.intp)
    all_uniques = []
    codes_per_dtype = {dtype: [] for dtype in _CODE_DTYPES}

    numerical_features = np.flatnonzero(np.asarray(n_categories) < 0)
    # np.unique sorts with the GIL released.
    n_threads = (
        min(_openmp_effective_n_threads(), numerical_features.shape[0])
        if n_samples * numerical_features.shape[0] >= _MIN_VALUES_FOR_PARALLEL_ENCODING
        else 1
    )
    encoded_features = Parallel(n_jobs=n_threads, prefer="threads")(
        delayed(_rank_encode_feature)(X[:, j]) for j in numerical_features
    )

    for j, (uniques, codes) in zip(numerical_features, encoded_features):
        code_width[j] = codes.dtype.itemsize
        code_column[j] = len(codes_per_dtype[codes.dtype.type])
        max_code[j] = uniques.shape[0] - 1
        codes_per_dtype[codes.dtype.type].append(codes)
        all_uniques.append(uniques)
        uniques_offset[j + 1] = uniques.shape[0]
    uniques_offset = np.cumsum(uniques_offset)

    codes = tuple(
        np.empty((n_samples, len(codes_list)), dtype=dtype, order="F")
        for dtype, codes_list in codes_per_dtype.items()
    )
    for dtype_codes, codes_list in zip(codes, codes_per_dtype.values()):
        for column, column_codes in enumerate(codes_list):
            # Contiguous copy into a Fortran-ordered array: cheap.
            dtype_codes[:, column] = column_codes
    uniques = (
        np.concatenate(all_uniques).astype(np.float32, copy=False)
        if all_uniques
        else np.empty(0, dtype=np.float32)
    )
    return _RankEncoding(
        codes, code_width, code_column, max_code, uniques, uniques_offset
    )
