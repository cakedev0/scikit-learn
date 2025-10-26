# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.externals import array_api_extra as xpx


def _weighted_percentile(
    array, sample_weight, percentile_rank=50, average=False, xp=None
):
    """Compute the weighted percentile.

    Implement an array API compatible (weighted version) of NumPy's 'inverted_cdf'
    method when `average=False` (default) and 'averaged_inverted_cdf' when
    `average=True`.

    For an array ordered by increasing values, when the percentile lies exactly on a
    data point:

    * 'inverted_cdf' takes the exact data point.
    * 'averaged_inverted_cdf' takes the average of the exact data point and the one
      above it (this means it gives the same result as `median` for unit weights).

    E.g., for the array [1, 2, 3, 4] the percentile rank at each data point would
    be [25, 50, 75, 100]. Percentile rank 50 lies on '2'. 'average_inverted_cdf'
    computes the average of '2' and '3', making it 'symmetrical' because if you
    reverse the array, rank 50 would fall on '3'. It also matches 'median'.
    On the other hand, 'inverted_cdf', which does not satisfy the symmetry property,
    would give '2'.

    When the requested percentile lies between two data points, both methods return
    the higher data point.
    E.g., for the array [1, 2, 3, 4, 5] the percentile rank at each data point would
    be [20, 40, 60, 80, 100]. Percentile rank 50, lies between '2' and '3'. Taking the
    higher data point is symmetrical because if you reverse the array, 50 would lie
    between '4' and '3'. Both methods match median in this case.

    If `array` is a 2D array, the `values` are selected along axis 0.

    `NaN` values are ignored by setting their weights to 0. If `array` is 2D, this
    is done in a column-isolated manner: a `NaN` in the second column, does not impact
    the percentile computed for the first column even if `sample_weight` is 1D.

        .. versionchanged:: 0.24
            Accepts 2D `array`.

        .. versionchanged:: 1.7
            Supports handling of `NaN` values.

        .. versionchanged:: 1.8
            Supports `average`, which calculates percentile using the
            "averaged_inverted_cdf" method.

    Parameters
    ----------
    array : 1D or 2D array
        Values to take the weighted percentile of.

    sample_weight: 1D or 2D array
        Weights for each value in `array`. Must be same shape as `array` or of shape
        `(array.shape[0],)`.

    percentile_rank: scalar or 1D array, default=50
        The probability level(s) of the percentile(s) to compute, in percent. Must be
        between 0 and 100. If a 1D array, computes multiple percentiles.

    average : bool, default=False
        If `True`, uses the "averaged_inverted_cdf" quantile method, otherwise
        defaults to "inverted_cdf". "averaged_inverted_cdf" is symmetrical with
        unit `sample_weight`, such that the total of `sample_weight` below or equal to
        `_weighted_percentile(percentile_rank)` is the same as the total of
        `sample_weight` above or equal to `_weighted_percentile(100-percentile_rank)`.
        This symmetry is not guaranteed with non-unit weights.

    xp : array_namespace, default=None
        The standard-compatible namespace for `array`. Default: infer.

    Returns
    -------
    percentile : scalar, 1D array, or 2D array
        Weighted percentile at the requested probability level(s).
        If `array` is 1D and `percentile_rank` is scalar, returns a scalar.
        If `array` is 2D and `percentile_rank` is scalar, returns a 1D array
            of shape `(array.shape[1],)`
        If `array` is 1D and `percentile_rank` is 1D, returns a 1D array
            of shape `(percentile_rank.shape[0],)`
        If `array` is 2D and `percentile_rank` is 1D, returns a 2D array
            of shape `(percentile_rank.shape[0], array.shape[1])`
    """
    method = "averaged_inverted_cdf" if average else "inverted_cdf"
    return xpx.quantile(
        array,
        percentile_rank / 100,
        axis=0,
        method=method,
        weights=sample_weight,
        xp=xp,
        nan_policy="omit",
    )
