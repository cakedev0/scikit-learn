- Fixed several bugs in :class:`preprocessing.SplineTransformer`. With
  `extrapolation="linear"` and `degree` 0 or 1, all features but the first one
  were extrapolated from wrong boundaries. `order="C"` was ignored with
  `include_bias=False`. `sparse_output=True` returned float64 values for float32
  input. With `degree=0` and `extrapolation="constant"`, values out of the
  fitted range raised an error, or were encoded as zeros below the range with
  `sparse_output=True`. `degree=0` also failed with `extrapolation="periodic"`
  and `sparse_output=True`. With `handle_missing="zeros"`, infinite values were
  accepted and gave infinite or NaN outputs: they now raise an error.
  By :user:`Arthur Lacote <cakedev0>`.
