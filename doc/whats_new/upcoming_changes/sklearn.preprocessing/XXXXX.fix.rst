- Fixed several bugs in :class:`preprocessing.SplineTransformer`. With
  `extrapolation="linear"` and `degree` 0 or 1, all features but the first one
  were extrapolated from wrong boundaries. `order="C"` was ignored with
  `include_bias=False`. `sparse_output=True` returned float64 values for float32
  input. `degree=0` failed with `extrapolation="constant"` on values out of the
  fitted range, and with `extrapolation="periodic"` and `sparse_output=True`.
  By :user:`Arthur Lacote <cakedev0>`.
