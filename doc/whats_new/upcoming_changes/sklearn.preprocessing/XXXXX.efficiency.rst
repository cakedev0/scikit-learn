- :meth:`preprocessing.SplineTransformer.transform` is faster: up to ~1.7x with
  `extrapolation="constant"` or `"linear"`, ~3x with `include_bias=False` and
  ~7x with `sparse_output=True` and `n_knots=10`. The sparse output is slower
  than before with several hundred knots. `fit` with `knots="uniform"` is also
  faster.
  By :user:`Arthur Lacote <cakedev0>`.
