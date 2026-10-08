- :meth:`preprocessing.SplineTransformer.transform` is faster: ~1.5x in common
  settings, ~3x with `include_bias=False` and ~7x with `sparse_output=True`.
  `fit` is also faster with `knots="uniform"`.
  By :user:`Arthur Lacote <cakedev0>`.
