- :meth:`preprocessing.SplineTransformer.transform` is ~5x faster on a single
  thread, and is now parallelized with OpenMP. It is more than 15x faster with
  `sparse_output=True` or `include_bias=False`. `fit` is also faster with
  `knots="uniform"`.
  By :user:`Arthur Lacote <cakedev0>`.
