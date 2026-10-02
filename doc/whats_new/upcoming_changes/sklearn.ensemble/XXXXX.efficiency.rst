- The `fit` method of :class:`ensemble.RandomForestClassifier`,
  :class:`ensemble.RandomForestRegressor`, :class:`ensemble.ExtraTreesClassifier`,
  :class:`ensemble.ExtraTreesRegressor` and :class:`ensemble.RandomTreesEmbedding`
  is faster with many threads (`n_jobs`), especially on free-threaded Python:
  trees are fitted with a thread pool of lower overhead than joblib's. A joblib
  backend set with :func:`joblib.parallel_config` is still respected.
  By :user:`Arthur Lacote <cakedev0>`.
