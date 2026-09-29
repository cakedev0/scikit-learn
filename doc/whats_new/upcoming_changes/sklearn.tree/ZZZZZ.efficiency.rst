- :class:`tree.DecisionTreeClassifier`, :class:`tree.DecisionTreeRegressor`,
  :class:`ensemble.RandomForestClassifier`, :class:`ensemble.RandomForestRegressor`,
  :class:`ensemble.GradientBoostingClassifier` and
  :class:`ensemble.GradientBoostingRegressor` are much faster to fit on dense data,
  typically 2 to 4 times. Numerical features are now rank-encoded once per fit
  (once for all the trees of an ensemble), and the samples of each node are sorted
  by radix sort on these ranks instead of comparison sort. The fitted trees are the
  same, up to how ties between equally good splits are broken.
  By :user:`Arthur Lacote <cakedev0>`.
