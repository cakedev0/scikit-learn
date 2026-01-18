# sklearn.tree internals

Quick map of how `sklearn.tree` is wired and how other modules use it.

## File map
- `_classes.py`: public estimators + `BaseDecisionTree` (validation, builder setup).
- `_tree.pyx` / `_tree.pxd`: core `Tree` data structure, builders, prediction,
  decision path, pruning helpers; defines `DTYPE` (float32) and `DOUBLE` (float64).
- `_splitter.pyx` / `_splitter.pxd`: split search logic (best/random) using
  criteria and partitioners.
- `_partitioner.pyx` / `_partitioner.pxd`: dense/sparse sample partitioning for
  a given feature/threshold; handles missing values layout.
- `_criterion.pyx` / `_criterion.pxd`: impurity criteria (Gini/Entropy/MSE/MAE/Poisson),
  plus helper structures for efficient updates.
- `_utils.pyx` / `_utils.pxd`: small C helpers, RNG helpers, and `WeightedFenwickTree`
  used by criteria.
- `_export.py`: `export_graphviz`, `export_text`, `plot_tree`.
- `_reingold_tilford.py`: layout algorithm used by `plot_tree`.

## Core objects and data flow
### Fit path
1. `BaseDecisionTree._fit` in `_classes.py` validates input, handles class/target
   encoding, computes missing-value mask, and resolves hyperparameters.
2. A `Criterion` instance is created (classification or regression).
3. A `Splitter` instance is created (dense vs sparse, best vs random), optionally
   with monotonic constraints.
4. A `Tree` object is allocated to hold node arrays and node values.
5. A `TreeBuilder` builds the tree:
   - `DepthFirstTreeBuilder` grows nodes in DFS order (default).
   - `BestFirstTreeBuilder` grows by impurity improvement when `max_leaf_nodes` is set.
6. Optional cost-complexity pruning replaces `tree_` with a pruned tree.

### Predict path
- `Tree.predict` walks each sample to a leaf and returns per-node values.
- Estimator methods (`predict`, `predict_proba`, `apply`, `decision_path`) call `tree_`.

## Data structures
- `Tree` stores a binary tree as parallel arrays of nodes (see `_tree.pxd`).
  Key arrays include `children_left/right`, `feature`, `threshold`, `impurity`,
  `n_node_samples`, `weighted_n_node_samples`, `missing_go_to_left`, and `value`.
- Node values are stored as a `(node_count, n_outputs, max_n_classes)` array so
  the same structure supports regression and classification.

## Split search and partitioning
- `Splitter` owns the feature subset, sample indices, and the current `Criterion`.
- `_partitioner` provides dense and sparse implementations for sorting and
  partitioning samples for a candidate feature, including missing-value handling.
- `_criterion` computes impurity, impurity improvement, and node values using
  incremental updates as the split position moves.

## Missing values and monotonic constraints
- Missing values are only supported on dense inputs and when tags allow it;
  `_compute_missing_values_in_feature_mask` computes a per-feature mask.
- Missing values are grouped to the left or right during split evaluation and
  recorded in `missing_go_to_left` so prediction follows the same rule.
- Monotonic constraints are validated in `_classes.py` and passed to splitters.

## Pruning
- `_build_pruned_tree_ccp` (in `_tree.pyx`) builds a new `Tree` from a subtree
  based on `ccp_alpha`. `cost_complexity_pruning_path` in `_classes.py` exposes
  the pruning path utility.

## How other modules use `sklearn.tree`
- `sklearn.ensemble` uses `DecisionTree*` and `ExtraTree*` as base estimators:
  - `_forest.py` calls `tree._fit(...)` directly to reuse input validation and
    pass bootstrap weights; it also calls `tree.apply` and `tree.decision_path`.
  - Feature importances are aggregated via `tree.tree_.compute_feature_importances()`.
- Gradient boosting and related ensemble models also build trees via
  `DecisionTreeRegressor` and rely on the same `Tree`/`Splitter`/`Criterion`
  machinery.

## Notes for edits
- Most heavy lifting is in Cython; change signatures in `.pxd` and `.pyx` together.
- Input arrays are coerced to `DTYPE` (float32) for `X` and `DOUBLE` (float64) for `y`.
- Missing-value support is conservative; ensure changes keep the mask and
  `missing_go_to_left` consistent.
