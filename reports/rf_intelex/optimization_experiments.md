# RandomForest optimization experiments

## `rfopt/disable_three_way_sort`

Branch commit: `6319799c7b` (`RFOPT experiment disabling three-way dense sort`)

Status: rejected as a general direction.

Change:

- In `sklearn/tree/_partitioner.pyx`, call `simultaneous_sort(...,
  use_three_way_partition=False)` for dense splitter feature sorting.

Hypothesis:

- The benchmark data is mostly continuous, so equal feature values are rare.
  The three-way partition variant appears expensive in native profiles. Disabling
  it might reduce sorting overhead in the exact splitter for this subset.

Focused benchmark results, scikit-learn backend only, `n_estimators=20`,
`n_jobs=1`:

| Case | Baseline median s | Branch median s | Speedup |
|---|---:|---:|---:|
| `reg_1f_deep_full` | 1.662 | 1.339 | 1.24x |
| `reg_12f_shallow_bootstrap` | 2.057 | 1.740 | 1.18x |
| `clf_96f_sqrt_leaf8` | 1.566 | 1.348 | 1.16x |

Validation:

- `pytest sklearn/tree/tests/test_tree.py -q`
- Result: `525 passed, 98 warnings in 4.02s`

Assessment:

- This is a cautionary experiment, not a candidate optimization. It improves the
  continuous-focused synthetic subset, but that subset is too narrow.
- Prior investigation found this change detrimental on real-world datasets,
  where many numerical columns have low cardinality or many tied values.
- The benchmark suite now includes retained low-cardinality numerical cases so
  future sort-related experiments do not overfit to continuous synthetic data.
- Do not pursue this branch as a general scikit-learn change without a new
  algorithm that keeps the duplicate-heavy benefit of three-way partitioning.

## Global sorted sample index

Branches:

- `rfopt/global-sorted-index-intp`, commit `c2f99937cf`
- `rfopt/global-sorted-index-uint32`, commit `5bc4d0e362`
- `rfopt/global-sorted-index-uint16`, commit `bf4c55f14d`

Status: promising for continuous / high-cardinality dense data, but still a
prototype.

Change:

- At splitter initialization, precompute a per-feature global sorted sample
  order for dense data when `max_features == n_features`.
- At large nodes, filter that global order through a node membership marker
  instead of sorting the node-local sample values again.
- Skip the fast path for sampled-feature cases, low-cardinality-looking
  features, small nodes, and missing-valued features.
- The three variants differ only in the dtype used to store the global sample
  order: `intp`, `uint32`, or `uint16`.

Focused benchmark results, scikit-learn backend only, retained suite,
`n_estimators=20`, `n_jobs=1`, 3 repeats:

| Case | Baseline median s | intp s | uint16 s | uint32 s | Best speedup |
|---|---:|---:|---:|---:|---:|
| `reg_1f_deep_full` | 1.681 | 1.336 | 1.314 | 1.300 | 1.29x |
| `reg_12f_full_deep` | 4.557 | 3.628 | 3.620 | 3.619 | 1.26x |
| `reg_12f_shallow_bootstrap` | 2.060 | 1.683 | 1.673 | 1.678 | 1.23x |
| `reg_80f_sqrt_leaf8` | 1.044 | 1.016 | 1.017 | 1.019 | 1.03x |
| `reg_12f_full_f64` | 2.843 | 2.288 | 2.296 | 2.308 | 1.24x |
| `clf_12f_full_deep` | 5.616 | 3.967 | 3.975 | 3.955 | 1.42x |
| `clf_12f_shallow_bootstrap` | 3.108 | 2.328 | 2.296 | 2.298 | 1.35x |
| `clf_96f_sqrt_leaf8` | 1.685 | 1.547 | 1.539 | 1.530 | 1.10x |
| `reg_24f_low_card` | 4.044 | 3.966 | 3.994 | 3.998 | 1.02x |
| `clf_24f_low_card` | 2.450 | 2.485 | 2.492 | 2.475 | 0.99x |

Suite timings:

- `intp`: 119.0s
- `uint16`: 116.8s
- `uint32`: 116.7s

Validation:

- `pytest sklearn/tree/tests/test_tree.py -q`
- `intp`: `525 passed, 98 warnings in 3.72s`
- `uint16`: `525 passed, 98 warnings in 3.76s`
- `uint32`: `525 passed, 98 warnings in 3.83s`

Assessment:

- The main win comes from avoiding repeated node-local sorting on continuous
  features, especially for full-feature forests. That is consistent with the
  sklearnex / oneDAL strategy of building indexed feature representations and
  scanning those representations at training time.
- Narrowing the sample-index dtype does not materially improve CPU time on this
  suite. `uint16` and `uint32` are mostly useful to reduce temporary memory
  traffic and footprint. `uint16` is only valid for `n_samples <= 65535`.
- `uint8` was not implemented as a retained-suite variant because it can only
  encode datasets up to 255 samples if used for global sample ids. That would
  not exercise the RandomForest fit bottleneck measured here.
- This prototype is not yet a candidate patch: it precomputes the sorted order
  per tree instead of per forest, uses NumPy `argsort` during splitter
  initialization, and needs a better memory policy. It is still useful evidence
  that a scikit-learn-portable top-node indexed path can recover a substantial
  part of sklearnex's advantage on high-cardinality dense data.

## `rfopt/hist-best-splitter`

Branch status: implemented prototype.

Change:

- Add opt-in `max_bins=None` to dense `DecisionTreeClassifier`,
  `DecisionTreeRegressor`, `RandomForestClassifier`, and
  `RandomForestRegressor`.
- When `max_bins` is set, supported dense trees use a hybrid
  `HistBestSplitter`.
- Each feature is binned into at most `max_bins` ordered bins. Features with at
  most `max_bins` unique non-missing values keep exact value bins;
  higher-cardinality features use quantile-derived bin thresholds.
- In `RandomForest*`, the bin codes are now precomputed once per forest fit and
  shared by all newly grown trees instead of being recomputed per tree.
- Hot histogram loops now keep the owning arrays as memoryviews but use local C
  pointers for codes, samples, weights, histogram workspaces, and class counts.
- The small-node fallback now follows the oneDAL-inspired
  `node_size <= 0.02 * n_bins` rule instead of an absolute 512-sample cutoff.
- When `max_bins=None`, the default behavior is unchanged. A possible future
  improvement is to automatically detect low-cardinality features and use the
  binned path only for those features, while keeping exact sorting for the rest.
- Supported criteria: `gini`, `entropy` / `log_loss`, `squared_error`.
- Supported: sample weights and missing values.
- Explicitly unsupported: sparse input, random splitter, multi-output,
  monotonic constraints, `absolute_error`, `poisson`, and `friedman_mse`.

Focused retained low-cardinality RF timings, `n_estimators=20`, `n_jobs=1`,
3 repeats:

| Case | sklearn sort median s | sklearn `max_bins=n_samples` median s | sklearn speedup | sklearnex median s |
|---|---:|---:|---:|---:|
| `reg_24f_low_card` | 4.027 | 1.697 | 2.37x | 1.431 |
| `clf_24f_low_card` | 2.520 | 0.759 | 3.32x | 0.552 |

Validation:

- `pytest sklearn/tree/tests/test_tree.py -q`
- Result: `534 passed, 98 warnings in 4.18s`
- `pytest sklearn/tree/tests/test_tree.py -k 'hist_best_splitter' -q`
- Result: `9 passed`
- `pytest sklearn/ensemble/tests/test_forest.py -k 'max_bins' -q`
- Result: `1 passed`
- `pre-commit run cython-lint --files sklearn/tree/_splitter.pyx`
- Result: passed
- `pre-commit run ruff-check --files sklearn/ensemble/_forest.py
  sklearn/tree/_classes.py`
- Result: passed
- `pre-commit run ruff-format --files sklearn/ensemble/_forest.py
  sklearn/tree/_classes.py`
- Result: passed

Assessment:

- This now recovers most of sklearnex's low-cardinality advantage in these two
  retained cases. sklearnex remains faster, especially for classification.
- With the updated `max_bins` semantics, high-cardinality features are no longer
  exact unless `max_bins` is large enough to keep every unique value. This makes
  `max_bins=255` comparable in spirit to sklearnex's binned training mode.
- Further speedups likely require more oneDAL-like specialization: compact
  `uint8` / `uint16` codes, sparse workspace clearing, incremental child
  histogram reuse, and a less generic tree-building loop around the histogram
  splitter.

### `max_bins=255` branch vs sklearnex retained-suite run

The retained suite was rerun with both this branch and sklearnex configured with
`max_bins=255`, `n_estimators=20`, `n_jobs=1`, and 3 repeats. All sklearnex
cases passed the oneDAL preflight and fitted with `_onedal_estimator`.

In this run both implementations use `max_bins=255` as a binning parameter.
Low-cardinality features are still exact in this branch when their unique
non-missing values fit within 255 bins; high-cardinality features are quantile
binned.

Raw outputs:

- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex/retained_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex/retained_summary.csv`
- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex/environment.json`

Suite elapsed time: 57.1s. Every individual fit remained below 10s.

| Case | branch `max_bins=255` median s | sklearnex `max_bins=255` median s | sklearnex speedup |
|---|---:|---:|---:|
| `clf_12f_full_deep` | 1.556 | 0.595 | 2.61x |
| `clf_12f_shallow_bootstrap` | 0.999 | 0.308 | 3.25x |
| `clf_24f_low_card` | 0.920 | 0.588 | 1.57x |
| `clf_96f_sqrt_leaf8` | 0.565 | 0.304 | 1.86x |
| `reg_12f_full_deep` | 2.714 | 1.301 | 2.09x |
| `reg_12f_full_f64` | 1.765 | 0.860 | 2.05x |
| `reg_12f_shallow_bootstrap` | 0.881 | 0.293 | 3.01x |
| `reg_1f_deep_full` | 0.269 | 0.105 | 2.55x |
| `reg_24f_low_card` | 2.067 | 1.673 | 1.24x |
| `reg_80f_sqrt_leaf8` | 0.366 | 0.170 | 2.15x |

Takeaways:

- The updated semantics close a large part of the previous high-cardinality gap
  because the branch no longer falls back to exact sorting for those features.
- sklearnex remains faster across the retained suite, by roughly 1.2x to 3.3x in
  this run. The remaining gap is likely implementation polish: compact code
  dtypes, tighter histogram memory management, less generic criterion and tree
  builder control flow, and oneDAL's native parallel/runtime code.

### Profiling and cleanup toward the 2x target

Follow-up profiling used native `py-spy` speedscope recordings under
`reports/rf_intelex/profiles_max_bins_255/`.

Main finding:

- `HistBestSplitter` inherited `BestSplitter.init`, which still precomputed the
  exact splitter's global sorted index. This was dead work for the histogram
  path. In `clf_12f_shallow_bootstrap`, py-spy showed large NumPy sort/merge
  time from `_global_sorted_index` before the tree fitting work.

Fixes:

- `HistBestSplitter.init` now calls `Splitter.init` directly and creates only
  the dense partitioner needed for final sample partitioning.
- Histogram workspace clearing now uses `memset` and only clears buffers needed
  by the active criterion. Classification no longer clears regression-only
  sum/squared-sum workspaces; regression no longer clears class-count
  workspaces.

After these changes, the retained `max_bins=255` suite was rerun against
sklearnex with the same settings. All sklearnex cases passed oneDAL preflight.

Raw outputs:

- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex_under2x/retained_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex_under2x/retained_summary.csv`
- `reports/rf_intelex/results_max_bins_255_branch_vs_sklearnex_under2x/environment.json`

Suite elapsed time: 45.3s. Every individual fit remained below 10s.

| Case | branch `max_bins=255` median s | sklearnex `max_bins=255` median s | branch / sklearnex |
|---|---:|---:|---:|
| `clf_12f_full_deep` | 0.808 | 0.594 | 1.36x |
| `clf_12f_shallow_bootstrap` | 0.404 | 0.311 | 1.30x |
| `clf_24f_low_card` | 0.820 | 0.608 | 1.35x |
| `clf_96f_sqrt_leaf8` | 0.475 | 0.327 | 1.45x |
| `reg_12f_full_deep` | 1.800 | 1.350 | 1.33x |
| `reg_12f_full_f64` | 1.173 | 0.864 | 1.36x |
| `reg_12f_shallow_bootstrap` | 0.398 | 0.293 | 1.36x |
| `reg_1f_deep_full` | 0.150 | 0.108 | 1.39x |
| `reg_24f_low_card` | 1.801 | 1.484 | 1.21x |
| `reg_80f_sqrt_leaf8` | 0.337 | 0.172 | 1.96x |

Current remaining gap:

- The largest remaining gap is `reg_80f_sqrt_leaf8` at 1.96x. Profiling shows
  most branch time is now real histogram work: `_build_feature_histogram` and
  `_node_split_hist`. sklearnex still has lower constant factors from compact
  `unsigned char` bins, specialized oneDAL split kernels, and tighter data
  layout / runtime integration.

### Final warm-up + 10-repeat retained-suite run

The retained `max_bins=255` branch-vs-sklearnex suite was rerun one last time
with a 30s untimed warm-up followed by 10 timed repeats per case/backend.

Raw outputs:

- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10/retained_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10/retained_summary.csv`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10/warmup_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10/environment.json`

Warm-up elapsed time: 30.2s, with 44 untimed fits. Timed elapsed time: 138.3s.
The variability column is `(max - min) / median` across the 10 timed repeats.

| Case | branch median s | sklearnex median s | branch / sklearnex | branch variability | sklearnex variability |
|---|---:|---:|---:|---:|---:|
| `clf_12f_full_deep` | 0.768 | 0.576 | 1.33x | 0.043 | 0.049 |
| `clf_12f_shallow_bootstrap` | 0.386 | 0.304 | 1.27x | 0.075 | 0.034 |
| `clf_24f_low_card` | 0.747 | 0.511 | 1.46x | 0.110 | 0.163 |
| `clf_96f_sqrt_leaf8` | 0.477 | 0.287 | 1.66x | 0.161 | 0.040 |
| `reg_12f_full_deep` | 1.772 | 1.196 | 1.48x | 0.039 | 0.118 |
| `reg_12f_full_f64` | 1.154 | 0.789 | 1.46x | 0.034 | 0.047 |
| `reg_12f_shallow_bootstrap` | 0.390 | 0.287 | 1.36x | 0.047 | 0.028 |
| `reg_1f_deep_full` | 0.144 | 0.104 | 1.38x | 0.010 | 0.016 |
| `reg_24f_low_card` | 1.765 | 1.319 | 1.34x | 0.215 | 0.127 |
| `reg_80f_sqrt_leaf8` | 0.310 | 0.165 | 1.88x | 0.149 | 0.128 |

The 10-repeat run keeps every retained case below the 2x target. The highest
observed ratio is `reg_80f_sqrt_leaf8` at 1.88x. Some low-cardinality rows have
noticeable repeat-to-repeat variability, especially `reg_24f_low_card`, so small
changes in those rows should be interpreted with that noise level in mind.

### Float64-input warm-up + 10-repeat retained-suite run

The same final protocol was rerun after forcing every generated `X` to
`float64` before fitting both implementations. This checks whether sklearnex's
dtype handling gives it an unfair advantage when the benchmark datasets are
mixed `float32` / `float64`.

Raw outputs:

- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10_xfloat64/retained_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10_xfloat64/retained_summary.csv`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10_xfloat64/warmup_raw.jsonl`
- `reports/rf_intelex/results_max_bins_255_warmup30_repeats10_xfloat64/environment.json`

Warm-up elapsed time: 30.1s, with 39 untimed fits. Timed elapsed time: 139.3s.
The variability column is `(max - min) / median` across the 10 timed repeats.

| Case | branch median s | sklearnex median s | branch / sklearnex | branch variability | sklearnex variability |
|---|---:|---:|---:|---:|---:|
| `clf_12f_full_deep` | 0.771 | 0.572 | 1.35x | 0.028 | 0.048 |
| `clf_12f_shallow_bootstrap` | 0.389 | 0.306 | 1.27x | 0.125 | 0.025 |
| `clf_24f_low_card` | 0.742 | 0.510 | 1.46x | 0.148 | 0.172 |
| `clf_96f_sqrt_leaf8` | 0.418 | 0.289 | 1.45x | 0.047 | 0.071 |
| `reg_12f_full_deep` | 1.792 | 1.241 | 1.44x | 0.046 | 0.218 |
| `reg_12f_full_f64` | 1.170 | 0.795 | 1.47x | 0.027 | 0.054 |
| `reg_12f_shallow_bootstrap` | 0.448 | 0.308 | 1.45x | 0.225 | 0.475 |
| `reg_1f_deep_full` | 0.146 | 0.096 | 1.52x | 0.170 | 0.319 |
| `reg_24f_low_card` | 1.768 | 1.272 | 1.39x | 0.097 | 0.173 |
| `reg_80f_sqrt_leaf8` | 0.311 | 0.170 | 1.84x | 0.155 | 0.140 |

Forcing `X` to `float64` does not materially change the conclusion: every
retained case remains below the 2x target. The largest ratio is still the
wide/sqrt regression case, now at 1.84x.
