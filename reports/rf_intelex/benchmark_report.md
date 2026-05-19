# RandomForest fit benchmark: scikit-learn vs sklearnex

Date: 2026-05-18

## Methodology

The benchmark compares fit time for local scikit-learn RandomForest estimators
against `sklearnex.ensemble` RandomForest estimators on CPU. All benchmarked
forests use `n_estimators=20`. For sklearnex, every retained case sets
`max_bins=n_samples` to avoid intentionally benchmarking low-bin approximate
histogram splitting against scikit-learn's exact splitter.

The retained suite uses dense, finite, single-output synthetic datasets only.
Before each sklearnex timing the harness calls
`_onedal_cpu_supported("fit", X, y, None)`, and after fit it requires
`_onedal_estimator` to be present. Unsupported cases are excluded from timing
summaries.

Artifacts:

- Harness: `benchmarks/rf_intelex/bench_rf_fit.py`
- Raw timings: `reports/rf_intelex/results/retained_raw.jsonl`
- Summary CSV: `reports/rf_intelex/results/retained_summary.csv`
- Environment: `reports/rf_intelex/results/environment.json`
- Native profiles: `reports/rf_intelex/profiles/*_native.speedscope.json`

Environment:

- Python: 3.14.3
- scikit-learn: 1.9.dev0
- sklearnex: 2199.9.9
- oneDAL: 2021.6
- CPU count: 16 logical, 12 physical reported by `psutil`
- Primary timing setting: `n_jobs=1` and `threadpool_limits(1)`

## Retained suite

The retained suite completed in 134.5s, under the 3 minute target. The slowest
individual retained timing was 5.62s, under the 10s per-benchmark target.

| Case | Task | Shape | HP focus | sklearn s | sklearnex s | speedup |
|---|---:|---:|---|---:|---:|---:|
| `reg_1f_deep_full` | regression | 60000 x 1 float32 | bootstrap=False, max_depth=None, max_features=1.0 | 1.681 | 0.669 | 2.51x |
| `reg_12f_full_deep` | regression | 24000 x 12 float32 | bootstrap=False, max_depth=None, max_features=1.0 | 4.557 | 2.751 | 1.66x |
| `reg_12f_shallow_bootstrap` | regression | 24000 x 12 float32 | bootstrap=True, max_depth=10, max_features=1.0 | 2.060 | 1.323 | 1.56x |
| `reg_80f_sqrt_leaf8` | regression | 9000 x 80 float32 | bootstrap=False, max_depth=None, max_features=sqrt, min_samples_leaf=8 | 1.044 | 0.481 | 2.17x |
| `reg_12f_full_f64` | regression | 16000 x 12 float64 | bootstrap=False, max_depth=None, max_features=1.0 | 2.843 | 1.766 | 1.61x |
| `clf_12f_full_deep` | classification | 28000 x 12 float32 | bootstrap=False, max_depth=None, max_features=1.0 | 5.616 | 2.824 | 1.99x |
| `clf_12f_shallow_bootstrap` | classification | 28000 x 12 float32 | bootstrap=True, max_depth=10, max_features=1.0 | 3.108 | 1.911 | 1.63x |
| `clf_96f_sqrt_leaf8` | classification | 10000 x 96 float32 | bootstrap=False, max_depth=None, max_features=sqrt, min_samples_leaf=8 | 1.685 | 0.930 | 1.81x |
| `reg_24f_low_card` | regression | 30000 x 24 float32 | bootstrap=False, max_depth=None, max_features=1.0 | 4.044 | 1.270 | 3.18x |
| `clf_24f_low_card` | classification | 30000 x 24 float32 | bootstrap=False, max_depth=None, max_features=1.0 | 2.450 | 0.488 | 5.02x |

Pilot cases removed from the retained suite:

- `clf_96f_full_depth12`: scikit-learn timing was 13.4s, above the 10s target.
- `reg_80f_full_depth12`: retained as an exploratory pilot only; it was useful but
  less distinct than the final wide-feature `sqrt` case.

The low-cardinality cases are dense numerical arrays with many repeated values,
intended to represent the common real-world situation where numerical columns
are effectively discrete or quantized.

Fallback validation:

- All retained sklearnex cases passed oneDAL CPU preflight.
- Explicit negative checks rejected sparse input, `warm_start=True`, and
  multi-output targets with the expected sklearnex support messages.

## Profiling findings

Profiles were collected with `py-spy record --native --format speedscope`.
Some `py-spy` invocations returned `No child process` after writing the file,
but the profile artifacts were produced and reported sample counts with zero or
near-zero sampling errors.

Main scikit-learn finding: exact tree fitting is dominated by repeated sorting
of feature values in the dense splitter.

Representative native samples:

- `reg_12f_shallow_bootstrap` scikit-learn:
  `_sorting_introsort_3way` was the top exclusive frame at about 36% of sampled
  time. The tree build path goes through `_parallel_build_trees`,
  `DepthFirstTreeBuilder_build`, `BestSplitter_node_split`, and
  `DensePartitioner_sort_samples_and_feature_values`.
- `clf_96f_sqrt_leaf8` scikit-learn:
  `_sorting_introsort_3way` was again the top exclusive frame at about 23% of
  sampled time, followed by classification criterion update and Gini impurity
  computations.
- `reg_1f_deep_full` scikit-learn:
  sorting remained the largest identifiable cost, with `_sorting_introsort_3way`
  and `_sorting_heapsort` prominent, plus criterion initialization and updates.

Main sklearnex finding: the oneDAL path spends most sampled native time inside
decision forest training kernels, with split search and internal sorting visible
as `findSplit...`, `qSort`, and `OrderedRespHelperBest` /
`UnorderedRespHelperBest` frames. Even with `max_bins=n_samples`, the native
symbols still include `decision_forest::method::hist`; the important fairness
control here is that sklearnex was not using a small bin count.

## Improvement hypotheses

1. Dense feature sorting is still the first scikit-learn target, but any change
   must preserve duplicate-heavy performance. Real-world datasets often contain
   low-cardinality numerical features, so continuous-only synthetic speedups are
   not enough evidence.
2. The low-cardinality cases show sklearnex has a particularly large advantage
   on repeated-value data: 3.18x for regression and 5.02x for classification in
   this suite.
3. Bootstrap and shallow-depth overheads still matter, but they are secondary in
   these profiles: sklearnex speedups are smaller in shallow bootstrap cases,
   while sorting remains visible.

See `reports/rf_intelex/optimization_experiments.md` for the first branch-level
experiment.
