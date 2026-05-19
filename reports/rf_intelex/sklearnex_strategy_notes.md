# What sklearnex / oneDAL does faster

Date: 2026-05-18

Reference sources inspected:

- `external-src/scikit-learn-intelex` at `7e3972a`
- `external-src/oneDAL` at `8f6aa2b`

## Short answer

The sklearnex speedup is not just a more polished implementation of the same
node-local sort loop. It uses a different representation strategy in oneDAL:
features are indexed/binned globally, compact integer index types are selected
when possible, and many split searches operate over per-feature index/bin
histograms. Scikit-learn's exact dense splitter instead repeatedly copies and
comparison-sorts node-local feature values for each candidate feature.

With `max_bins=n_samples`, the sklearnex benchmark is still not using a small
approximate bin count, but the oneDAL code path still goes through its
`hist`/indexed-feature machinery. When a feature has few distinct values, this
is a major advantage.

## Evidence from source

sklearnex passes scikit-learn parameters into oneDAL, including:

- `features_per_node`
- `tree_count`
- `bootstrap`
- `observations_per_tree_fraction`
- `max_bins`
- `min_bin_size`

Relevant files:

- `external-src/scikit-learn-intelex/sklearnex/ensemble/_forest.py`
- `external-src/scikit-learn-intelex/onedal/ensemble/forest.cpp`

oneDAL constructs an `IndexedFeatures` representation:

- `external-src/oneDAL/cpp/daal/src/algorithms/dtrees/dtrees_feature_type_helper.h`
- `external-src/oneDAL/cpp/daal/src/algorithms/dtrees/dtrees_feature_type_helper.i`

The header describes the structure as creating and storing an index of every
feature, mapping feature values to indices in the sorted unique-value array.
The implementation initializes those indices per column in parallel.

In decision forest training, oneDAL chooses compact index types based on the
maximum number of indices:

- `uint8_t` when `maxNumIndices() <= 256`
- `uint16_t` when `maxNumIndices() <= 65536`
- otherwise the default index type

Relevant file:

- `external-src/oneDAL/cpp/daal/src/algorithms/dtrees/forest/regression/df_regression_train_dense_default_impl.i`

The split loop decides whether to use indexed features:

```text
bUseIndexedFeatures =
    !_memorySavingMode
    && n > qMax * indexedFeatures.numIndices(iFeature)
```

If enabled, oneDAL calls `findSplitForFeatureSorted`, which builds histograms
over indexed values and scans `nDiffFeatMax`, the number of distinct indices,
not the raw number of samples.

Scikit-learn's dense exact path is different:

- `sklearn/tree/_splitter.pyx` calls
  `partitioner.sort_samples_and_feature_values(current_split.feature)` for each
  candidate feature at each node.
- `sklearn/tree/_partitioner.pyx` copies the node-local feature values and calls
  `simultaneous_sort(..., use_three_way_partition=True)`.
- The criterion then scans the sorted sample order.

This is exact and general, but duplicate-heavy and low-cardinality features
still pay for repeated comparison sorting.

## Evidence from profiles

Profiles were collected with `py-spy record --native --format speedscope`.
The percentages below are from exclusive native samples in the generated
speedscope files. They are approximate, but the pattern is stable.

| Case | Backend | Timing in suite | Main exclusive costs |
|---|---:|---:|---|
| `reg_12f_shallow_bootstrap` | sklearn | 2.060s | sort/index 36.9%, criterion/hist 7.5%, split-search 2.3% |
| `reg_12f_shallow_bootstrap` | sklearnex | 1.323s | split-search 14.7%, sort/index 11.9%, alloc/libc 12.8% |
| `clf_96f_sqrt_leaf8` | sklearn | 1.685s | sort/index 25.2%, criterion/hist 12.2%, split-search 2.2% |
| `clf_96f_sqrt_leaf8` | sklearnex | 0.930s | split-search 15.3%, sort/index 7.0%, alloc/libc 16.2% |
| `clf_24f_low_card` | sklearn | 2.450s | sort/index 35.4%, criterion/hist 10.3%, split-search 0.9% |
| `clf_24f_low_card` | sklearnex | 0.488s | split-search 21.3%, sort/index 2.2%, alloc/libc 13.0% |

Representative top frames:

- scikit-learn:
  - `_sorting_introsort_3way`
  - `DensePartitioner_sort_samples_and_feature_values`
  - `ClassificationCriterion_update`
  - `RegressionCriterion_update`
  - `Gini_children_impurity`
- sklearnex / oneDAL:
  - `decision_forest::...findBestSplitSerial`
  - `findSplitForFeatureSorted`
  - `findSplitFewClassesDispatch`
  - `ColIndexTask::getSorted`
  - `qSort`

The low-cardinality classification case is the strongest proof in this suite:
scikit-learn spends about 35% of native samples in sorting/indexing and takes
2.45s, while sklearnex spends about 2% in sort/index work and takes 0.49s.

## Fundamental differences

### 1. Global feature indexing

oneDAL builds a global mapping from feature values to ordered integer indices or
bins. This can amortize sorting/indexing and exposes the number of distinct
values per feature.

scikit-learn only keeps raw `X` plus node-local sample arrays. It repeatedly
copies and sorts values for the current node and feature.

### 2. Histogram over unique values

For indexed features, oneDAL can build per-node histograms over feature indices:

- regression: sums / weights per indexed value
- classification: class counts per indexed value

It then scans the ordered unique values to find the best split.

This is still exact for low-cardinality numerical features when the indices
represent true unique values. With `max_bins=n_samples`, it also avoids small-bin
approximation in the benchmarked cases.

### 3. Compact index types

oneDAL specializes on `uint8_t`, `uint16_t`, or larger index types depending on
the maximum number of indexed values. This reduces memory bandwidth for
low-cardinality features.

### 4. Native whole-forest training kernel

oneDAL trains the forest inside a native decision forest kernel with its own
threading and per-tree buffers. scikit-learn dispatches trees through joblib and
then uses Cython for each tree. In these `n_jobs=1` measurements, this is less
important than the split representation, but it still reduces Python boundary
surface.

## Ideas worth porting to scikit-learn

### 1. Low-cardinality counting-sort path in `DensePartitioner`

Most practical first step.

Instead of replacing three-way sort, add a special path that detects globally
low-cardinality dense numerical features and sorts node samples by precomputed
integer codes using counting sort. This preserves the existing sorted-sample
criterion loop, so it is less invasive than a full histogram splitter.

Sketch:

- During dense splitter initialization, optionally precompute per-feature ordered
  codes for features with `n_unique <= threshold`, e.g. 256 or 1024.
- Store compact codes (`uint8` / `uint16`) for those features.
- In `sort_samples_and_feature_values`, if the feature has codes, counting-sort
  `samples[start:end]` by code and fill `feature_values` from the bin borders or
  representative values.
- Then reuse the existing `next_p`, criterion updates, missing handling, and
  split record machinery.

Why this is promising:

- It targets the exact area where sklearnex is strongest on real-world-like
  repeated-value data.
- It avoids the rejected `disable_three_way_sort` mistake: duplicate-heavy
  features should get faster, not slower.
- It is localized mostly to dense partitioning/splitter state.

Risks:

- Extra memory can be large if codes are stored for many features.
- Need careful handling for sample weights, missing values, float32 equality,
  monotonic constraints, and multi-output targets.
- The precompute cost must be amortized. This is likely more attractive in
  forests than in a single small tree.

### 2. Constant / single-value feature detection before sorting

Lower effort but smaller upside.

oneDAL can use indexed features to detect when a node has no differing values for
a feature before doing expensive split work. scikit-learn currently sorts then
checks whether the sorted range is constant.

Possible scikit-learn experiment:

- For dense features, add a cheap pre-sort min/max or code-based
  `has_different_values` check for known low-cardinality / indexed features.
- Skip sorting entirely when all node samples share one value.

This helps datasets with many constant-in-node features, but it adds a scan for
cases that would not be constant. It is best paired with the indexed/coded path.

### 3. Exact histogram split path for low-cardinality features

Larger effort.

Implement a true low-cardinality exact histogram splitter that computes class
counts or regression sums per unique value and scans those bins directly. This is
closer to oneDAL and can avoid sorting entirely, but it duplicates criterion
logic and has a larger compatibility surface.

This may be worthwhile long term, but the counting-sort path is the safer first
experiment because it reuses existing criterion code.

## Ideas not worth pursuing as-is

### Disable three-way sort globally

This was measured as faster on continuous synthetic cases, but prior experience
found it detrimental on real-world data with many ties. The current report now
keeps it as a rejected experiment. It should not be used as a general
optimization.

## Recommended next experiment

Create `rfopt/low_card_counting_sort`:

1. Add a dense-only, no-missing first prototype.
2. Detect low-cardinality float32 features at splitter initialization.
3. Precompute ordered integer codes for features with `n_unique <= 256`.
4. Use counting sort by code in `DensePartitioner.sort_samples_and_feature_values`.
5. Reuse the existing criterion scan over sorted samples.
6. Benchmark at least:
   - `clf_24f_low_card`
   - `reg_24f_low_card`
   - `clf_96f_sqrt_leaf8`
   - one continuous case to confirm no regression from detection overhead

Acceptance bar:

- Clear speedup on low-cardinality cases.
- Neutral within noise on continuous cases.
- Tree tests pass, plus targeted duplicate-heavy RandomForest tests.
