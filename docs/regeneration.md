# Regenerating rules to bound feature-selection RAM

The command-line pipeline now computes feature-selection rules in blocks,
retains compact per-rule statistics, and regenerates binary vectors when needed.
It never constructs the complete pair-by-sample matrix. There are no new
dependencies or configuration settings.

## Algorithm and compatibility

`RegeneratingRuleEvaluator.py` performs the following steps:

1. Enumerate protein pairs in the original combinations order using two `uint32`
   indices per pair. Keep a contiguous protein-by-sample view/copy of the
   original numeric data, with the same missing-value handling as before.
2. Generate at most approximately 1,000,000 binary elements per block (at least
   one rule). Compute the original true scores and proportion buckets, and
   calculate a 128-bit fingerprint from transient packed binary rows. Discard
   the block instead of storing its vectors.
3. Sort fingerprints and regenerate matching groups to verify exact binary
   equality. Fingerprints alone never determine duplication. Retain the first
   original pair for each identical vector, including when different vectors
   have the same fingerprint. Verification buffers are also bounded by the
   block size. Deliberate collisions can increase CPU work, but do not change
   selected rules.
4. Regenerate surviving rules for null scoring. Preserve the initial label
   permutation, bucket first-occurrence order and every additional permutation
   used to expand small null distributions. Preserve the original floating-point
   score, bucket-rounding and p-value calculations.
   Binary dot products are evaluated as exact wide-integer counts before the
   original divisions, avoiding promotion of whole binary blocks to integers.
5. Rank numeric arrays by p-value, descending score and original summary order.
   Regenerate candidates for the existing normalized mutual-information
   calculation and cache only accepted vectors. Disjoint selection and the
   original first-rule behavior for `k_rules=1` are preserved.
6. Construct and export the selected-feature DataFrame with the original schema.

Fingerprint and score storage scales with the number of pairs, not with pairs
times samples. The numeric input still scales with samples times proteins.
All pairs are evaluated; no protein downcasting, sample reduction, approximation
or statistical pruning was added. Input quantification tables are not mutated.

`nifty.py` calls `FeatureSelector.find_features(configs, return_details=False)`.
This returns the selected-feature DataFrame. For existing Python callers,
`find_features(configs)` retains its historical detailed return contract and
dense implementation; use `return_details=False` to obtain the RAM savings.
The new evaluator's `run(..., return_details=True)` can explicitly construct
the complete legacy-shaped score/summary outputs for diagnostics, at additional
memory cost. Its normal path constructs only the selected rows.

## Exactness tests

All 258 unit tests pass. `UnitTests/RegenerationTests.py` compares regenerated
results against the existing dense implementation, including exact complete
score arrays, summary DataFrames, selected rows and output bytes. Cases cover
both filtering switches, `k_rules=1`, ties, missing values, infinities, duplicate
vectors, deliberately identical fingerprints, different block sizes, buckets
above and below the null-expansion threshold, and seeded/unseeded RNG state.
Counts above 255 are exercised without narrowing the accumulator dtype.

The included-data benchmark compares against commit
`1156973ead366f25f6a71db51dda48033bbec696`, the previous blocked implementation.
All 14 paired comparisons across eight cases produce byte-identical feature
files, model information, saved model pickles and predictions, including RF
and SVM probabilities. Repetitions also produce the same output hashes.
All five original regression tests also pass against the repository's saved
expected outputs.

## Included-data benchmark

Measured on Windows with Python 3.12.14, NumPy 2.2.6, pandas 2.2.3,
scikit-learn 1.6.1, SciPy 1.15.3, statsmodels 0.14.6 and cloudpickle 3.1.2.
Both versions use seed 42 and `PYTHONHASHSEED=0`. The supplied fixture sets each
contain 100 samples and 2,844 proteins; individual-input feature selection
retains 962 proteins. Reference mode concatenates the three disjoint reference
fixture sets and uses the existing 15%/65%/20% split.

Each measurement uses a fresh process and Windows `PeakWorkingSetSize`, which
includes native allocations and imported libraries. Runs are sequential with
alternating version order. Repeated cases report medians; other cases are
single measurements. Runtime covers the pipeline call, including file I/O but
excluding import/startup, and varies with machine load and caching.

| Case | Repetitions per version | Previous peak MiB | Regeneration peak MiB | RAM reduction | Previous seconds | Regeneration seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feature selection | 3 | 486.3 | 190.7 | 60.8% | 5.85 | 2.01 |
| Feature selection + training | 1 | 490.3 | 196.7 | 59.9% | 5.95 | 3.10 |
| Full pipeline, RF classes | 3 | 494.0 | 200.6 | 59.4% | 5.92 | 3.41 |
| Training only | 1 | 167.4 | 167.3 | 0.1% | 0.96 | 0.94 |
| Application only | 1 | 162.3 | 163.1 | -0.5% | 0.28 | 0.28 |
| Full pipeline, RF probabilities | 1 | 494.1 | 200.4 | 59.4% | 6.30 | 3.42 |
| Reference-split full pipeline | 3 | 440.4 | 197.4 | 55.2% | 5.70 | 3.67 |
| Full pipeline, SVM probabilities | 1 | 493.8 | 199.3 | 59.6% | 6.34 | 3.89 |

Full-pipeline mean CV accuracy remains 0.99; validation accuracy remains 0.98.
Training-only/application-only memory is effectively unchanged. Although
regeneration performs additional comparisons, avoiding large dictionaries,
DataFrames and copies improves runtime on these fixtures. Larger inputs may
have a different runtime tradeoff.

## Full-size capacity measurement

The completed synthetic capacity test used **10,000 samples and 5,000 features**,
with no feature filtering or pair subsampling. All **12,497,500 pairs** were
distinct, all survived duplicate checking, and all were regenerated for null
scoring. The run selected and exported 15 rules.

| Measurement | Result |
| --- | ---: |
| Peak resident RAM (Windows peak working set) | **1.612 GiB** |
| Peak committed memory | **2.837 GiB** |
| Feature-selection runtime | **1,575.17 seconds / 26.25 minutes** |
| Samples / retained features | 10,000 / 5,000 |
| Evaluated / distinct pairs | 12,497,500 / 12,497,500 |
| Selected rules | 15 |

This is a single measured run of the final implementation. It includes the
in-memory float64 input, compact statistics, temporary blocks and imported
libraries. It calls the feature-selection evaluator directly, so it excludes
CSV parsing, reference splitting and subsequent model training/application.
Those stages and other datasets may require additional RAM. Committed memory
is distinct from currently resident physical memory.

The numeric input comes from `default_rng(20260911).standard_normal`, with
balanced alternating class labels and pipeline seed 42. It tests capacity and
completion, not biological prediction quality. Exact output equivalence is
established separately by the dense comparisons and supplied regression
fixtures above; the old full-size dense algorithm was not run.

The raw result and matching evaluator source hashes are stored in
`Test_Output/regeneration_scale/measurement.json`; the run log and selected
features are alongside it. The original approximately 500 GB dense peak was
an allocation estimate, not a measured baseline for this capacity test.

## Reproduction

From the repository root, with the dependencies above available:

```powershell
python RegressionTests/BenchmarkMemory.py --baseline-ref 1156973 --repeats 3 --output-dir Test_Output/regeneration_benchmark
python RegressionTests/BenchmarkRegenerationScale.py --samples 10000 --features 5000
python -c "import sys, unittest; sys.argv=['unittest']; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('UnitTests', pattern='*Tests.py')); sys.exit(not result.wasSuccessful())"
python RegressionTests/RunAllRegressionTests.py
```

The added `--output-dir` option keeps previous benchmark artifacts intact.
Raw measurements, configurations, baseline source snapshots, output hashes
and outputs are in ignored `Test_Output/regeneration_benchmark`. The separate
scale benchmark writes to `Test_Output/regeneration_scale`. It uses seeded
synthetic data and balanced labels, not a biological accuracy dataset, and
does not attempt to allocate the old dense baseline at that size.
