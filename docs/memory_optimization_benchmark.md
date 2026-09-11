# RAM optimization: implementation and before/after validation

Measured on Windows on 2026-09-11. Baseline commit:
`da3db09ae16013e269cc8e3d524e0534831a02da`.

The optimized source produces byte-identical output files in all 14 paired
comparisons below. Peak resident RAM is approximately halved for the included
individual-input feature-selection and full-pipeline cases.

## Changes

1. `DataTransformer.vectorize_all_pairs` preallocates the same C-contiguous
   `int8` matrix and computes comparisons in blocks targeting 1,000,000
   pair-by-sample elements. It no longer allocates two full floating-point
   pair matrices or full Python index lists. Protein-pair order, strict `>`,
   missing values represented as `-inf`, and ties are unchanged.
2. `EvaluateRules.evaluate_pairs` scores blocks of the same size, bounding
   temporary allocations and dtype promotion during dot products. It retains
   the original label dtype, integer accumulation, division and subtraction.
   Permutations remain outside the block loop: each permutation is shared by
   all blocks, and the random-number call sequence is unchanged.
3. `nifty.main` releases reference tables after splitting, and releases the
   full pair list, score array, summary and feature-selection input tables
   after feature selection. The selected feature table remains available.
   Existing component return values are unchanged.

The full binary rule matrix, duplicate filtering, pair bookkeeping, models,
hyperparameters and statistical filtering remain as before. Memory still grows
quadratically with the number of retained proteins; these changes do not make
arbitrarily large searches fit in RAM.

## Inputs and method

The unmodified included `RegressionTests/test_data` TSVs contain 100 samples
per set and 2,844 protein columns. Feature selection retains 962 proteins,
generates 462,241 pairs and retains 461,962 distinct binary rules.

The additional reference-mode case concatenates the included FS, Train_Test
and Val sets, without altering values or sample IDs. Their 300 distinct samples
are split by the existing pipeline into 45 feature-selection, 195 training and
60 validation samples. This case retains 924 proteins and 410,513 distinct
binary rules from 426,426 pairs. Experimental data remains the included
Experimental set.

Both versions use Python 3.12.14, NumPy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1,
SciPy 1.15.3, statsmodels 0.14.6 and cloudpickle 3.1.2. Scikit-learn matches the
included serialized model fixture. Both use seed 42 and `PYTHONHASHSEED=0`.

Each pipeline runs in a fresh process. RAM is Windows `PeakWorkingSetSize`
from `GetProcessMemoryInfo`, including imported libraries and native array
allocations. This is an operating-system high-water mark, not periodically
sampled RAM or Python-only allocation tracking. Runs are sequential. The three
main cases have three repetitions per version with alternating version order;
the other cases have one per version. Tables report medians for repeated cases.
One MiB is 1,048,576 bytes.

Runtime measures the `nifty.main()` call, including input/output but excluding
module import and process startup. Short timings vary with machine load and
file caching; the RAM and equivalence results are the primary findings.

## Peak RAM and runtime

| Case | Runs per version | Before MiB | After MiB | RAM reduction | Before seconds | After seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feature selection | 3 | 986.6 | 485.8 | 50.8% | 4.31 | 4.02 |
| Feature selection + training | 1 | 993.5 | 491.5 | 50.5% | 4.16 | 4.10 |
| Full pipeline, RF classes | 3 | 994.3 | 494.3 | 50.3% | 5.99 | 4.79 |
| Training only | 1 | 166.8 | 167.3 | -0.3% | 0.93 | 0.92 |
| Application only | 1 | 163.4 | 163.5 | -0.1% | 0.30 | 0.31 |
| Full pipeline, RF probabilities | 1 | 994.2 | 493.8 | 50.3% | 7.09 | 5.92 |
| Reference-split full pipeline | 3 | 535.4 | 440.2 | 17.8% | 6.82 | 6.05 |
| Full pipeline, SVM probabilities | 1 | 995.4 | 494.1 | 50.4% | 6.20 | 6.45 |

Training-only and application-only memory is essentially unchanged. The
reference case has smaller pair-by-sample matrices, so removing full-size
temporaries saves less. These measurements cover the three changes together;
they do not isolate each change's individual contribution.

Resident RAM immediately before entering model generation also falls:

| Case | Before MiB | After MiB |
| --- | ---: | ---: |
| Full pipeline | 220.0 | 182.2 |
| Reference-split full pipeline | 221.6 | 177.4 |

The benchmark verifies that completed-stage keys are absent at that boundary
and `feature_table` is still present. Removing references does not guarantee
that Python or native allocators immediately return every freed byte to Windows.

Windows peak committed memory is separately recorded in the raw JSON. For
feature selection it falls from 2,170.6 to 1,668.8 MiB, for the full pipeline
from 2,178.8 to 1,676.8 MiB, and for the reference case from 1,719.1 to 1,623.2
MiB. Committed memory includes allocations that are not currently resident;
it is distinct from the peak physical-RAM measure above.

## Results preserved

All 14 paired comparisons pass exact byte comparisons for every applicable
output: `selected_features.tsv`, `model_information.txt`,
`predicted_classes.tsv`, and `trained_model_and_model_metadata.pkl`.
This includes row/column ordering, scores, p-values, class predictions,
RF/SVM probabilities and the complete serialized models, without numerical
tolerance. Across repetitions, outputs are also identical within each case.

For the included full RF pipeline:

| Result | Before | After |
| --- | ---: | ---: |
| Selected features | 15 | 15 |
| Top pair | P02545 > P31146 | P02545 > P31146 |
| Top pair score / p-value | 1.0 / 0.0 | 1.0 / 0.0 |
| Mean CV accuracy | 0.99 | 0.99 |
| Validation accuracy | 0.98 | 0.98 |
| Validation precision | 0.9807692307692308 | 0.9807692307692308 |
| Validation recall | 0.98 | 0.98 |

Validation completed:

- Baseline: all 250 existing unit tests pass in the same environment.
- Optimized: all 252 unit tests pass, including new block-boundary, empty-shape,
  missing-value, infinity, tie, duplicate-order, wide-integer accumulation and
  permutation-sequence checks.
- All five original regression tests pass against the repository's expected
  output fixtures.
- All 14 before/after output comparisons and completed-stage lifetime checks
  pass. These establish equivalence for the tested inputs, not every possible
  future dataset or dependency version.

## Reproduction and artifacts

Use Python >=3.11 with the versions above. From the repository root:

```powershell
python RegressionTests/BenchmarkMemory.py --baseline-ref da3db09 --repeats 3
python -c "import sys, unittest; sys.argv=['unittest']; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('UnitTests', pattern='*Tests.py')); sys.exit(not result.wasSuccessful())"
python RegressionTests/RunAllRegressionTests.py
```

The unit-test command deliberately clears runner arguments because an existing
argument-parser test inspects `sys.argv`. On this machine dependencies were
installed into ignored `Test_Output/python_deps`; its path was prepended to
`PYTHONPATH`, with the bundled Python executable used instead of PATH's older
Python 3.9. Temporary test directories were directed to `Test_Output/tmp`.

`RegressionTests/BenchmarkMemory.py` preserves the Git baseline source and
writes all raw evidence under ignored `Test_Output/ram_benchmark`:

- `measurements.json`: all 28 process measurements, timings, versions and
  stage-boundary memory/key snapshots.
- `manifest.json`: baseline commit, input hashes and optimized source hashes.
- `<case>/<repeat>/{before,after}/`: configurations, logs, memory counters and
  complete pipeline output files.
- `<case>/<repeat>/exact_comparison.json`: matching output SHA-256 hashes.
- `unit_before.log`, `unit_after.log`, `regression_after.log`: test logs from
  this validation session.

The benchmark reruns into these same artifact paths. Copy them elsewhere first
if historical measurements need to be retained across future benchmark runs.
