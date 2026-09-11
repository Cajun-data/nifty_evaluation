# NIFty
Never Impute Features (thank you).

The pre-print manuscript associated with this tool can be found here: [Classification with Missing Data - A *NIFty* Pipeline for Single-Cell Proteomics](https://doi.org/10.64898/2026.03.06.710179)

NIFty is a python program for data-driven cell annotation (classification). NIFty can be used for top-scoring pairs (TSP)-based rule generation and feature selection, classification model generation, and model application on unlabaled data. NIFty is unique in that it does not require missing-value imputation, avoids common circular analysis pitfalls by default, and overcomes batch effects. 
The primary application is for classifying large molecular data, like proteomics. 

NIFty uses a minimum of two user-provided tables as input for feature selection and model generation: 
1. a table with quantification data, proteins (or some other molecular data type) as the columns and samples as the rows; and
2. a table that has the label (class) for each sample. 

For model application, a minimum of one user-provided table is used as input:
1. a table with quantification data, proteins (or some other molecular data type) as the columns and samples as the rows.

Quantitation measurements can come from any search tool and any number of measurements (minimum of 2) can be provided.

The output from our program depends on which functionalty the user would like to run. 
In the 'find_features' mode, the output is a list of the *k* best TSP-based features/rules that can be used to train a machine learning classifier for sample annotation. 
In the 'train_model' mode, the output is a machine learning model trained on the selected features. 
In the 'apply_model' mode, the output is a list of predicted sample labels or label probabilities from applying the trained model on experimental, unlabeled data.

The important thing is that we never impute; we can deal with null values.

After downloading this repository, run NIFty on your own data with the following command on the commandline (assuming config.toml exists in the same directory):
> python nifty.py

After downloading this repository, run NIFty on your own data with the following command on the commandline (with a custom config filepath):
> python nifty.py -c <config/file/path>

## Requirements

NIFty requires Python (>= 3.11) and the following Python packages to be installed:

* cloudpickle
* numpy
* pandas
* scikit-learn
* statsmodels

## Codebase Structure

The codebase functions as follows:
![NIFty Flowchart](images/Pipeline_flow.png)

## RAM Improvements and Evaluation

This fork includes two rounds of computational changes to reduce feature-selection
RAM while preserving the existing statistical calculations and output formats:

1. **Blockwise computation and data cleanup.** Rule generation and scoring operate
   in bounded blocks instead of allocating full floating-point pair matrices.
   Reference tables and feature-selection intermediates are released when their
   stages finish.
2. **Rule regeneration.** The command-line pipeline no longer stores the full
   pair-by-sample binary matrix. It retains compact protein indices, fingerprints
   and score arrays, then regenerates vectors for duplicate verification, null
   scoring and mutual-information filtering. Hash matches are checked against
   the actual vectors, so a hash collision cannot remove a distinct rule. Binary
   scores use exact integer counts, and only selected vectors are cached.

These changes preserve protein-pair order, missing-value comparisons, ranking
ties, label-permutation sequences, model settings and selected-feature output.
All candidate pairs are evaluated; no samples or retained proteins are subsampled.
The command-line pipeline uses regeneration automatically, with no new dependencies
or configuration settings. Python callers using `FeatureSelector.find_features`
should pass `return_details=False` for this path; the default retains the historical
detailed return contract and dense implementation.

### Measured RAM reductions on the included test data

The two evaluation rounds measured Windows peak resident RAM in fresh processes
under the same dependency versions and seeds. Each table entry is the median of
three runs per version. The fixture sets contain 100 samples and 2,844 proteins;
individual-input feature selection retains 962 proteins. The reference-split case
combines three disjoint sets into 300 samples, of which 45 enter feature selection.

| Evaluation round | Workflow | Before peak RAM | After peak RAM | Reduction |
| --- | --- | ---: | ---: | ---: |
| Original → blockwise computation | Feature selection | 986.6 MiB | 485.8 MiB | 50.8% |
| Original → blockwise computation | Full pipeline | 994.3 MiB | 494.3 MiB | 50.3% |
| Original → blockwise computation | Reference-split full pipeline | 535.4 MiB | 440.2 MiB | 17.8% |
| Blockwise computation → regeneration | Feature selection | 486.3 MiB | 190.7 MiB | 60.8% |
| Blockwise computation → regeneration | Full pipeline | 494.0 MiB | 200.6 MiB | 59.4% |
| Blockwise computation → regeneration | Reference-split full pipeline | 440.4 MiB | 197.4 MiB | 55.2% |

The intermediate implementation was measured again in the second round, explaining
the small differences between its values in the two comparisons. Training-only
and application-only RAM was essentially unchanged.

### Full-size capacity test: 10,000 samples × 5,000 features

A separate synthetic test completed feature selection with all **12,497,500
distinct protein pairs**, without feature filtering or pair subsampling:

| Measurement | Result |
| --- | ---: |
| Peak resident RAM | **1.612 GiB** |
| Peak committed memory | **2.837 GiB** |
| Feature-selection runtime | **26.25 minutes** |
| Selected rules | 15 |

This single run includes the in-memory float64 input, libraries and evaluator
allocations, but excludes CSV parsing, reference splitting and model training or
application. Committed memory is distinct from resident physical RAM. The test
demonstrates computational capacity, not biological prediction quality; the dense
implementation was not run at this size. Other datasets and run settings can have
different memory and runtime requirements.

### Results preserved and reproducibility

Each evaluation round passed **14 byte-for-byte before/after comparisons** across
eight cases, including selected features, scores, p-values, model information,
serialized models, class predictions and RF/SVM probabilities. In the included
full-pipeline test, mean cross-validation accuracy remained **0.99** and validation
accuracy remained **0.98**. The final implementation also passed **258 unit tests**
and **all five original regression tests**, with explicit checks for hash
collisions, duplicate rules, ties, block boundaries and randomization behavior.

See the [initial optimization report](docs/memory_optimization_benchmark.md) and
the [regeneration implementation and benchmark report](docs/regeneration.md) for
dependency versions, measurement details, limitations and reproduction commands.

## Run Modes
NIFty can be executed in several modes depending on which steps of the pipeline you want to run:
1. **find_features**: Generate and score rules to find the best *k* features for classification.
2. **train_model**: Train a machine learning classifier using the selected features.
3. **apply_model**: Apply the trained classifier on experimental, unlabeled data.

You can control this behavoir using a `.toml` configuration file.

## File Formats and Descriptions
* A description of all necessary input files and their required formats can be found [here](docs/input_file_formats.md).
* A description of all output files can be found [here](docs/output_file_formats.md).

## Use Cases
Each of the use-case documents below contain the following information: (1) a brief description about when to run a particular use case of NIFty; and (2) changes to default configurations needed to run that particular use case (to see a default configuration file, see [**File Formats and Descriptions**](#file-formats-and-descriptions)).

* [Full Pipeline (Feature Selection, Model Training, and Model Application)](docs/run_full_pipeline.md)
* [Feature Selection](docs/run_feature_selection.md)
* [Model Training](docs/run_model_training.md)
* [Model Application](docs/run_model_application.md)
* [Feature Selection and Model Training](docs/run_feature_selection_and_model_training.md)
* [Model Training and Model Application](docs/run_model_training_and_application.md)

## Citing NIFty

Classification with Missing Data - A NIFty Pipeline for Single-Cell Proteomics

Alyssa A Nitz, Benjamin Echarry, Blake McGee, Samuel H Payne

bioRxiv 2026.03.06.710179; doi: https://doi.org/10.64898/2026.03.06.710179 
