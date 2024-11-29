# octopize-linkage

## Setup
Dependencies are managed with poetry and nix. Installation steps are given below:

### poetry

```bash
poetry lock
poetry install
```

### nix

A `default.nix` is provided at the root of the project. Build the environment using:

```bash
nix-build
```

Drop into an interactive shell using:

```bash
nix-shell
```

From this shell, you can start your IDE which should then have access to the packages.

## Run use cases

Use case scripts are available to run all steps in one go and see how the whole pipeline can be executed.

```bash
poetry run python run_use_case_pra.py
```

or, if you’re using nix:

```bash
nix-shell --run 'python run_use_case_pra.py'
```

## Run tests 

```bash
poetry run pytest <test_feature.py>
```

## Run anonymization with avatar

Generation of anonymous synthetic data can be done with the solution avatar, provided a license to the avatar solution is in place. 

```bash
poetry run python anonymize_pra.py
```

## Run linkages

The `run_many_linkage.py` script can be used to perform linkage between two datasets (anonymized or original). It enables analysis of a specific use case under different linkage settings (different distances and algorithms).
The script generates csv files of the linked data.

```bash
poetry run python run_many_linkage.py
nix-shell --run 'python run_many_linkage.py'
```

## Analyze linkages

The analyze_many_linkage.py script can be used to analyse the results obtained in the previous step (`run_many_linkage.py`).
If `random` and `row_order` distances have been included in the runs, then any other method can be compared to a close-to-ideal linkage (`row_order`) and to a bad linkage (`random`).

Make sure the selected settings match those of `run_many_linkage.py`.

```bash
poetry run python analyze_many_linkage.py
nix-shell --run 'python analyze_many_linkage.py'
```

## Run and analyze many scenarios (for experimental purposes)

The `run_many_pipelines.py` script can be used to perform linkage between two datasets (anonymized or original) for many scenarios. Scenarios are defined by a dataset (several open source datasets are available), by a set of common variables (sets of different sizes), by the use of original split data or their avatars and under different linkage settings. The script generates a csv file containing linkage metrics.

The `analyze_many_pipelines.py` scripts can then be used to generate plots of this metric data.

```bash
# Perform many linkages and compute metrics
poetry run python run_many_pipelines.py
nix-shell --run 'python run_many_pipelines.py'

# Analyze
poetry run python analyze_many_pipelines.py
nix-shell --run 'python analyze_many_linkage.py'
```


## Documentation :books:

The [documentation](./docs/intro.md) is available in `\docs`.

