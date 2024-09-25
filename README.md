# octopize-linkage

## Setup
Dependencies are managed with poetry. Installation steps are given below:

```bash
poetry lock
poetry install
```

## Planned linkage steps

1. Evaluate linkage potential
Identify common/shared variables between two datasets and evaluate their correlation with the other variables in both datasets
    - if low correlations: impossible to perform linkage (user ned to find a way to get more common variables to "help" linkage)
    - Note: it is important to have correlation with variables in **both** datasets
    - For experiments, this can be evaluated on original data
(If there are many shared variables, find the optimum set of variables to use for linkage (if fewer = better))
    
2. Perform linkage
    - several strategies available
    - need to find the more robust one(s), able to work well on a wide range of datasets

3. Evaluate linkage
    - linkage metrics should be made available
    - metrics will mainly be used in development stage because it may be difficult to assess a good linkage if we cannot get access to both original datasets to get the "ground truth" informations (pairing, correlations etc...).
    - Can we think of linkage metrics computed **without** original datasets ?
    - Comparison of `original_full_df` and `linked(avatar_split_1_df, avatar_split_2_df)`
    - E.g. of metrics:
        - Correlations between variables in data1 (x-axis) and variables in data2 (y-axis) - difference between original "ground truth" correlations and correlations on linked data

4. Correlate pre and post linkage metrics
    - Objective: find out if pre-linkage metrics can predict post-linkage metrics because in practice, we will not be able to run post-linkage metrics.

