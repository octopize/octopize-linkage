# Future work

## Pre-linkage metrics
- Measure contribution of individual variables to dataset (i.e. contribution to projection). Potential correlations involving variables that are almost random (low contributions) are at risk to be lost after anonymization (and linkage cannot fix this). Identifying those variables would be good to know what use case / explorations can be considered on the linked anonymized data.
- Include dataset size (number of individuals) to existing metrics (because it is an important factor of linkage success).

## Linkage methods
- The current solutions do not bias linkage towards specific variables. This could be done to ensure some correlations are kept at linkage.
    - Use LDA as a projection method on which euclidean distances are computed 
    - Use weights on specific variables in saiph projections
- Work on greedy linkage algorithms

## Post-linkage metrics

## Experiments
- Run pipeline on many more datasets

