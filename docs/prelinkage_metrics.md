# Pre-linkage metrics

Pre-linkage metrics are necessary to evaluate the potential for linkage. If
variables common to both datasets are too few or not representative enough of
the datasets, then success of linkage cannot be guaranteed and it is recommended
to look for additional or alternative common variables before proceeding with
linkage. Pre-linkage metrics are available to measure the chances for a linkage
to be successful.

## Unicity

Given a dataset $A$ with variables $V_A$, the unicity score for a set of shared
variables $V_S$ is given by the ratio between number of unique value
combinations of $V_S$ (its cardinality *card(V_S)*) and the number of records in
the dataset:

$$
U(V_S, A) = \frac{card(V_S)}{|A|}
$$

The unicity score represents how unique records are when only defined by the
common variables $V_S$. The best possible linkage variable is a direct
identifier as one value represents exactly one individual. Its unicity score
is 1. On the other hand, a variable such as *gender* is an example of a variable
with poor unicity score. If $V_S$ only contains such a variable, then it should
not be expected that linkage will be successful. Variables with low cardinality
should still be considered for inclusion in $V_S$ because it can give high
unicity score if combined with other variables with low cardinality.


Note that the unicity score needs to be computed for both data sources.

## Contribution score

Common variables $V_S$ can also lead to good linkage if those are representative
of the dataset as a whole. To model this representativeness, a model can be
learnt at a data and contribution of the variables in $V_S$ be computed.

Although many models can be considered, we use factor analysis to project the
data. This type of model is also used when computing some distances in the
linkage phase. Such model represents the data using a set of dimensions, each
explaining for a certain proportion of the variance in the data. For example, a
PCA can represent data with 10 dimensions where the first 3 dimensions will
explain respectively 55%, 25% and 10% of the variance.

For each dimension, it is possible to retrieve the individual contribution of
each variable. For example, a specific variable (*age*) may contribute to 80% of
the first dimension of a PCA.

The pre-linkage contribution score makes use of the concepts of proportion of
variance explained and contribution. For a model $M$ of size (number of
dimensions $|M|$), the contribution of a variable i from $V_S$ to the dimension
$j$ is $\alpha_{i,j}$. For the same model, the variance explained by each
dimension $j$ is $\sigma^2_{j}$. The contribution score is defined as:

$$
  C(V_S, A) = \frac{\sum_{i \in V_S}{\sum_{j \in |M|}{\alpha_{i,j}*\sigma^2_i}}}{\sum_{i \in V_A}{\sum_{j \in |M|}{\alpha_{i,j}*\sigma^2_i}}}
$$

$C(V_S, A)$ is defined between 0 and 1. If $V_S = V_A$, then $C(V_S, A) = 1$.

Note that the contribution score needs to be computed for both data sources.

## Relationship between pre-linkage metrics

Both metrics can be considered prior to performing data linkage. They measure
different concepts and are not correlated with each other.
