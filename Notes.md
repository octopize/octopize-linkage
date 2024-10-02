# Notes


## Pre-linkage metrics

Good linkage variables should be:
- as much unique as possible. The extreme case is a direct identifier, it is the best for linkage but not available in theory (or there is no need for linkage). But combinations of shared variables' values can be unique.
- accounting for a lot of variance in the data. In other words, variables are structuring or representing well the data
- be correlated / or predictive of the target variable(s) (when information on target use of linked data is known). 

So complementary metrics can be:
- Measure of "uniqueness" or "unicity"
- Measure of representativeness of the data
- Measure of predictability of target variable(s)

At least one of those measures should be good ("Good" is yet to be defined):
- A direct identifier will be good on uniqueness but bad on representativeness of the data and bad in predictability
- On some data, a set of variables will not be good in uniqueness but good on representativeness
- A very large set of variables will be good on uniqueness and representativeness
- In some cases, both uniqueness and representativeness can be bad but predictability be good.