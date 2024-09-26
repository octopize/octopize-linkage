"""Example of how to call the different bricks on the PRA dataset."""

import pandas as pd

from pre_linkage_metrics import ImputeMethod, contribution_score


################################
# Load and prepare data
################################
df = pd.read_csv("data/pra_2023.csv")
print(df)
print(df.columns)

# drop IDs and constant columns
df = df.drop(columns=['household_number', 'survey_year'])

# replace comma by dot in column elevator
df['elevator'] = df['elevator'].str.replace(',', '.').astype(float)

################################
# Split data
################################
all_columns = df.columns

# Select common/shared columns
shared_columns = ['territory', 'sex', 'nationality', 'age', 'main_occupation']

# Split data into two sources df1 and df2
columns1 = set(all_columns[:15]).union(set(shared_columns))
columns2 = set(all_columns[15:]).union(set(shared_columns))

print("Columns of data source 1: ", columns1)
print("Columns of data source 2: ", columns2)
print("Columns in common (shared columns): ", columns1.intersection(columns2))

df1 = df[list(columns1)]
df2 = df[list(columns2)]

################################
# Pre-linkage metrics
################################
# ?? How representative of the dataset at source 1 are my common variables ?
contribution_score_dict1 = contribution_score(
    df=df1, 
    shared_columns=shared_columns, 
    target_explained_variance=0.9,
    impute_method=ImputeMethod.MEDIAN,
    should_consider_missing=True,
    seed=None)
print("\n\nContribution score for source 1: \n", contribution_score_dict1)

# ?? How representative of the dataset at source 2 are my common variables ?
contribution_score_dict2 = contribution_score(
    df=df2, 
    shared_columns=shared_columns, 
    target_explained_variance=0.9,
    impute_method=ImputeMethod.MEDIAN,
    should_consider_missing=True,
    seed=None)
print("\n\nContribution score for source 2: \n", contribution_score_dict2)

print("\n\n We recommend using `selected_columns_relative_contribution_score` which is bounded between 0 and 1:")
print(f"\n\t score of source 1: {contribution_score_dict1['selected_columns_relative_contribution_score']}")
print(f"\t score of source 2: {contribution_score_dict2['selected_columns_relative_contribution_score']}")

################################
# Linkage
################################


################################
#Post-linkage metrics
################################

