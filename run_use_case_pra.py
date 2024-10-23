"""Example of how to call the different bricks on the PRA dataset."""

import pandas as pd

from pre_linkage_metrics import ImputeMethod, contribution_score, get_best_n_from_m_variables, get_unicity_score
from linkage import Distance, link_datasets, LinkingAlgorithm

################################
# Load and prepare data
################################
df = pd.read_csv("data/pra_2023.csv")

# drop IDs and constant columns
df = df.drop(columns=['household_number', 'survey_year', 'rowid', 'matricule'])

# replace comma by dot in column elevator
# df['elevator'] = df['elevator'].str.replace(',', '.').astype(float)

################################
# Split data
################################
all_columns = df.columns

# Select common/shared columns
shared_columns = ['sex', 'nationality', 'age', 'province', 'place_birth']

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


print("\n\n In some cases, we may be interested in getting the best n variables from m variables potentially available.")
n = 3
best_variables1 = get_best_n_from_m_variables(contribution_score_dict1['contribution_score_per_variable'], n=n)
best_variables_from_shared_variables1 = get_best_n_from_m_variables(contribution_score_dict1['contribution_score_per_variable'], n=3, m_variables=shared_columns)
print(f"\n\tBest {n} variables at source 1: {best_variables1}")
print(f"\tBest {n} variables from shared variables at source 1: {best_variables_from_shared_variables1}")

best_variables2 = get_best_n_from_m_variables(contribution_score_dict2['contribution_score_per_variable'], n=n)
best_variables_from_shared_variables2 = get_best_n_from_m_variables(contribution_score_dict2['contribution_score_per_variable'], n=3, m_variables=shared_columns)
print(f"\tBest {n} variables at source 2: {best_variables2}")
print(f"\tBest {n} variables from shared variables at source 2: {best_variables_from_shared_variables2}")


n_unique_comb1 = get_unicity_score(df1, shared_columns)
print(f"\n\tUnicity score of source 1: {n_unique_comb1}")

n_unique_comb2 = get_unicity_score(df2, shared_columns)
print(f"\tUnicity score of source 2: {n_unique_comb2}")

################################
# Linkage
################################

# load avatars from both sources
df1_avatars = pd.read_csv("data/pra_A_unshuffled_avatars.csv")
df2_avatars = pd.read_csv("data/pra_B_unshuffled_avatars.csv")

should_be_categorical_columns_1 = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2']
should_be_categorical_columns_2 = ['nationality', 'place_birth', 'sex', 'province', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']

for col in should_be_categorical_columns_1:
    df1_avatars[col] = df1_avatars[col].astype(object)
for col in should_be_categorical_columns_2:
    df2_avatars[col] = df2_avatars[col].astype(object)


# link the two sources
distance=Distance.PROJECTION_DIST_ALL_SOURCES
# linking_algo=LinkingAlgorithm.MIN_REORDER
linking_algo=LinkingAlgorithm.MIN_ORDER

# linked_df = link_datasets(df1_avatars, df2_avatars, shared_columns, distance=Distance.ROW_ORDER, linking_algo=LinkingAlgorithm.LSA)
linked_df = link_datasets(df1_avatars, df2_avatars, shared_columns, distance=distance, linking_algo=linking_algo)

linked_df.to_csv(f"data/pra_linked_data__avatar__{linking_algo.value}__{distance.value}.csv", index=False)
print("\n\nLinked data: \n", linked_df)

################################
#Post-linkage metrics
################################

