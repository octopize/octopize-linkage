"""Example of how to call the different bricks on the PRA dataset."""

import pandas as pd

from post_linkage_metrics import generate_projection_plot, get_correlations, get_non_shared_var_projections, plot_correlations
from pre_linkage_metrics import ImputeMethod, contribution_score, get_best_n_from_m_variables, get_unicity_score
from linkage import Distance, link_datasets, LinkingAlgorithm



should_be_categorical_columns_1 = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2']
should_be_categorical_columns_2 = ['nationality', 'place_birth', 'sex', 'province', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']



# Select common/shared columns
shared_columns = ['sex', 'nationality', 'age', 'province', 'place_birth']


# Load split data to get column divisions
df1_avatars = pd.read_csv("data/pra_A_unshuffled_avatars.csv")
df2_avatars = pd.read_csv("data/pra_B_unshuffled_avatars.csv")
columns1 = set(df1_avatars.columns) - set(shared_columns)
columns2 = set(df2_avatars.columns) - set(shared_columns)



stats = {
    "ava_ori": [],
    "distance": [],
    "linkage_algo": [],
    "corr_diff_sum": []
}

#### Get statistics from original unified PRA data
original_df = pd.read_csv("data/pra_2023.csv")
# drop IDs and constant columns
original_df = original_df.drop(columns=['household_number', 'survey_year', 'rowid', 'matricule'])

for col in should_be_categorical_columns_1:
    original_df[col] = original_df[col].astype(object)
for col in should_be_categorical_columns_2:
    original_df[col] = original_df[col].astype(object)


stats["ava_ori"].append("original")
stats["distance"].append("unlinked")
stats["linkage_algo"].append("unlinked")
stats["corr_diff_sum"].append(0)


# #### Get statistics from anonymized unified PRA data
# avatars_df = ...

# stats["ava_ori"].append("avatars")
# stats["distance"].append("unlinked")
# stats["linkage_algo"].append("unlinked")


# #### Get statistics from each original+linked PRA data
# linked_original_df = ...

# stats["ava_ori"].append("original")
# stats["distance"].append(...)
# stats["linkage_algo"].append(...)


#### Get statistics from each anonymized+linked PRA data

linkage_algos = [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
distances = [Distance.GOWER, Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_SECOND_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]

# linkage_algos = [LinkingAlgorithm.LSA]
# distances = [Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.RANDOM]

# linkage_algos = [LinkingAlgorithm.MIN_ORDER]
# distances = [Distance.RANDOM]


for linkage_algo in linkage_algos:
    for distance in distances:
        df = pd.read_csv(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}.csv")
        
        for col in should_be_categorical_columns_1:
            df[col] = df[col].astype(object)
        for col in should_be_categorical_columns_2:
            df[col] = df[col].astype(object)

        corr_records, corr_avatars, corr_diff = get_correlations(original_df, df, list(columns1), list(columns2))

        print(corr_avatars)
        print(corr_records)
        print('corr_diff.sum().sum():', corr_diff.sum().sum())


        plt = plot_correlations(corr_records, corr_avatars, title=f"{linkage_algo.value} / {distance.value}")
        plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_correlations.png")

        # assert False


        proj_original, proj_linked = get_non_shared_var_projections(original_df, df, list(columns1), list(columns2))
        print('proj_original: ', proj_original)
        print('proj_linked: ', proj_linked)

        plt = generate_projection_plot(proj_original, proj_linked, title=f"Projection of non-shared variables for {linkage_algo}/{distance}")
        plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_non_shared_var_projections.png")

        stats["ava_ori"].append("avatars")
        stats["distance"].append(distance.value)
        stats["linkage_algo"].append(linkage_algo.value)
        stats["corr_diff_sum"].append(corr_diff.sum().sum())

stats_df = pd.DataFrame(stats)
print(stats_df)