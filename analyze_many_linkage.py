"""Script analyzing linkages performed under different settings for a given dataset."""

import pandas as pd

from post_linkage_metrics import generate_projection_plot, get_correlation_retention, get_correlations, get_non_shared_var_projections, get_reconstruction_score, plot_correlations
from linkage import Distance, LinkingAlgorithm

stats = {
    "ava_ori": [],
    "distance": [],
    "linkage_algo": [],

    # post-linkage metrics
    "correlation_difference_sum": [],
    "correlation_difference_mean": [],
    "correlation_difference_std": [],
    "correlation_difference_max": [],
    "reconstruction_difference_sum": [],
    "reconstruction_difference_mean": [],
    "reconstruction_difference_std": [],
    "reconstruction_difference_max": [],
}

# Load and prepare original data
original_df = pd.read_csv("data/pra_2023.csv")
# drop IDs and constant columns
original_df = original_df.drop(columns=['household_number', 'survey_year', 'rowid', 'matricule'])

should_be_categorical_columns_1 = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2']
should_be_categorical_columns_2 = ['nationality', 'place_birth', 'sex', 'province', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']
for col in should_be_categorical_columns_1:
    original_df[col] = original_df[col].astype(object)
for col in should_be_categorical_columns_2:
    original_df[col] = original_df[col].astype(object)

# Select common/shared columns
shared_columns = ['sex', 'nationality', 'age', 'province', 'place_birth']

# Load split data to get column divisions
df1_avatars = pd.read_csv("data/pra_A_unshuffled_avatars.csv")
df2_avatars = pd.read_csv("data/pra_B_unshuffled_avatars.csv")
columns1 = set(df1_avatars.columns) - set(shared_columns)
columns2 = set(df2_avatars.columns) - set(shared_columns)

LINK_ORI_AVA = ["original", "avatars"]
linkage_algos = [LinkingAlgorithm.LSA] # [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
distances = [Distance.GOWER, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
# distances = [Distance.GOWER, Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_SECOND_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]

for ori_ava in LINK_ORI_AVA:
        for linkage_algo in linkage_algos:
            for distance in distances:
                # Load and prepare linked data
                linked_df = pd.read_csv(f"data/pra_linked_data__{ori_ava}__{linkage_algo.value}__{distance.value}.csv")
                
                for col in should_be_categorical_columns_1:
                    linked_df[col] = linked_df[col].astype(object)
                for col in should_be_categorical_columns_2:
                    linked_df[col] = linked_df[col].astype(object)

                # Compute and store statistics
                # statistics
                corr_records, corr_avatars = get_correlations(original_df, linked_df, list(columns1), list(columns2))
                corr_retention_stats = get_correlation_retention(original_df, linked_df, list(columns1), list(columns2))
                
                reconstruction_stats = get_reconstruction_score(original_df, linked_df)

                stats["ava_ori"].append(ori_ava)
                stats["distance"].append(distance.value)
                stats["linkage_algo"].append(linkage_algo.value)

                stats["correlation_difference_sum"].append(corr_retention_stats['corr_diff_sum'])
                stats["correlation_difference_mean"].append(corr_retention_stats['corr_diff_mean'])
                stats["correlation_difference_std"].append(corr_retention_stats['corr_diff_std'])
                stats["correlation_difference_max"].append(corr_retention_stats['corr_diff_max'])
        
                # stats["reconstruction_score"].append(reconstruction_score)
                stats["reconstruction_difference_sum"].append(reconstruction_stats['reconstruction_diff_sum'])
                stats["reconstruction_difference_mean"].append(reconstruction_stats['reconstruction_diff_mean'])
                stats["reconstruction_difference_std"].append(reconstruction_stats['reconstruction_diff_std'])
                stats["reconstruction_difference_max"].append(reconstruction_stats['reconstruction_diff_max'])

                # Compute and store plots
                plt = plot_correlations(corr_records, corr_avatars, title=f"{linkage_algo.value} / {distance.value}")
                plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_correlations.png")

                proj_original, proj_linked = get_non_shared_var_projections(original_df, linked_df, list(columns1), list(columns2))
                plt = generate_projection_plot(proj_original, proj_linked, title=f"Projection of non-shared variables for {linkage_algo}/{distance}")
                plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_non_shared_var_projections.png")

                stats_df = pd.DataFrame(stats)
                print(' ==== stats_df ====')
                print(stats_df)

stats_df = pd.DataFrame(stats)
date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
stats_df.to_csv(f"data/many_linkage_stats_{date}.csv", index=False)

print(' ==== Final stats_df ====')
print(stats_df)