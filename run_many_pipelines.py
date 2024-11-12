"""Example of how to call the different bricks on the PRA dataset."""

import os
import random
import numpy as np
import pandas as pd

from post_linkage_metrics import generate_projection_plot, get_correlations, get_non_shared_var_projections, plot_correlations
from pre_linkage_metrics import ImputeMethod, contribution_score, get_best_n_from_m_variables, get_unicity_score
from linkage import Distance, link_datasets, LinkingAlgorithm

from avatars.client import ApiClient
from avatars.models import (
    AvatarizationJobCreate,
    AvatarizationParameters,
    ReportCreate,
    PrivacyMetricsJobCreate,
    PrivacyMetricsParameters,
    SignalMetricsJobCreate,
    SignalMetricsParameters,
)

################################
# Server connection
################################

# url = os.environ.get("AVATAR_PROD_URL")
# username = os.environ.get("AVATAR_PROD_USERNAME")
# password = os.environ.get("AVATAR_PROD_PASSWORD")

url = os.environ.get("AVATAR_BASE_URL")
username = os.environ.get("AVATAR_USERNAME")
password = os.environ.get("AVATAR_PASSWORD")

client = ApiClient(base_url=url)
client.authenticate(username=username, password=password)


################################
# Load and prepare data
################################
# df = pd.read_csv("data/pra_2023.csv")
_df1 = pd.read_csv("data/pra_A.csv")
_df2 = pd.read_csv("data/pra_B.csv")
_shared_cols = ['sex', 'nationality', 'age', 'province', 'place_birth']
_df2 = _df2.drop(columns=_shared_cols)
# concatenate the two sources
df = pd.concat([_df1, _df2], axis=1)
df = df.drop(columns=["matricule"])  # drop IDs

print(df)

# drop IDs and constant columns
# df = df.drop(columns=['household_number', 'survey_year', 'rowid', 'matricule'])
# df = df.drop(columns=["elevator", "JORB","formal_educ_system" ,"professional_training" ,"retirement_status","part-time_empl", "reason_reduced_work"])

should_be_categorical_columns = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']
for col in should_be_categorical_columns:
    df[col] = df[col].astype(object)

all_columns = list(df.columns)

number_of_random_column_combinations = 10
min_number_of_random_column_in_combinations = 2
max_number_of_random_column_in_combinations = 8
# max_number_of_random_column_in_combinations = len(all_columns)

# get a random sample of columns
random_columns_combinations = []
for i in range(number_of_random_column_combinations):
    number_of_columns_in_combination = np.random.randint(min_number_of_random_column_in_combinations, max_number_of_random_column_in_combinations)
    random_columns_combinations.append(random.sample(all_columns, number_of_columns_in_combination))

combination_dict = {}
for combination_i, shared_columns in enumerate(random_columns_combinations):
    combination_dict[combination_i] = shared_columns
# save combination_dict to text file one line per combination

date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
with open(f"data/random_column_combinations_{date}.txt", "w") as f:
    for key, value in combination_dict.items():
        f.write(f"{key}\t {value}\n")

print("combination_dict: ", combination_dict)


LINK_ORI_AVA = ["original", "avatars"]
# linkage_algos = [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
linkage_algos = [LinkingAlgorithm.LSA]

# distances = [Distance.GOWER, Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_SECOND_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
distances = [Distance.GOWER, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
# distances = [Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
# distances = [Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER]
should_shuffle_before_linkage = True

stats = {
    "combination_id": [],
    "number_of_shared_variables": [],
    "total_number_variables": [],
    "contribution_score1": [],
    "unicity_score1": [],
    "contribution_score2": [],
    "unicity_score2": [],
    "ava_ori": [],
    "distance": [],
    "linkage_algo": [],
    "corr_diff_sum": []
}

for combination_i, shared_columns in enumerate(random_columns_combinations):
    # shared_columns = ['sex', 'nationality', 'age', 'province', 'place_birth']  # DEBUG

    print("shared_columns: ", shared_columns)
    ################################
    # Split data
    ################################
    
    # Split data into two sources df1 and df2
    columns1 = set(all_columns[:15]).union(set(shared_columns))
    columns2 = set(all_columns[15:]).union(set(shared_columns))

    df1 = df[list(columns1)].copy()
    df2 = df[list(columns2)].copy()

    for ori_ava in LINK_ORI_AVA:
        print(f"\n\nRunning with ori_ava: {ori_ava} ...")
        if ori_ava == "avatars":

            ################################
            # Avatarization - SRC 1
            ################################

            ## Data loading
            dataset1 = client.pandas_integration.upload_dataframe(df1)
            dataset1 = client.datasets.analyze_dataset(dataset1.id)

            ## Avatarization

            avatarization_job1 = client.jobs.create_avatarization_job(
                AvatarizationJobCreate(
                    parameters=AvatarizationParameters(k=10, dataset_id=dataset1.id, use_categorical_reduction=True),
                )
            )

            avatarization_job1 = client.jobs.get_avatarization_job(
                avatarization_job1.id, timeout=1800
            )
            print(avatarization_job1.status)


            ################################
            # Avatarization - SRC 2
            ################################

            ## Data loading
            dataset2 = client.pandas_integration.upload_dataframe(df2)
            dataset2 = client.datasets.analyze_dataset(dataset2.id)

            ## Avatarization

            avatarization_job2 = client.jobs.create_avatarization_job(
                AvatarizationJobCreate(
                    parameters=AvatarizationParameters(k=10, dataset_id=dataset2.id, use_categorical_reduction=True),
                )
            )

            avatarization_job2 = client.jobs.get_avatarization_job(
                avatarization_job2.id, timeout=1800
            )
            print(avatarization_job2.status)

            df1_avatars = client.pandas_integration.download_dataframe(
                avatarization_job1.result.sensitive_unshuffled_avatars_datasets.id
            )
            df2_avatars = client.pandas_integration.download_dataframe(
                avatarization_job2.result.sensitive_unshuffled_avatars_datasets.id
            )
        else:
            df1_avatars = df1
            df2_avatars = df2


        ################################
        # Pre-linkage metrics
        ################################
        # ?? How representative of the dataset at source 1 are my common variables ?
        contribution_score_dict1 = contribution_score(
            df=df1_avatars, 
            shared_columns=shared_columns, 
            target_explained_variance=0.9,
            impute_method=ImputeMethod.MEDIAN,
            should_consider_missing=True,
            seed=None)

        # ?? How representative of the dataset at source 2 are my common variables ?
        contribution_score_dict2 = contribution_score(
            df=df2_avatars, 
            shared_columns=shared_columns, 
            target_explained_variance=0.9,
            impute_method=ImputeMethod.MEDIAN,
            should_consider_missing=True,
            seed=None)

        n_unique_comb1 = get_unicity_score(df1_avatars, shared_columns)
        n_unique_comb2 = get_unicity_score(df2_avatars, shared_columns)

        ################################
        # Linkage
        ################################

        for linkage_algo in linkage_algos:
            for distance in distances:
                _df1_avatars = df1_avatars.copy()
                _df2_avatars = df2_avatars.copy()
                if distance != Distance.ROW_ORDER and should_shuffle_before_linkage:
                    _df2_avatars = _df2_avatars.sample(frac=1).reset_index(drop=True)
                print(f"\n\nRunning with linkage algorithm: {linkage_algo.value}, distance: {distance.value} ...")
                # link the two sources
                linked_df = link_datasets(_df1_avatars, _df2_avatars, shared_columns, distance=distance, linking_algo=linkage_algo)
                linked_df.to_csv(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}.csv", index=False)
                print("\n\nLinked data: \n", linked_df)


                ################################
                # Post linkage Metrics
                ################################
                _columns1 = set(df1_avatars.columns) - set(shared_columns)
                _columns2 = set(df2_avatars.columns) - set(shared_columns)

                # corr_records, corr_avatars, corr_diff = get_correlations(df, linked_df, list(columns1), list(columns2))
                corr_records, corr_avatars, corr_diff = get_correlations(df, linked_df, list(_columns1), list(_columns2))

                plt = plot_correlations(corr_records, corr_avatars, title=f"{linkage_algo.value} / {distance.value}")
                plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_correlations.png")


                proj_original, proj_linked = get_non_shared_var_projections(df, linked_df, list(_columns1), list(_columns2))

                plt = generate_projection_plot(proj_original, proj_linked, title=f"Projection of non-shared variables for {linkage_algo}/{distance}")
                plt.savefig(f"data/pra_linked_data__avatar__{linkage_algo.value}__{distance.value}_non_shared_var_projections.png")

                stats["combination_id"].append(combination_i)
                stats["number_of_shared_variables"].append(len(shared_columns))
                stats["total_number_variables"].append(len(all_columns))
                stats["contribution_score1"].append(contribution_score_dict1['selected_columns_relative_contribution_score'])
                stats["unicity_score1"].append(n_unique_comb1)
                stats["contribution_score2"].append(contribution_score_dict2['selected_columns_relative_contribution_score'])
                stats["unicity_score2"].append(n_unique_comb2)

                stats["distance"].append(distance.value)
                stats["linkage_algo"].append(linkage_algo.value)
                stats["corr_diff_sum"].append(corr_diff.sum().sum())
                stats["ava_ori"].append(ori_ava)

                stats_df = pd.DataFrame(stats)
                print(' ==== stats_df ====')
                print(stats_df)


stats_df = pd.DataFrame(stats)
# save stats to csv
date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
stats_df.to_csv(f"data/many_pipeline_stats_{date}.csv", index=False)

print(' ==== Final stats_df ====')
print(stats_df)