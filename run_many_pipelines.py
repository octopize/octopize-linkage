"""Script performing linkage under different settings for different scenarios and different datasets."""

import os
import random
import numpy as np
import pandas as pd

from data_loader import Dataset, load_dataset
from post_linkage_metrics import generate_projection_plot, get_correlation_retention, get_correlations, get_non_shared_var_projections, get_reconstruction_score, plot_correlations
from pre_linkage_metrics import ImputeMethod, contribution_score, get_unicity_score
from linkage import Distance, link_datasets, LinkingAlgorithm

from avatars.client import ApiClient
from avatars.models import (
    AvatarizationJobCreate,
    AvatarizationParameters,
    PrivacyMetricsJobCreate,
    PrivacyMetricsParameters,
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

number_of_random_column_combinations = 10  # per dataset
dataset_names = [Dataset.STUDENT_DROPOUT, Dataset.STUDENT_PERFORMANCE, Dataset.ADULT, Dataset.PRA]
# dataset_names = [Dataset.PRA]
dataset_number_of_records = {
    Dataset.ADULT: 10000,
    Dataset.PRA: None,
    Dataset.STUDENT_PERFORMANCE: None,
    Dataset.STUDENT_DROPOUT: None
    }

LINK_ORI_AVA = ["avatars", "original"]
linkage_algos = [LinkingAlgorithm.LSA] # [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
distances = [Distance.GOWER, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
# distances = [Distance.GOWER, Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_SECOND_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]

should_shuffle_before_linkage = True

stats = {
    "dataset": [],
    "combination_id": [],
    "number_of_shared_variables": [],
    "total_number_variables": [],
    "ava_ori": [],
    "distance": [],
    "linkage_algo": [],

    # pre-linkage metrics
    "contribution_score1": [],
    "unicity_score1": [],
    "contribution_score2": [],
    "unicity_score2": [],

    # post-linkage metrics
    "correlation_difference_sum": [],
    "correlation_difference_mean": [],
    "correlation_difference_std": [],
    "correlation_difference_max": [],
    "reconstruction_difference_sum": [],
    "reconstruction_difference_mean": [],
    "reconstruction_difference_std": [],
    "reconstruction_difference_max": [],
    
    # privacy metrics
    "hr1": [],
    "hr2": [],
    "hr_linked": []
}

date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

for dataset_name in dataset_names:
    print(f"\n\nRunning for dataset: {dataset_name} ...")

    ################################
    # Load and prepare data
    ################################
    data = load_dataset(dataset_name, dataset_number_of_records[dataset_name])
    df = data['df']
    min_number_of_random_column_in_combinations = data['min_number_of_random_column_in_combinations']
    max_number_of_random_column_in_combinations = data['max_number_of_random_column_in_combinations']

    all_columns = list(df.columns)


    # get a random sample of columns
    random_columns_combinations = []
    for i in range(number_of_random_column_combinations):
        number_of_columns_in_combination = np.random.randint(min_number_of_random_column_in_combinations, max_number_of_random_column_in_combinations)
        random_columns_combinations.append(random.sample(all_columns, number_of_columns_in_combination))

    combination_dict = {}
    for combination_i, shared_columns in enumerate(random_columns_combinations):
        combination_dict[combination_i] = shared_columns
    # save combination_dict to text file one line per combination

    with open(f"data/random_column_combinations_{dataset_name.value}_{date}.txt", "w") as f:
        for key, value in combination_dict.items():
            f.write(f"{key}\t {value}\n")

    print("combination_dict: ", combination_dict)

    for combination_i, shared_columns in enumerate(random_columns_combinations):
        ################################
        # Split data
        ################################
        
        # TODO: fix this - makes it generic (15 only works for PRA and even it's not great)
        # idea: shuffle col names
        # split in half
        col_names = list(df.columns)
        random.shuffle(col_names)
        number_of_cols = len(col_names)
        split_index = int(number_of_cols//2)

        # Split data into two sources df1 and df2
        columns1 = set(all_columns[:split_index]).union(set(shared_columns))
        columns2 = set(all_columns[split_index:]).union(set(shared_columns))

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
                print("Avatarization1 job status:", avatarization_job1.status)

                privacy_job1 = client.jobs.create_privacy_metrics_job(
                    PrivacyMetricsJobCreate(
                        parameters=PrivacyMetricsParameters(
                            original_id=dataset1.id,
                            unshuffled_avatars_id=avatarization_job1.result.sensitive_unshuffled_avatars_datasets.id,
                            use_categorical_reduction=True
                        ),
                    )
                )

                privacy_job1 = client.jobs.get_privacy_metrics(privacy_job1.id, timeout=1800)
                print("Privacy1 job status:", privacy_job1.status)
                hr1 = privacy_job1.result.hidden_rate
                # for metric in privacy_job1.result:
                #     print(metric)


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
                print("Avatarization2 job status:", avatarization_job2.status)

                privacy_job2 = client.jobs.create_privacy_metrics_job(
                    PrivacyMetricsJobCreate(
                        parameters=PrivacyMetricsParameters(
                            original_id=dataset2.id,
                            unshuffled_avatars_id=avatarization_job2.result.sensitive_unshuffled_avatars_datasets.id,
                            use_categorical_reduction=True
                        ),
                    )
                )
                privacy_job2 = client.jobs.get_privacy_metrics(privacy_job2.id, timeout=1800)
                print("Privacy2 job status:", privacy_job2.status)
                hr2 = privacy_job2.result.hidden_rate

                df1_avatars = client.pandas_integration.download_dataframe(
                    avatarization_job1.result.sensitive_unshuffled_avatars_datasets.id
                )
                df2_avatars = client.pandas_integration.download_dataframe(
                    avatarization_job2.result.sensitive_unshuffled_avatars_datasets.id
                )
            else:
                df1_avatars = df1
                df2_avatars = df2
                hr1 = np.nan
                hr2 = np.nan


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
                    linked_df.to_csv(f"data/{dataset_name.value}_linked_data__avatar__{linkage_algo.value}__{distance.value}.csv", index=False)
                    print("\n\nLinked data: \n", linked_df)


                    ################################
                    # Post linkage Metrics
                    ################################
                    _columns1 = set(df1_avatars.columns) - set(shared_columns)
                    _columns2 = set(df2_avatars.columns) - set(shared_columns)

                    # statistics
                    corr_records, corr_avatars = get_correlations(df, linked_df, list(_columns1), list(_columns2))
                    corr_retention_stats = get_correlation_retention(df, linked_df, list(_columns1), list(_columns2))
                    
                    reconstruction_stats = get_reconstruction_score(df, linked_df)

                    # plots
                    plt = plot_correlations(corr_records, corr_avatars, title=f"{linkage_algo.value} / {distance.value}")
                    plt.savefig(f"data/{dataset_name.value}_linked_data__avatar__{linkage_algo.value}__{distance.value}_correlations.png")

                    proj_original, proj_linked = get_non_shared_var_projections(df, linked_df, list(_columns1), list(_columns2))
                    plt = generate_projection_plot(proj_original, proj_linked, title=f"Projection of non-shared variables for {linkage_algo}/{distance}")
                    plt.savefig(f"data/{dataset_name.value}_linked_data__avatar__{linkage_algo.value}__{distance.value}_non_shared_var_projections.png")


                    ################################
                    # Privacy Metrics
                    ################################
                    dataset_original = client.pandas_integration.upload_dataframe(df)
                    dataset_linked = client.pandas_integration.upload_dataframe(linked_df)
                    privacy_job_linked = client.jobs.create_privacy_metrics_job(
                        PrivacyMetricsJobCreate(
                            parameters=PrivacyMetricsParameters(
                                original_id=dataset_original.id,
                                unshuffled_avatars_id=dataset_linked.id,
                                use_categorical_reduction=True
                            ),
                        )
                    )
                    privacy_job_linked = client.jobs.get_privacy_metrics(privacy_job_linked.id, timeout=1800)
                    print("PrivacyLinked job status:", privacy_job_linked.status)
                    hr_linked = privacy_job_linked.result.hidden_rate

                    # Compute and store statistics
                    stats["dataset"].append(dataset_name.value)
                    stats["combination_id"].append(combination_i)
                    stats["number_of_shared_variables"].append(len(shared_columns))
                    stats["total_number_variables"].append(len(all_columns))

                    stats["contribution_score1"].append(contribution_score_dict1['selected_columns_relative_contribution_score'])
                    stats["unicity_score1"].append(n_unique_comb1)
                    stats["contribution_score2"].append(contribution_score_dict2['selected_columns_relative_contribution_score'])
                    stats["unicity_score2"].append(n_unique_comb2)

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

                    stats["ava_ori"].append(ori_ava)

                    stats["hr1"].append(hr1)
                    stats["hr2"].append(hr2)
                    stats["hr_linked"].append(hr_linked)

                    stats_df = pd.DataFrame(stats)
                    print(' ==== stats_df ====')
                    print(stats_df)


stats_df = pd.DataFrame(stats)
# save stats to csv
date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
stats_df.to_csv(f"data/many_pipeline_stats_{date}.csv", index=False)

print(' ==== Final stats_df ====')
print(stats_df)

