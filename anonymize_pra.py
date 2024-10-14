import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")

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

## Connection to server

# url = os.environ.get("AVATAR_PROD_URL")
# username = os.environ.get("AVATAR_PROD_USERNAME")
# password = os.environ.get("AVATAR_PROD_PASSWORD")

url = os.environ.get("AVATAR_BASE_URL")
username = os.environ.get("AVATAR_USERNAME")
password = os.environ.get("AVATAR_PASSWORD")

client = ApiClient(base_url=url)
client.authenticate(username=username, password=password)

source = "pra_A"
# source = "pra_B"


## Data preparation
df = pd.read_csv(f"./data/{source}.csv")
df = df.drop(columns=["matricule"])  # drop IDs
if source == "pra_A":
    should_be_categorical_columns = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2']
else:
    should_be_categorical_columns = ['nationality', 'place_birth', 'sex', 'province', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']

for col in should_be_categorical_columns:
    df[col] = df[col].astype(object)



## Data loading

dataset = client.pandas_integration.upload_dataframe(df)
dataset = client.datasets.analyze_dataset(dataset.id)
print(f"Lines: {dataset.nb_lines}, dimensions: {dataset.nb_dimensions}")

## Avatarization

avatarization_job = client.jobs.create_avatarization_job(
    AvatarizationJobCreate(
        parameters=AvatarizationParameters(k=10, dataset_id=dataset.id, use_categorical_reduction=True),
    )
)

avatarization_job = client.jobs.get_avatarization_job(
    avatarization_job.id, timeout=1800
)
print(avatarization_job.status)

## Privacy metrics

# if source == "pra_A":
#     known_variables=["Gender", "Age", "Height", "family_history_with_overweight", "SMOKE", "MTRANS"],
#     target="NObeyesdad",
# else:
#     known_variables=["Gender", "Age", "Height", "family_history_with_overweight", "SMOKE", "MTRANS"],
#     target="NObeyesdad",

privacy_job = client.jobs.create_privacy_metrics_job(
    PrivacyMetricsJobCreate(
        parameters=PrivacyMetricsParameters(
            original_id=dataset.id,
            unshuffled_avatars_id=avatarization_job.result.sensitive_unshuffled_avatars_datasets.id,
            closest_rate_percentage_threshold=0.3,
            closest_rate_ratio_threshold=0.3,
            # known_variables=known_variables,
            # target=target,
            use_categorical_reduction=True
        ),
    )
)

privacy_job = client.jobs.get_privacy_metrics(privacy_job.id, timeout=1800)
print(privacy_job.status)
print("*** Privacy metrics ***")
for metric in privacy_job.result:
    print(metric)


## Signal metrics

signal_job = client.jobs.create_signal_metrics_job(
    SignalMetricsJobCreate(
        parameters=SignalMetricsParameters(
            original_id=dataset.id,
            avatars_id=avatarization_job.result.avatars_dataset.id,
        ),
    )
)

signal_job = client.jobs.get_signal_metrics(signal_job.id, timeout=1800)
print(signal_job.status)
print("*** Utility metrics ***")
for metric in signal_job.result:
    print(metric)

## Data output

# Download the avatars as a pandas dataframe
avatars_df = client.pandas_integration.download_dataframe(
    avatarization_job.result.avatars_dataset.id
)

# Download the unshuffled avatars as a pandas dataframe
unshuffled_avatars_df = client.pandas_integration.download_dataframe(
    avatarization_job.result.sensitive_unshuffled_avatars_datasets.id
)

# Save the avatars to csv files
avatars_str = avatars_df.to_csv(index=False)
with open(f"./data/{source}_avatars.csv", "wb") as f:
    f.write(avatars_str.encode())

unshuffled_avatars_str = unshuffled_avatars_df.to_csv(index=False)
with open(f"./data/{source}_unshuffled_avatars.csv", "wb") as f:
    f.write(unshuffled_avatars_str.encode())

fig  = df.hist()[0][0].get_figure()
fig.savefig(f'./data/{source}_distributions_original.png')

fig = avatars_df.hist()[0][0].get_figure()
fig.savefig(f'./data/{source}_distributions_avatars.png')


## Report

report = client.reports.create_report(
    ReportCreate(
        avatarization_job_id=avatarization_job.id,
        privacy_job_id=privacy_job.id,
        signal_job_id=signal_job.id,
    ),
    timeout=240,
)

result = client.reports.download_report(id=report.id)

with open(f"./data/{source}_avatarization_report.pdf", "wb") as f:
    f.write(result)

