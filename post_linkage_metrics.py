from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import saiph 
import seaborn as sns
from sklearn.metrics import hamming_loss, mean_squared_error, accuracy_score

from linkage import _compute_gower_distances, _handle_missing_values
from pre_linkage_metrics import _CATEGORICAL_DTYPES, ImputeMethod, enrich_df_with_is_missing_variables, impute_missing_values

def plot_correlations(corr_original, corr_avatars, title: str = None):
  
    # instantiate plot with 2 sub figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # use same color scale for both plots
    vmin = min(corr_original.min().min(), corr_avatars.min().min())
    vmax = max(corr_original.max().max(), corr_avatars.max().max())

    # plot original correlation
    im = ax1.imshow(corr_original, vmin=vmin, vmax=vmax, cmap='bwr')
    ax1.set_title('Original Data')

    # plot avatars correlation
    im = ax2.imshow(corr_avatars, vmin=vmin, vmax=vmax, cmap='bwr')
    ax2.set_title('Avatarized Data')

    # add legend
    # plt.colorbar(ax1.imshow(corr_original))
    # plt.colorbar(ax2.imshow(corr_avatars))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # set title
    if title:
        plt.suptitle(title)

    return plt

def get_correlations(
    df: pd.DataFrame, linked_records: pd.DataFrame, columns1: List[str], columns2: List[str]):
    """Compute correlation statistics between columns from different sources.

    Correlations between columns from the same source are not considered.

    Arguments
    ---------
        df: original data
        linked_records: linked data
        columns1: columns from first data source
        columns2: columns from second data source
    """
    split_records_0 = df[columns1]
    split_records_1 = df[columns2]

    linked_records_0 = linked_records[columns1]
    linked_records_1 = linked_records[columns2]

    # remove linkage/common variables
    shared_columns = list(set(columns1).intersection(set(columns2)))

    split_records_0 = split_records_0.drop(columns=shared_columns)
    split_records_1 = split_records_1.drop(columns=shared_columns)
    linked_records_0 = linked_records_0.drop(columns=shared_columns)
    linked_records_1 = linked_records_1.drop(columns=shared_columns)

    # dummifies
    split_record_dummies_0 = pd.get_dummies(split_records_0)
    split_record_dummies_1 = pd.get_dummies(split_records_1)
    linked_records_dummies_0 = pd.get_dummies(linked_records_0)
    linked_records_dummies_1 = pd.get_dummies(linked_records_1)

    corr_records = pd.concat([split_record_dummies_0, split_record_dummies_1], axis=1, keys=['split_record_dummies_0', 'split_record_dummies_1']).corr().loc['split_record_dummies_1', 'split_record_dummies_0']
    corr_avatars = pd.concat([linked_records_dummies_0, linked_records_dummies_1], axis=1, keys=['linked_records_dummies_0', 'linked_records_dummies_1']).corr().loc['linked_records_dummies_1', 'linked_records_dummies_0']

    return corr_records, corr_avatars


def get_correlation_retention(df: pd.DataFrame, linked_records: pd.DataFrame, columns1: List[str], columns2: List[str]) -> float:
    corr_records, corr_avatars = get_correlations(df, linked_records, columns1, columns2)
    corr_diff = abs(corr_records - corr_avatars)
    sum_of_diff = corr_diff.sum().sum()
    mean_diff = np.mean(corr_diff)
    std_diff = np.mean(np.std(corr_diff))
    max_diff = np.max(corr_diff)
    return {
        'corr_diff_sum': sum_of_diff,
        'corr_diff_mean': mean_diff,
        'corr_diff_std': std_diff,
        'corr_diff_max': max_diff,
    }


def generate_projection_plot(data_ori, data_linked, title: str = None):

    # instantiate plot with 2 sub figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # use same color scale for both plots
    # vmin = min(data_ori.min().min(), corr_avatars.min().min())
    # vmax = max(corr_original.max().max(), corr_avatars.max().max())

    # plot original correlation
    # im = ax1.imshow(corr_original, vmin=vmin, vmax=vmax, cmap='bwr')
    x = data_ori["Dim. 1"]
    y  = data_ori["Dim. 2"]
    # create scatter plot in ax1 with x and y
    ax1.scatter(x, y)


    x = data_linked["Dim. 1"]
    y  = data_linked["Dim. 2"]
    # create scatter plot in ax1 with x and y
    ax2.scatter(x, y)

    # im = ax1.imshow(data_ori[["Dim. 1", "Dim. 2"]], cmap='bwr')
    ax1.set_title('Original Data')

    # plot avatars correlation
    # im = ax2.imshow(data_linked[["Dim. 1", "Dim. 2"]], cmap='bwr')
    ax2.set_title('Linked Data')

    # add legend
    # plt.colorbar(ax1.imshow(corr_original))
    # plt.colorbar(ax2.imshow(corr_avatars))

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # set title
    if title:
        plt.suptitle(title)
    
    return plt


def get_non_shared_var_projections(df: pd.DataFrame, linked_records: pd.DataFrame, columns1, columns2) -> float:
    # Fit the projection model (uses PCA, MCA or FAMD depending on type of data)

    should_consider_missing = True
    impute_method = ImputeMethod.MODE
    separate_models = False
    
    df = df.copy()
    linked_records = linked_records.copy()

    # Add (if required) columns indicating if a variable was missing
    if should_consider_missing:
        df = enrich_df_with_is_missing_variables(df)

    # Impute missing values (because PCA-like methods require no missing values)
    columns_with_missing = df.columns[df.isnull().any()].tolist()
    df = impute_missing_values(df, impute_method=impute_method)

    # Get number of modalities for each column to mitigate the impact of cardinality: for a
    # categorical variable, the sum of the contributions will be n_modalities * 100% 
    # (e.g. 3 modalities => 300%)
    n_modalities = df.nunique()
    n_modalities = n_modalities.to_dict()
    cat_columns: List[str] = df.select_dtypes(
        include=_CATEGORICAL_DTYPES,
    ).columns.to_list()
    n_modalities = {col: n_modalities[col] if col in cat_columns else 1 for col in df.columns}


# Add (if required) columns indicating if a variable was missing
    if should_consider_missing:
        linked_records = enrich_df_with_is_missing_variables(linked_records)

    # Impute missing values (because PCA-like methods require no missing values)
    columns_with_missing = linked_records.columns[linked_records.isnull().any()].tolist()
    linked_records = impute_missing_values(linked_records, impute_method=impute_method)

    # Get number of modalities for each column to mitigate the impact of cardinality: for a
    # categorical variable, the sum of the contributions will be n_modalities * 100% 
    # (e.g. 3 modalities => 300%)
    n_modalities = linked_records.nunique()
    n_modalities = n_modalities.to_dict()
    cat_columns: List[str] = linked_records.select_dtypes(
        include=_CATEGORICAL_DTYPES,
    ).columns.to_list()
    n_modalities = {col: n_modalities[col] if col in cat_columns else 1 for col in linked_records.columns}

    df_filtered = df[columns1+columns2]
    linked_records_filtered = linked_records[columns1+columns2]

    # print('df_filtered: ', df_filtered)
    model_ori = saiph.fit(
        df_filtered, nf=None
    )
    df_proj = saiph.transform(df_filtered, model_ori)

    if separate_models:
        model_linked = saiph.fit(
            linked_records_filtered, nf=None
        )
        linked_proj = saiph.transform(linked_records_filtered, model_linked)
    else:
        linked_proj = saiph.transform(linked_records_filtered, model_ori)

    return df_proj, linked_proj

# Function to convert values back to their original types
def convert_value(value, dtype):
    if pd.isna(value):
        return value
    if dtype == 'int64':
        return int(value)
    elif dtype == 'float64':
        return float(value)
    elif dtype == 'bool':
        return value.lower() in ['true', '1']
    elif dtype == 'object':
        return str(value)
    else:
        return value
    

def get_reconstruction_score(df: pd.DataFrame, linked_records: pd.DataFrame, nb_gower_batches = 20, number_of_model_components = 5):
    """Compute reconstruction score.

    The reconstruction score is the difference between the reconstruction error of the original 
    data and the reconstruction error of the linked data.
    Reconstruction error is computed by first fitting a model on the original data and by keeping
    the `number_of_model_components` first dimensions (the ones explaining most variance). 
    This model represents the original data. By projecting any data on this model and 
    reconstructing the data (i.e. inverse transform), we can compute the reconstruction error. 
    If the data has similar statistical properties to the original data, the reconstruction error
    will be close to the reconstruction error of the original data. On the other hand, if the data
    has different statistical properties, the reconstruction error will be higher. Measuring this 
    difference is a way to capture how globally similar the linked data is to the original data.
    This metric is multivariate since each model dimension potentially combines information from 
    all variables.

    Reconstruction error is measured by means of the Gower distance.

    Arguments
    ---------
        df: original data
        linked_records: linked data
        nb_gower_batches: number of batches to split the data into to compute gower distances. 
            Gower distance can be computationally expensive for large datasets. 
            Dividing the computation in batches makes the process faster and uses less memory while
            not impacting the results.
        number_of_model_components: number of model components to keep. 
            The higher the number, the smaller the differences will be. 
    """
    _dfs = [df, linked_records]
    _dfs = _handle_missing_values(_dfs)
    df = _dfs[0].reset_index(drop=True)
    linked_records = _dfs[1].reset_index(drop=True)

    # convert all values in categorical variables of linked_reconstructed to str. This is because
    # we may have different value types in the dataframes to compare following the inverse_transform
    original_dtypes  = df.dtypes

    for col in df.select_dtypes(include=['object_']).columns:
        df[col] = df[col].apply(lambda x: convert_value(x, original_dtypes[col]))

    for col in linked_records.select_dtypes(include=['object_']).columns:
        linked_records[col] = linked_records[col].apply(lambda x: convert_value(x, original_dtypes[col]))

    # fit saiph model on original data
    model_ori = saiph.fit(
        df, nf=number_of_model_components
    )

    # transform and inverse transform original data
    df_proj = saiph.transform(df, model_ori)
    df_reconstructed = saiph.inverse_transform(df_proj, model_ori)

    # using same model, transform and inverse transform linked data
    linked_proj = saiph.transform(linked_records, model_ori)

    linked_reconstructed = saiph.inverse_transform(linked_proj, model_ori)

    # convert all values in categorical variables of linked_reconstructed to str. This is because
    # we may have different value types in the dataframes to compare following the inverse_transform
    for col in df_reconstructed.select_dtypes(include=['object_']).columns:
        df_reconstructed[col] = df_reconstructed[col].apply(lambda x: convert_value(x, original_dtypes[col]))

    for col in linked_reconstructed.select_dtypes(include=['object_']).columns:
        linked_reconstructed[col] = linked_reconstructed[col].apply(lambda x: convert_value(x, original_dtypes[col]))


    all_diagonal_values = []
    all_diagonal_values_linked = []
    
    # raise error if index are not the same in df, df_reconstructed, linked_records, linked_reconstructed
    assert df.index.equals(df_reconstructed.index)
    assert df.index.equals(linked_records.index)
    assert df.index.equals(linked_reconstructed.index)


    for i in range(nb_gower_batches):
        # take the i-th portion of df
        all_ids = df.index.tolist()
        if i == nb_gower_batches - 1:
            ids_batch = all_ids[i * (len(all_ids) // nb_gower_batches):]
        else:
            ids_batch = all_ids[
                i * (len(all_ids) // nb_gower_batches): (i + 1) * (len(all_ids) // nb_gower_batches)
            ]

        # print(f'metric using gower for batch {i+1}/{nb_gower_batches}')
        df_batched = df[df.index.isin(ids_batch)]
        df_reconstructed_batched = df_reconstructed[df_reconstructed.index.isin(ids_batch)]
        distances_original  =_compute_gower_distances(df_batched, df_reconstructed_batched, list(df.columns))
        diagonal_values_original = np.diag(distances_original)  # Get diagonal values from distances
        all_diagonal_values.extend(diagonal_values_original)

        df_linked_batched = linked_records[linked_records.index.isin(ids_batch)]
        df_linked_reconstructed_batched = linked_reconstructed[linked_reconstructed.index.isin(ids_batch)]
        distances_linked  =_compute_gower_distances(df_linked_batched, df_linked_reconstructed_batched, list(df.columns))
        diagonal_values_linked = np.diag(distances_linked)  # Get diagonal values from distances
        all_diagonal_values_linked.extend(diagonal_values_linked)

    reconstruction_diff = [abs(x - y) for x, y in zip(all_diagonal_values, all_diagonal_values_linked)]

    reconstruction_diff_sum = sum(reconstruction_diff)
    reconstruction_diff_mean = np.mean(reconstruction_diff)
    reconstruction_diff_std = np.std(reconstruction_diff)
    reconstruction_diff_max = np.max(reconstruction_diff)

    return {
        'reconstruction_diff_sum': reconstruction_diff_sum,
        'reconstruction_diff_mean': reconstruction_diff_mean,
        'reconstruction_diff_std': reconstruction_diff_std,
        'reconstruction_diff_max': reconstruction_diff_max,
    }