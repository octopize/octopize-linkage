from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import saiph 
import seaborn as sns

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

    # share legend


    # add legend
    # plt.colorbar(ax1.imshow(corr_original))
    # plt.colorbar(ax2.imshow(corr_avatars))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # set title
    if title:
        plt.suptitle(title)


    # plt.show()
    
    return plt

def get_correlations(
    df: pd.DataFrame, linked_records: pd.DataFrame, columns1, columns2
) -> float:
    """Compute correlation retention evaluation.

    Arguments
    ---------
        records: original data
        avatars: avatarized data
    """
    # Concate to get the same dummy matrix
    # full_set = pd.concat([split_records, avatars])

    split_records_0 = df[columns1]
    split_records_1 = df[columns2]

    # split_records_0 = split_records_0.replace(np.nan, "__nan__")
    # split_records_1 = split_records_1.replace(np.nan, "__nan__")
    # linked_records = linked_records.replace(np.nan, "__nan__")

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



    # Create array to easily separate original from avatars after transformation
    # x: NDArray[np.str_] = np.array(["Original", "Avatar"])
    # label = np.repeat(x, [len(split_records), len(avatars)], axis=0)
    # dummies = pd.get_dummies(full_set)
    # dummies["Type"] = label
    # print("dummies done")

    # # Separate back records from avatars and drop Type column
    # records_dummies = dummies[dummies["Type"] == "Original"]
    # avatars_dummies = dummies[dummies["Type"] == "Avatar"]
    # records_dummies = records_dummies.drop(["Type"], axis=1)
    # avatars_dummies = avatars_dummies.drop(["Type"], axis=1)
    # print("separation done")

    # # Get column order for later
    # columns_names = records_dummies.columns.values

    # Get difference of correlation matrix between original and avatars
    # print(records_dummies.isna().sum())
    # print(avatars_dummies.isna().sum())


    # print('split_record_dummies_0: ', split_record_dummies_0.columns)
    # print('linked_records_dummies_0: ', linked_records_dummies_0.columns)

    # print('split_record_dummies_1: ', split_record_dummies_1.columns)
    # print('linked_records_dummies_1: ', linked_records_dummies_1.columns)

    corr_records = pd.concat([split_record_dummies_0, split_record_dummies_1], axis=1, keys=['split_record_dummies_0', 'split_record_dummies_1']).corr().loc['split_record_dummies_1', 'split_record_dummies_0']
    corr_avatars = pd.concat([linked_records_dummies_0, linked_records_dummies_1], axis=1, keys=['linked_records_dummies_0', 'linked_records_dummies_1']).corr().loc['linked_records_dummies_1', 'linked_records_dummies_0']

    # corr_records = split_record_dummies_0.corrwith(split_record_dummies_1)
    # corr_avatars = linked_records_dummies_0.corrwith(linked_records_dummies_1)
    corr_diff = corr_records - corr_avatars

    # print('corr_records:', corr_records)
    # print("corr done")
    # print('corr_diff:', corr_diff)
    # print('corr_diff.sum().sum():', corr_diff.sum().sum())

    # # Replace values from diag and upper triangle by zeros and get absolute values from others
    # diff = abs(
    #     (corr_diff.mask(np.tril(np.ones(corr_diff.shape)).astype(np.bool_))).fillna(0)
    # )

    # # Get sum of difference
    # diff_value = np.matrix(diff).sum()

    return corr_records, corr_avatars, corr_diff


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

    # share legend


    # add legend
    # plt.colorbar(ax1.imshow(corr_original))
    # plt.colorbar(ax2.imshow(corr_avatars))

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # set title
    if title:
        plt.suptitle(title)


    # plt.show()
    
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