import pandas as pd
import numpy as np
from typing import Dict, List, Optional

import saiph
from sklearn.impute import KNNImputer

_CATEGORICAL_DTYPES = [
    "object",
    "category",
    "boolean",
]


class ImputeMethod:
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"


IS_MISSING_VARIABLES_PREFIX = "is_missing_**_"

def contribution_score(*, df: pd.DataFrame, shared_columns: list, target_explained_variance: float = 0.95, number_of_dimensions: Optional[int] = None, impute_method: ImputeMethod = ImputeMethod.MODE, should_consider_missing: bool = False, seed: Optional[int] = None) -> Dict[str, float]:
    """Compute score based on projection contribution of each shared columns.

    The score is calculated from 
        - the projection-based contribution of each shared columns
        - the variance explained by each projection dimension
    
    Args:
        df (pd.DataFrame): 
            data (from one source)
        shared_columns (list): 
            list of columns that are common with the other source
        target_explained_variance (float):
            Expected variance to use to determine the number of dimensions. Default to 0.95.
        compute_contributions_nf (int): 
            number of dimensions to compute contributions from. 
            This overrides target_explained_variance if set. Defaults to None.
        impute_method (ImputeMethod):
            Method to use to impute missing values for the score calculation.
            Default to ImputeMethod.MODE
        should_consider_missing (bool):
            Whether to consider missing values as a separate variable and take it into account in
            score computation. Including it in the score will split the contribution of the variable
            between the variable itself and the missing value indicator. Default to False.
        seed (int, optional): 
            random seed. Defaults to None.

    Returns:
            Dict containing the following:
            - selected_columns_contribution_score:
                Non-bounded contribution score
            -selected_columns_relative_contribution_score:
                Bounded between 0 (selected columns do not represent the dataset) 
                and 1 (selected columns represent fully the dataset).
            - contribution_score_per_variable:
                Dictionary containing for each variable the contribution score.

    """
    df = df.copy()

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

    # Fit the projection model (uses PCA, MCA or FAMD depending on type of data)
    model = saiph.fit(
        df, nf=None, seed=seed
    )

   # Set number of dimensions based on explained variance or specified number of dimensions
    n_dim = 0
    if number_of_dimensions is not None:
        n_dim = number_of_dimensions
    else:
        explained_var_ratio = model.explained_var_ratio
        for i in range(len(explained_var_ratio)):
            if sum(explained_var_ratio[:i+1]) >= target_explained_variance:
                n_dim = i+1
                break

    # Retrieve contributions of each column to the projection
    contributions: pd.DataFrame = saiph.projection.get_variable_contributions(
        model, df
    )

    # keep only the n_dim most important dimensions in contributions and explained_var_ratio
    contributions = contributions.iloc[:, :n_dim]
    explained_var_ratio = model.explained_var_ratio[:n_dim]

    # Compute the contribution score for each variable, considering the number of modalities and the variance explained by the dimensions
    contribution_score_all_vars = {var: np.sum(contributions.loc[var] / n_modalities[var] * explained_var_ratio) for var in df.columns}

    # Merge contribution of missing values with the contribution of the variable itself if required
    if should_consider_missing:
        for col in columns_with_missing:
            contribution_score_all_vars[col] = (contribution_score_all_vars[col] + contribution_score_all_vars[f"{IS_MISSING_VARIABLES_PREFIX}{col}"])/ 2
    
    # Compute the contribution score for the set of selected columns
    selected_columns_contribution_score = np.sum([contribution_score_all_vars[col] for col in shared_columns])

    # Compute the relative contribution score for the set of selected columns (relative to the contribution score of the set of all variables)
    selected_columns_relative_contribution_score = selected_columns_contribution_score / np.sum([contribution_score_all_vars[col] for col in df.columns])
    
    return {
        "selected_columns_contribution_score": selected_columns_contribution_score,
        "selected_columns_relative_contribution_score": selected_columns_relative_contribution_score,
        "contribution_score_per_variable": contribution_score_all_vars,
        "number_of_dimensions_used": n_dim,
        "variance_explained_by_dimensions": sum(explained_var_ratio),
    }

    

def impute_missing_values(df: pd.DataFrame, impute_method: ImputeMethod, k_impute: int = 3) -> pd.DataFrame:
    """Impute missing values in the dataframe.

    Args:
        df (pd.DataFrame): 
            data to potentially impute

    Returns:
        pd.DataFrame: 
            data with missing values imputed
    """
    original_columns_order = df.columns.to_list()
    original_dtypes = df.dtypes

    cat_columns: List[str] = df.select_dtypes(
        include=_CATEGORICAL_DTYPES,
    ).columns.to_list()
    num_columns = df.select_dtypes(exclude=_CATEGORICAL_DTYPES).columns.to_list()

    # replace missing values with __nan__ for categorical variables only
    df[cat_columns] = df[cat_columns].replace(np.nan, "__nan__")

    # fit imputer for numerical variables
    if impute_method == ImputeMethod.MEDIAN:
        new_val = getattr(df[num_columns].dropna(), impute_method)()
        df[num_columns] = df[num_columns].fillna(new_val)
    elif impute_method == ImputeMethod.MODE:
        new_val = getattr(df[num_columns].dropna(), impute_method)().iloc[0]
        df[num_columns] = df[num_columns].fillna(new_val)
    elif impute_method == ImputeMethod.KNN:
        training_sample = df[num_columns]
        df = pd.concat([
            pd.DataFrame(
            KNNImputer(n_neighbors=k_impute).fit(training_sample).transform(df[num_columns]),
            columns=df[num_columns].columns.values,
        ),
        df[cat_columns]
        ], axis=1)

        df = df[original_columns_order]

        # set dtypes back to original
        df = df.astype(original_dtypes)
    
    return df


def enrich_df_with_is_missing_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with columns indicating if a variable was missing.

    Note that this step will add an extra column for continuous variables with missing values but also 
    for categorical ones with missing values.
    This is to consider the missing value information in the same way between the two types of 
    variables and not favour one or the other in the scoring function.

    Args:
        df (pd.DataFrame): 
            data to enrich

    Returns:
        pd.DataFrame: 
            data with additional columns indicating if a variable was missing
    """
    df_enriched =  df.assign(**{f"{IS_MISSING_VARIABLES_PREFIX}{col}": df[col].isnull() for col in df.columns})
    
    # get variables from df with no missing
    df_no_missing = df.dropna(axis=1).columns.to_list()

    # remove the corresponding columns added to df_enriched
    columns_to_remove = [f"{IS_MISSING_VARIABLES_PREFIX}{col}" for col in df_no_missing]
    df_enriched = df_enriched.drop(columns=columns_to_remove)

    return df_enriched