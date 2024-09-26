import numpy as np
import pytest
from pre_linkage_metrics import IS_MISSING_VARIABLES_PREFIX, ImputeMethod, contribution_score, enrich_df_with_is_missing_variables, impute_missing_values
import pandas as pd
from pandas.testing import assert_frame_equal

def test_contribution_score_is_one_with_all_columns():
    df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": [1, 2, 9],
        "c": [7, 8, 9]
    })
    shared_columns = ["a", "b", "c"]
    number_of_dimensions = 2
    result = contribution_score(df=df, shared_columns=shared_columns, number_of_dimensions=number_of_dimensions, seed=None)
    assert result["selected_columns_relative_contribution_score"] == 1


def test_contribution_score_is_split_when_no_structure():
    # all variables fully independent
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    shared_columns = ["a", "b"]
    number_of_dimensions = 3
    result = contribution_score(df=df, shared_columns=shared_columns, number_of_dimensions=number_of_dimensions, seed=None)
    for key in result["contribution_score_per_variable"]:
        assert round(result["contribution_score_per_variable"][key], 2) == 33.33


def test_contribution_score_with_target_variance():
    df = pd.DataFrame({
        f"var_{i}": np.random.uniform(0, 1, 15) for i in range(10)
    })
    shared_columns = ["var_0", "var_1"]
    target_explained_variance = 0.95
    result = contribution_score(df=df, shared_columns=shared_columns, target_explained_variance=target_explained_variance, seed=None)
    assert result['variance_explained_by_dimensions'] >= target_explained_variance


def test_contribution_score_with_target_n_dim():
    df = pd.DataFrame({
        f"var_{i}": np.random.uniform(0, 1, 15) for i in range(10)
    })
    shared_columns = ["var_0", "var_1"]
    n_dim=2  # this will override target_explained_variance of 0.95 by default
    result = contribution_score(df=df, shared_columns=shared_columns, number_of_dimensions=n_dim, seed=None)
    assert result['number_of_dimensions_used'] == n_dim


@pytest.mark.parametrize(
    "impute_method", [ImputeMethod.MODE, ImputeMethod.MEDIAN, ImputeMethod.KNN]
)
def test_missing_value_impute(impute_method: ImputeMethod):
    df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": [4, 3, 6],
        "c": ["a", np.nan, "c"]
    })
    df_imputed = impute_missing_values(df, impute_method = impute_method)
    assert df_imputed.isnull().sum().sum() == 0  # no missing values after impute
    assert "__nan__" in df_imputed["c"].values  # categorical missing imputed as new modality


@pytest.mark.parametrize(
    "impute_method", [ImputeMethod.MODE, ImputeMethod.MEDIAN, ImputeMethod.KNN]
)
def test_missing_value_impute_works_when_no_impute_needed(impute_method: ImputeMethod):
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 3, 6],
        "c": ["a", "b", "c"]
    })
    df_imputed = impute_missing_values(df, impute_method = impute_method)
    assert_frame_equal(df, df_imputed)


def test_enrich_df_with_is_missing_variables():
    df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": [4, 3, 6],
        "c": ["a", np.nan, "c"]
    })
    df_enriched = enrich_df_with_is_missing_variables(df)

    assert f"{IS_MISSING_VARIABLES_PREFIX}a" in df_enriched.columns
    assert f"{IS_MISSING_VARIABLES_PREFIX}b" not in df_enriched.columns    
    assert f"{IS_MISSING_VARIABLES_PREFIX}c" in df_enriched.columns
    assert df_enriched[f"{IS_MISSING_VARIABLES_PREFIX}a"].values.tolist() == [0, 0, 1]
    assert df_enriched[f"{IS_MISSING_VARIABLES_PREFIX}c"].values.tolist() == [0, 1, 0]


def test_contribution_score_with_missing_data():
    df = pd.DataFrame({
        f"var_{i}": np.random.uniform(0, 1, 15) for i in range(10)
    })
    # add missing values
    df.loc[0, "var_0"] = np.nan
    df.loc[1, "var_1"] = np.nan
    df.loc[2, "var_2"] = np.nan

    shared_columns = ["var_0", "var_1"]
    target_explained_variance = 0.95
    result = contribution_score(df=df, shared_columns=shared_columns, target_explained_variance=target_explained_variance, impute_method=ImputeMethod.KNN, should_consider_missing=True, seed=None)
    assert result['variance_explained_by_dimensions'] >= target_explained_variance
