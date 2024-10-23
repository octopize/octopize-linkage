from enum import Enum
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple, Union
from numpy.typing import NDArray
import gower
import saiph
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

from pre_linkage_metrics import _CATEGORICAL_DTYPES


class Distance(Enum):
    GOWER = "gower"
    PROJECTION_DIST_FIRST_SOURCE = "proj_eucl_first_source"
    PROJECTION_DIST_SECOND_SOURCE = "proj_eucl_second_source"
    PROJECTION_DIST_ALL_SOURCES = "proj_eucl_all_source"
    ROW_ORDER = "row_order"
    RANDOM = "random"


class LinkingAlgorithm(Enum):
    LSA = "lsa"
    MIN_ORDER = "min_order"
    MIN_REORDER = "min_reorder"


def link_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    shared_columns: list,
    distance: Distance,
    linking_algo: LinkingAlgorithm,
) -> pd.DataFrame:
    """Link two avatarized datasets.

    Arguments
    ---------
        df1: the first avatarized data frame to link
        df2: the second avatarized data frame to link
        shared_columns: the common variables between the datasets
        distance: the distance to be used by the linking algorithm
        linking_algo: the algorithm to use to match records based on the distance matrix

    Returns
    -------
        dataframe: a dataframe with the result of the linkage.
    """
    # Build distance matrix based on distance method
    linkage_distances: Dict[
        Distance, Callable[[pd.DataFrame, pd.DataFrame, List[str]], NDArray[np.float64]]
    ] = {
        Distance.GOWER: _compute_gower_distances,
        Distance.PROJECTION_DIST_FIRST_SOURCE: _compute_euclidean_firstsource_projection_distances,
        Distance.PROJECTION_DIST_SECOND_SOURCE: _compute_euclidean_secondsource_projection_distances,
        Distance.PROJECTION_DIST_ALL_SOURCES: _compute_euclidean_allsources_projection_distances,
        Distance.ROW_ORDER: _compute_row_order_distances,
        Distance.RANDOM: _compute_random_distances,
    }
    distances = linkage_distances[distance](df1, df2, shared_columns)

    # Correct distances to avoid zeros as needed by some solvers
    min_nz_val = distances[np.where(distances != 0)].min()
    distances[np.where(distances == 0)] = min_nz_val * 0.5

    # Solve the assignment problem
    linkage_algos: Dict[
        LinkingAlgorithm, Callable[[NDArray[np.float64]], Tuple[List[int], List[int]]]
    ] = {
        LinkingAlgorithm.LSA: _solve_linear_sum_assignment,
        # "min_weight_full_bipartite_matching": _solve_min_weight_full_bipartite_matching,
        LinkingAlgorithm.MIN_ORDER: _solve_min_order,
        LinkingAlgorithm.MIN_REORDER: _solve_min_reorder,
    }
    row_ind, col_ind = linkage_algos[linking_algo](distances)
    print("row_ind:", row_ind[:10])
    print("col_ind:", col_ind[:10])

    # Prepare columns to add to the df. Need to keep the column name order
    cols_to_add = filter(lambda col: col not in df1.columns, df2.columns)

    # Sort left df according to row_ind and adding missing indices at the end
    missing_indices = [i for i in range(len(df1)) if i not in row_ind]
    merged_df = df1.copy().reindex(row_ind + missing_indices).reset_index()

    # Sort right df according to col_ind
    right_df = df2[list(cols_to_add)].reindex(col_ind).reset_index(drop=True)

    # Merge the 2 dfs
    merged_df = merged_df.join(right_df)

    # Finally, re-sort according to original order
    merged_df = merged_df.set_index("index").sort_index()
    merged_df.index.name = None

    return merged_df


def _compute_row_order_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: Optional[List[str]] = None
) -> NDArray[np.float64]:
    distances = np.full((len(df1), len(df2)), 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def _compute_random_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: Optional[List[str]] = None
) -> NDArray[np.float64]:
    random_gen = np.random.default_rng()
    distances = random_gen.random((len(df1), len(df2)))
    return distances


def _compute_gower_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: List[str]
) -> NDArray[np.float64]:
    # convert input data to correct type and keep only linkage variables
    data_x: NDArray[Union[np.float64, np.str_]] = np.asarray(
        _convert_int_to_float(df1[linkage_var])
    )
    data_y: NDArray[Union[np.float64, np.str_]] = np.asarray(
        _convert_int_to_float(df2[linkage_var])
    )

    # Extract categorical column indices for gower - required to ensure categorical variables
    # with nan values are considered as categorical
    # _, categorical = split_column_types(df1[linkage_var])
    categorical: List[str] = df1[linkage_var].select_dtypes(
        include=_CATEGORICAL_DTYPES,
    ).columns.to_list()
    cat_features = [col in categorical for col in df1[linkage_var].columns]

    distances: NDArray[np.float64] = gower.gower_matrix(
        data_x=data_x, data_y=data_y, cat_features=cat_features
    )
    return distances


def _convert_int_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all numeric columns in dataframe to float.

    This is required to ensure gower distance is never calculated from data containing only int
    values. See issue:  https://github.com/wwwjk366/gower/issues/2
    """
    working = df.copy()
    numeric_columns = df.select_dtypes(exclude=_CATEGORICAL_DTYPES).columns.to_list()
    working[numeric_columns] = working[numeric_columns].astype(float)
    return working


def _compute_euclidean_firstsource_projection_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: List[str]
) -> NDArray[np.float64]:
    df_train = df1[linkage_var]
    to_transform_dfs = [df_train, df2[linkage_var]]
    return _compute_euclidean_projection_distances(df_train, to_transform_dfs)


def _compute_euclidean_secondsource_projection_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: List[str]
) -> NDArray[np.float64]:
    df_train = df2[linkage_var]
    to_transform_dfs = [df1[linkage_var], df_train]
    return _compute_euclidean_projection_distances(df_train, to_transform_dfs)


def _compute_euclidean_allsources_projection_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, linkage_var: List[str]
) -> NDArray[np.float64]:
    df_train = pd.concat([df1[linkage_var], df2[linkage_var]], ignore_index=True)
    to_transform_dfs = [df1[linkage_var], df2[linkage_var]]
    return _compute_euclidean_projection_distances(df_train, to_transform_dfs)


def _compute_euclidean_projection_distances(
    df_train: pd.DataFrame, to_transform_dfs: List[pd.DataFrame]
) -> NDArray[np.float64]:

    # fit PCA / FAMD
    model = saiph.fit(df_train)

    # for each source, project a sub df composed of the linkage_var only
    projections = []
    projections.append(saiph.transform(to_transform_dfs[0], model))
    projections.append(saiph.transform(to_transform_dfs[1], model))

    # compute pairwise euclidean distance between projections
    distances: NDArray[np.float64] = euclidean_distances(projections[0], projections[1])
    return distances


def _solve_linear_sum_assignment(
    distances: NDArray[np.float64],
) -> Tuple[List[int], List[int]]:
    row_ind, col_ind = linear_sum_assignment(distances)
    return list(row_ind), list(col_ind)


def _solve_min_order(distances: NDArray[np.float64]) -> Tuple[List[int], List[int]]:
    # Order once records in order of min distance to another record, then assign
    # to closest record, respecting this order (greedy)
    min_values = [(i, min(distances[i])) for i in range(len(distances))]
    # print('min_values:', min_values)
    min_values = sorted(min_values, key=lambda tup: (tup[1]))
    # print('min_values_sorted:', min_values)
    insertion_order = [x[0] for x in min_values]
    # print('insertion_order:', insertion_order)

    # row_ind = set()
    # col_ind = set()
    row_ind = []
    col_ind = []
    max_val = math.ceil(np.max(distances) + 1.0)
    for i in range(np.shape(distances)[1]):
        # print('---------------')
        # print(i, len(col_ind))
        row_ind.append(insertion_order[i])
        # print(insertion_order[i])

        # replace already drawn columns with max value
        col_ind_as_set = set(col_ind)
        tmp = [
            distances[insertion_order[i]][j] if (j not in col_ind_as_set) else max_val
            for j in range(len(distances[insertion_order[i]]))
        ]
        # print('tmp:', tmp)
        # draw column with minimum distance
        val = np.argmin(tmp)
        # print('val:', val)
        col_ind.append(val)
        # if i ==4:
        #     break
    return list(row_ind), list(col_ind)  # type: ignore


def _solve_min_reorder(distances: NDArray[np.float64]) -> Tuple[List[int], List[int]]:
    # Order records in order of min distance to another record, then assign to closest record,
    # respecting this order (greedy). the order is recomputed after each pairing
    row_ind = set()
    col_ind = set()
    max_val = math.ceil(np.max(distances) + 1.0)
    for _ in range(distances.shape[1]):
        remaining_indices = filter(lambda ii: ii not in row_ind, range(len(distances)))

        min_values = [
            (
                i,
                min(
                    [
                        distances[i][j] if (j not in col_ind) else max_val
                        for j in range(len(distances[i]))
                    ]
                ),
            )
            for i in remaining_indices
        ]

        min_values = sorted(min_values, key=lambda tup: (tup[1]))

        insertion_order = [x[0] for x in min_values]

        i = insertion_order[0]
        row_ind.add(i)
        tmp = [
            distances[i][j] if (j not in col_ind) else max_val
            for j in range(len(distances[i]))
        ]

        val = np.argmin(tmp)
        col_ind.add(val)
    return list(row_ind), list(col_ind)  # type: ignore