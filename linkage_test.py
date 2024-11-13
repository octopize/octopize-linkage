
from typing import Any, Dict, List
from linkage import Distance, _compute_euclidean_allsources_projection_distances, _compute_euclidean_firstsource_projection_distances, _compute_euclidean_secondsource_projection_distances, _compute_gower_distances, _compute_row_order_distances, LinkingAlgorithm, link_datasets
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import pytest


DF1 = pd.DataFrame(
    {
        "linkage_1": [1, 2, 3],
        "linkage_2": [3.0, 1.2, 4.5],
    }
)

DF2 = pd.DataFrame(
    {
        "linkage_1": [1, 2, 3],
        "linkage_2": [3.0, 1.2, 4.5],
    }
)

GOWER_RESULT: NDArray[np.float64] = np.array(
    [[0.0, 0.5227273, 0.72727275], [0.5227273, 0.0, 0.75], [0.72727275, 0.75, 0.0]],
    dtype="float32",
)

REF_EUCL_1_RESULT: NDArray[np.float64] = np.array(
    [
        [0.0, 1.8111377032737683, 2.6900304340776025],
        [1.8111377032737683, 0.0, 2.735601667552585],
        [2.6900304340776025, 2.7356016675525847, 0.0],
    ],
    dtype="float64",
)


EUCL_ALL_RESULT: NDArray[np.float64] = np.array(
    [
        [0.0, 1.8111377032737683, 2.6900304340776025],
        [1.8111377032737683, 0.0, 2.735601667552585],
        [2.6900304340776025, 2.735601667552585, 0.0],
    ],
    dtype="float64",
)

ROWORDER_RESULT: NDArray[np.float64] = np.array(
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
)

TEST_DATA: Dict[str, Dict[str, List[Any]]] = {
    "data_1": {
        "linkage_1": [1, 2, 3, 1, 2, 2, 2, 1],
        "linkage_2": [3.0, 1.2, 4.5, 7.4, 2.8, 3.1, 2.1, 0.5],
        "linkage_3": ["A", "A", "C", "B", "A", "D", "A", "B"],
        "source_1": [12, 14, 18, 8, 15, 13, 14, 9],
    },
    "data_2": {
        "linkage_1": [1, 2, 3, 1, 2, 2, 2, 1],
        "linkage_2": [3.0, 1.2, 4.5, 7.4, 2.8, 3.1, 2.1, 0.5],
        "linkage_3": ["A", "A", "C", "B", "A", "D", "A", "B"],
        "source_2": [9, 14, 14, 10, 13, 10, 19, 7],
    }
}

TEST_DF = [pd.DataFrame(v) for v in TEST_DATA.values()]

def test_gower_distance() -> None:
    val = _compute_gower_distances(
        DF1, DF2, linkage_var=["linkage_1", "linkage_2"]
    )
    assert np.array_equal(np.round(val, decimals=3), np.round(GOWER_RESULT, decimals=3))


def test_euclidean_firstsource_projection_distances() -> None:
    val = _compute_euclidean_firstsource_projection_distances(
        DF1, DF2, linkage_var=["linkage_1", "linkage_2"]
    )
    assert np.array_equal(np.round(val, decimals=3), np.round(REF_EUCL_1_RESULT, decimals=3))


def test_euclidean_allsources_projection_distances() -> None:
    val = _compute_euclidean_allsources_projection_distances(
        DF1, DF2, linkage_var=["linkage_1", "linkage_2"]
    )
    assert np.array_equal(np.round(val, decimals=3), np.round(EUCL_ALL_RESULT, decimals=3))


def test_row_order_distances() -> None:
    val = _compute_row_order_distances(DF1, DF2)
    assert np.array_equal(val, ROWORDER_RESULT)


def test_gower_distance_with_int_values_only() -> None:
    """The gower_matrix function crashes if passed only int values. Tests it does not occur."""
    df1 = pd.DataFrame({"v1": [0, 1], "v2": [1, 2]})
    df2 = pd.DataFrame({"v1": [1, 2], "v2": [1, 2]})
    val = _compute_gower_distances(df1, df2, linkage_var=["v1", "v2"])
    expected: NDArray[np.float64] = np.array([[0.25, 1.0], [1.0, 0.25]])
    assert np.array_equal(val, expected)


@pytest.mark.parametrize(
    "distance",
    [
        Distance.GOWER,
        Distance.PROJECTION_DIST_FIRST_SOURCE,
        Distance.PROJECTION_DIST_ALL_SOURCES,
        Distance.ROW_ORDER,
        Distance.RANDOM,
    ],
)
@pytest.mark.parametrize(
    "linking_algo",
    [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER, LinkingAlgorithm.MIN_REORDER],
)
@pytest.mark.parametrize("linkage_var", [["linkage_1", "linkage_2"]])
def test_linkage(
    distance: str, linking_algo: str, linkage_var: List[str]
) -> None:
    linked_df = link_datasets(
        TEST_DF[0], TEST_DF[1], shared_columns=linkage_var, distance=distance, linking_algo=linking_algo
    )
    assert len(linked_df) == 8  # check same number of row
    assert set(linked_df.columns) == {
        "linkage_1",
        "linkage_2",
        "linkage_3",
        "source_1",
        "source_2",
    }


def test_gower_distance_diag_zero() -> None:
    """The gower_matrix function crashes if passed only int values. Tests it does not occur."""
    df1 = pd.DataFrame({"v1": [0, 1, 4, 5, 6, 3, np.nan, 2, 1], "v2": [1, 2, 2, 4, 1, 3, 2, 2, 1]})
    val = _compute_gower_distances(df1, df1, linkage_var=["v1", "v2"])
    expected: NDArray[np.float64] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.array_equal(np.diag(val), expected)
