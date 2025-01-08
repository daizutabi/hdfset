from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame, HDFStore, Series

from hdfset.data import DataSet


@pytest.fixture(scope="module")
def dataframes():
    df1 = DataFrame({"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = DataFrame({"id": [1, 1, 2, 2, 3, 3], "x": range(10, 16), "y": range(20, 26)})
    df3 = DataFrame({"id": [1, 1, 2, 2, 3, 3], "x": range(50, 56), "z": range(60, 66)})
    return [df1, df2, df3]


@pytest.fixture(scope="module")
def df(dataframes):
    return dataframes[1]


@pytest.mark.parametrize(
    ("value", "x"),
    [
        (1, [[1, 10, 20], [1, 11, 21]]),
        (2, [[2, 12, 22], [2, 13, 23]]),
        (3, [[3, 14, 24], [3, 15, 25]]),
        ([1, 3], [[1, 10, 20], [1, 11, 21], [3, 14, 24], [3, 15, 25]]),
        ((1, 2), [[1, 10, 20], [1, 11, 21], [2, 12, 22], [2, 13, 23]]),
    ],
)
def test_select_by_column(df: DataFrame, value, x):
    from hdfset.data import select_by_column

    df = select_by_column(df, "id", value)
    np.testing.assert_array_equal(df, x)


def test_select(df: DataFrame):
    from hdfset.data import select

    assert select(df).equals(df)


def test_select_columns(df: DataFrame):
    from hdfset.data import select

    assert select(df, columns=["id", "x"]).equals(df[["id", "x"]])


def test_select_str(df: DataFrame):
    from hdfset.data import select

    df = select(df, "id < 3", columns=["y"])
    np.testing.assert_array_equal(df, [[20], [21], [22], [23]])


def test_select_dict_list_list(df: DataFrame):
    from hdfset.data import select

    df = select(df, {"id": [1, 3], "x": [10, 13, 15]}, columns=["y"])
    np.testing.assert_array_equal(df, [[20], [25]])


def test_select_dict_tuple_list(df: DataFrame):
    from hdfset.data import select

    df = select(df, {"id": (1, 3), "x": [10, 13, 15]}, columns=["y"])
    np.testing.assert_array_equal(df, [[20], [23], [25]])


def test_select_dict_list_tuple(df: DataFrame):
    from hdfset.data import select

    df = select(df, {"id": [1, 3], "x": (10, 13)}, columns=["y"])
    np.testing.assert_array_equal(df, [[20], [21]])


@pytest.fixture(scope="module")
def path(dataframes, tmp_path_factory):
    path = tmp_path_factory.mktemp("test") / "test.h5"
    DataSet.to_hdf(path, dataframes)
    return path


@pytest.fixture(scope="module")
def store(path: Path):
    with HDFStore(path) as store:
        yield store


@pytest.fixture
def dataset(path: Path):
    df = DataFrame({"a": [4, 5, 6], "c": [14, 15, 16]})

    with DataSet(path) as dataset:
        dataset.merge(df)
        yield dataset


def test_df(dataset: DataSet):
    v = {"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9], "c": [14, 15, 16]}
    df = DataFrame(v)
    assert dataset.df.equals(df)


def test_id(dataset: DataSet):
    assert dataset.get_id_column() == "id"


def test_iter(dataset: DataSet):
    x = [["id", "a", "b", "c"], ["id", "x", "y"], ["id", "x", "z"]]
    assert list(dataset) == x


def test_columns(dataset: DataSet):
    x = ["id", "a", "b", "c", "id", "x", "y", "id", "x", "z"]
    assert dataset.columns == x


def test_length(dataset: DataSet):
    assert dataset.length == (3, 6, 6)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ("id", "/_0"),
        (("a", "b"), "/_0"),
        ("c", "/_0"),
        ("x", "/_1"),
        (("x", "y"), "/_1"),
        ("z", "/_2"),
        (("z", "x"), "/_2"),
    ],
)
def test_index(dataset: DataSet, columns, expected: str):
    assert dataset.index(columns) == expected


def test_index_error(dataset: DataSet):
    with pytest.raises(IndexError):
        dataset.index(["y", "z"])


@pytest.mark.parametrize(
    ("columns", "u", "v"),
    [(("x", "y"), "/_1", "/_1"), (("x", "z"), "/_2", "/_2")],
)
def test_index_dict(dataset: DataSet, columns, u, v):
    a = dataset.get_index_dict(["a", "b", columns])
    b = {"a": "/_0", "b": "/_0", columns[0]: u, columns[1]: v}
    assert a == b


@pytest.mark.parametrize(("index", "i"), [(1, 1), (2, 2), ("/_1", 1), ("/_2", 2)])
def test_dataset_select(dataset: DataSet, dataframes: list[DataFrame], index, i):
    df = dataset.select(index)
    assert isinstance(df, DataFrame)
    assert df.equals(dataframes[i])


@pytest.mark.parametrize("index", [0, "/_0"])
def test_dataset_select_zero(dataset: DataSet, index):
    df = dataset.select(index)
    np.testing.assert_array_equal(df["a"], [4, 5, 6])
    np.testing.assert_array_equal(df["c"], [14, 15, 16])


def test_dataset_select_int_error(dataset: DataSet):
    with pytest.raises(IndexError):
        dataset.select(3)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("id", [1, 2, 3]),
        ("a", [4, 5, 6]),
        ("c", [14, 15, 16]),
        ("x", range(10, 16)),
        ("y", range(20, 26)),
        ("z", range(60, 66)),
    ],
)
def test_get_series(dataset: DataSet, column, value):
    s = dataset.get(column)
    assert isinstance(s, Series)
    assert s.to_list() == list(value)


def test_get_frame(dataset: DataSet):
    df = dataset.get(["a", "c"])
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].to_list() == [4, 5, 6]
    assert df["c"].to_list() == [14, 15, 16]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("x", range(10, 16)),
        ("y", range(20, 26)),
        ("z", range(60, 66)),
    ],
)
def test_get_merge(dataset: DataSet, column, value):
    df = dataset.get(["c", column])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["c"].to_list() == [14, 14, 15, 15, 16, 16]
    assert df[column].to_list() == list(value)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("y", range(10, 16)),
        ("z", range(50, 56)),
    ],
)
def test_get_tuple(dataset: DataSet, column, value):
    df = dataset.get(["a", "c", ("x", column)])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 4)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["c"].to_list() == [14, 14, 15, 15, 16, 16]
    assert df["x"].to_list() == list(value)


@pytest.mark.parametrize(
    ("c", "column", "value"),
    [
        (14, "x", [10, 11]),
        (15, "y", [22, 23]),
        (16, "z", [64, 65]),
    ],
)
def test_get_where_value(dataset: DataSet, c, column, value):
    df = dataset.get(["c", column], c=c)
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 2)
    assert df["c"].to_list() == [c, c]
    assert df[column].to_list() == value


@pytest.mark.parametrize(
    ("c", "column", "value", "cvalue"),
    [
        (15, "y", [12, 13], [22, 23]),
        (16, "z", [54, 55], [64, 65]),
    ],
)
def test_get_where_value_tuple(dataset: DataSet, c, column, value, cvalue):
    df = dataset.get(["c", ("x", column)], c=c)
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 3)
    assert df["c"].to_list() == [c, c]
    assert df["x"].to_list() == value
    assert df[column].to_list() == cvalue


@pytest.mark.parametrize(
    ("c", "column", "value"),
    [
        ([14, 16], "x", [10, 11, 14, 15]),
        ([15, 16], "y", [22, 23, 24, 25]),
        ([14, 16], "z", [60, 61, 64, 65]),
    ],
)
def test_get_where_list(dataset: DataSet, c, column, value):
    df = dataset.get(["a", "c", column], c=c)
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 3)
    assert df["a"].to_list() == [c[0] - 10, c[0] - 10, c[1] - 10, c[1] - 10]
    assert df["c"].to_list() == [c[0], c[0], c[1], c[1]]
    assert df[column].to_list() == value


@pytest.mark.parametrize(
    ("c", "column", "value"),
    [
        ((14, 16), "x", range(10, 16)),
        ((15, 16), "y", range(22, 26)),
        ((14, 15), "z", range(60, 64)),
    ],
)
def test_get_where_tuple(dataset: DataSet, c, column, value):
    df = dataset.get(["c", column], c=c)
    assert isinstance(df, DataFrame)
    assert df[column].to_list() == list(value)


@pytest.mark.parametrize(
    ("c", "column", "value"),
    [
        ((15, 16), "y", range(12, 16)),
        ([14, 16], "y", [10, 11, 14, 15]),
        ([15, 16], "y", range(12, 16)),
        ((15, 16), "z", range(52, 56)),
        ([15, 16], "z", range(52, 56)),
        ([14, 15], "z", range(50, 54)),
        ((14, 16), "y", range(10, 16)),
        ([14, 15, 16], "z", range(50, 56)),
    ],
)
def test_get_where_tuple_column(dataset: DataSet, c, column, value):
    df = dataset.get(["c", ("x", column)], c=c)
    assert isinstance(df, DataFrame)
    assert df["x"].to_list() == list(value)


def test_get_where_empty(dataset: DataSet):
    df = dataset.get(["a", "c", "x", "y"], a=4, x=15)
    assert isinstance(df, DataFrame)
    assert df.shape == (0, 4)


@pytest.mark.parametrize("columns", [["a", "c", "x", "y"], ["a", "c", ("x", "y")]])
def test_get_where_tuple_xy(dataset: DataSet, columns):
    df = dataset.get(columns, x=(None, 12))
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 4)
    assert df["a"].to_list() == [4, 4, 5]
    assert df["c"].to_list() == [14, 14, 15]
    assert df["x"].to_list() == [10, 11, 12]
    assert df["y"].to_list() == [20, 21, 22]


def test_invalid_series(tmp_path: Path):
    s = Series([1, 2, 3])
    df = DataFrame([[1, 2, 3], [4, 5, 6]])

    path = tmp_path / "test.h5"
    s.to_hdf(path, key="/_0")
    df.to_hdf(path, key="/_1")

    with pytest.raises(TypeError):
        DataSet(path)
