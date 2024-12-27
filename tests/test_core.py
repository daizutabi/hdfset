from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame, HDFStore, Series

from hdfset import DataSet


def test_id_error(tmp_path):
    path = tmp_path / "test.h5"
    df1 = pd.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = pd.DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]})
    DataSet.to_hdf(path, [df1, df2])

    m = "The number of id columns is not equal to 1."
    with pytest.raises(ValueError, match=m):
        DataSet(path)


@pytest.fixture(scope="module")
def dataframes() -> list[DataFrame | None]:
    df1 = pd.DataFrame({"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3, 3], "x": range(10, 16), "y": range(20, 26)},
    )
    return [df1, None, df2]


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
    with DataSet(path) as dataset:
        yield dataset


def test_id(dataset: DataSet):
    assert dataset.id == "id"


def test_repr(dataset: DataSet):
    assert repr(dataset) == "<DataSet('test')>"


def test_storers(dataset: DataSet):
    from pandas.io.pytables import AppendableFrameTable  # type: ignore

    for storer in dataset.storers():
        assert isinstance(storer, AppendableFrameTable)


def test_len(dataset: DataSet):
    assert dataset.store.keys() == ["/_0", "/_2"]
    assert len(dataset) == 2


def test_iter(dataset: DataSet):
    assert list(dataset) == [["id", "a", "b"], ["id", "x", "y"]]


def test_columns(dataset: DataSet):
    assert dataset.columns == ["id", "a", "b", "id", "x", "y"]


def test_length(dataset: DataSet):
    assert dataset.length == (3, 6)


def test_str(dataset: DataSet):
    assert str(dataset) == "DataSet((3, 6))"


@pytest.mark.parametrize(
    ("columns", "expected"),
    [("id", "/_0"), (["a", "b"], "/_0"), ("x", "/_2"), (["x", "y"], "/_2")],
)
def test_index(dataset: DataSet, columns: str | list[str], expected: str):
    assert dataset.index(columns) == expected


def test_index_error(dataset: DataSet):
    with pytest.raises(IndexError):
        dataset.index(["a", "x"])


def test_index_dict(dataset: DataSet):
    a = dataset.get_index_dict(["a", "b", ("x", "y")])
    b = {"a": "/_0", "b": "/_0", "x": "/_2", "y": "/_2"}
    assert a == b


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"a": 1}, "a=1"),
        ({"a": 1, "b": 2}, "a=1 and b=2"),
        ({"a": [1, 2, 3]}, "a=[1, 2, 3]"),
        ({"a": (1, 2)}, "(a>=1 and a<=2)"),
        ({"a": (None, 2)}, "a<=2"),
        ({"a": (1, None)}, "a>=1"),
    ],
)
def test_query_string(kwargs, expected):
    from hdfset.core import query_string

    assert query_string(**kwargs) == expected


def test_select_int(dataset: DataSet, dataframes: list[DataFrame]):
    for i, df in enumerate(dataframes):
        ref = dataset.select(i)
        if df is None:
            assert ref is None
        else:
            assert isinstance(ref, DataFrame)
            assert ref.equals(df)


def test_select_str(dataset: DataSet, dataframes: list[DataFrame]):
    for i, df in enumerate(dataframes):
        if df is None:
            continue

        key = DataSet.key(i)
        ref = dataset.select(key)
        assert isinstance(ref, DataFrame)
        assert ref.equals(df)


def test_get_series(dataset: DataSet):
    s = dataset.get("a")
    assert isinstance(s, Series)
    assert s.tolist() == [4, 5, 6]


def test_get_frame(dataset: DataSet):
    df = dataset.get(["a", "b"])
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [4, 5, 6]
    assert df["b"].tolist() == [7, 8, 9]


def test_get_merge(dataset: DataSet):
    df = dataset.get(["a", "x"])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].tolist() == [4, 4, 5, 5, 6, 6]
    assert df["x"].tolist() == list(range(10, 16))


def test_get_tuple(dataset: DataSet):
    df = dataset.get(["a", "b", ("x", "y")])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 4)
    assert df["a"].tolist() == [4, 4, 5, 5, 6, 6]
    assert df["b"].tolist() == [7, 7, 8, 8, 9, 9]
    assert df["x"].tolist() == list(range(10, 16))
    assert df["y"].tolist() == list(range(20, 26))


def test_get_where_value(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], a=4)
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].tolist() == [4, 4]
    assert df["b"].tolist() == [7, 7]
    assert df["x"].tolist() == [10, 11]
    assert df["y"].tolist() == [20, 21]


def test_get_where_list(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], a=[4, 6])
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].tolist() == [4, 4, 6, 6]
    assert df["b"].tolist() == [7, 7, 9, 9]
    assert df["x"].tolist() == [10, 11, 14, 15]
    assert df["y"].tolist() == [20, 21, 24, 25]


def test_get_where_tuple(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].tolist() == [5, 5, 6, 6]
    assert df["b"].tolist() == [8, 8, 9, 9]
    assert df["x"].tolist() == [12, 13, 14, 15]
    assert df["y"].tolist() == [22, 23, 24, 25]


def test_get_where_empty(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], a=4, b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (0, 4)


def test_get_where_tuple_none_first(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], x=(None, 12))
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 4)
    assert df["a"].tolist() == [4, 4, 5]
    assert df["b"].tolist() == [7, 7, 8]
    assert df["x"].tolist() == [10, 11, 12]
    assert df["y"].tolist() == [20, 21, 22]


def test_get_where_tuple_none_second(dataset: DataSet):
    df = dataset.get(["a", "b", "x", "y"], y=(24, None))
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].tolist() == [6, 6]
    assert df["b"].tolist() == [9, 9]
    assert df["x"].tolist() == [14, 15]
    assert df["y"].tolist() == [24, 25]


def test_get_error(dataset: DataSet):
    m = "No data was found."
    with pytest.raises(ValueError, match=m):
        dataset.get([])


def test_getitem_int(dataset: DataSet):
    s = dataset[0]
    assert isinstance(s, Series)
    assert s.tolist() == [1, 2, 3]


def test_getitem_series(dataset: DataSet):
    s = dataset["a"]
    assert isinstance(s, Series)
    assert s.tolist() == [4, 5, 6]


def test_getitem_frame(dataset: DataSet):
    df = dataset[["a", "b"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [4, 5, 6]
    assert df["b"].tolist() == [7, 8, 9]


def test_getitem_merge(dataset: DataSet):
    df = dataset[["a", "x"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].tolist() == [4, 4, 5, 5, 6, 6]
    assert df["x"].tolist() == list(range(10, 16))
