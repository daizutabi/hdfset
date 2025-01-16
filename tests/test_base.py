from pathlib import Path

import pytest
from pandas import DataFrame, HDFStore, Series

from hdfset.base import BaseDataSet
from hdfset.data import DataSet


@pytest.fixture(scope="module")
def dataframes() -> list[DataFrame | None]:
    df1 = DataFrame({"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = DataFrame({"id": [1, 1, 2, 2, 3, 3], "x": range(10, 16), "y": range(20, 26)})
    return [df1, None, df2]


@pytest.fixture(scope="module")
def path(dataframes, tmp_path_factory):
    path = tmp_path_factory.mktemp("test") / "test.h5"
    BaseDataSet.to_hdf(path, dataframes)
    return path


@pytest.fixture(scope="module")
def store(path: Path):
    with HDFStore(path) as store:
        yield store


@pytest.fixture(params=[BaseDataSet, DataSet])
def dataset(path: Path, request: pytest.FixtureRequest):
    cls = request.param

    with cls(path) as dataset:
        yield dataset


def test_id(dataset: BaseDataSet):
    assert dataset.get_id_column() == "id"


def test_repr(dataset: BaseDataSet):
    assert repr(dataset).endswith("DataSet('test')>")


def test_str(dataset: BaseDataSet):
    assert str(dataset).endswith("DataSet((3, 6))")


def test_storers(dataset: BaseDataSet):
    from pandas.io.pytables import AppendableFrameTable  # type: ignore

    for storer in dataset.storers():
        assert isinstance(storer, AppendableFrameTable)


def test_len(dataset: BaseDataSet):
    assert dataset.store.keys() == ["/_0", "/_2"]
    assert len(dataset) == 2


def test_iter(dataset: BaseDataSet):
    assert list(dataset) == [["id", "a", "b"], ["id", "x", "y"]]


def test_columns(dataset: BaseDataSet):
    assert dataset.columns == ["id", "a", "b", "id", "x", "y"]


def test_contains(dataset: BaseDataSet):
    assert "id" in dataset
    assert "unknown" not in dataset


def test_length(dataset: BaseDataSet):
    assert dataset.length == (3, 6)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [("id", "/_0"), (["a", "b"], "/_0"), ("x", "/_2"), (["x", "y"], "/_2")],
)
def test_index(dataset: BaseDataSet, columns: str | list[str], expected: str):
    assert dataset.index(columns) == expected


def test_index_error(dataset: BaseDataSet):
    with pytest.raises(IndexError):
        dataset.index(["a", "x"])


def test_index_dict(dataset: BaseDataSet):
    a = dataset.get_index_dict(["a", "b", ("x", "y")])
    b = {"a": "/_0", "b": "/_0", "x": "/_2", "y": "/_2"}
    assert a == b


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"a": 1}, "a=1"),
        ({"a": 1, "b": 2}, "a=1 and b=2"),
        ({"a": [1, 2, 3]}, "a=[1, 2, 3]"),
        ({"a": (1, 2)}, "(a>=1 and a<=2)"),
        ({"a": (None, 2)}, "a<=2"),
        ({"a": (1, None)}, "a>=1"),
    ],
)
def test_query_string(where, expected):
    from hdfset.base import query_string

    assert query_string(where) == expected


@pytest.mark.parametrize("where", [None, {}])
def test_query_string_none(where):
    from hdfset.base import query_string

    assert query_string(where) == ""


@pytest.mark.parametrize(("index", "i"), [(0, 0), (2, 2), ("/_0", 0), ("/_2", 2)])
def test_select(dataset: BaseDataSet, dataframes: list[DataFrame], index, i):
    df = dataset.select(index)
    assert isinstance(df, DataFrame)
    assert df.equals(dataframes[i])


def test_select_columns_dataframe(dataset: BaseDataSet):
    df = dataset.select(0, columns=["a", "b"])
    assert isinstance(df, DataFrame)


def test_select_error(dataset: BaseDataSet):
    with pytest.raises(IndexError):
        dataset.select(1)


def test_get_series(dataset: BaseDataSet):
    s = dataset.get("a")
    assert isinstance(s, Series)
    assert s.to_list() == [4, 5, 6]


def test_get_frame(dataset: BaseDataSet):
    df = dataset.get(["a", "b"])
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].to_list() == [4, 5, 6]
    assert df["b"].to_list() == [7, 8, 9]


def test_get_merge(dataset: BaseDataSet):
    df = dataset.get(["a", "x"])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_get_tuple(dataset: BaseDataSet):
    df = dataset.get(["a", "b", ("x", "y")])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 4)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["b"].to_list() == [7, 7, 8, 8, 9, 9]
    assert df["x"].to_list() == list(range(10, 16))
    assert df["y"].to_list() == list(range(20, 26))


def test_get_where_value(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], a=4)
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].to_list() == [4, 4]
    assert df["b"].to_list() == [7, 7]
    assert df["x"].to_list() == [10, 11]
    assert df["y"].to_list() == [20, 21]


def test_get_where_list(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], a=[4, 6])
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].to_list() == [4, 4, 6, 6]
    assert df["b"].to_list() == [7, 7, 9, 9]
    assert df["x"].to_list() == [10, 11, 14, 15]
    assert df["y"].to_list() == [20, 21, 24, 25]


def test_get_where_tuple(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].to_list() == [5, 5, 6, 6]
    assert df["b"].to_list() == [8, 8, 9, 9]
    assert df["x"].to_list() == [12, 13, 14, 15]
    assert df["y"].to_list() == [22, 23, 24, 25]


def test_get_where_empty(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], a=4, b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (0, 4)


def test_get_where_tuple_none_first(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], x=(None, 12))
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 4)
    assert df["a"].to_list() == [4, 4, 5]
    assert df["b"].to_list() == [7, 7, 8]
    assert df["x"].to_list() == [10, 11, 12]
    assert df["y"].to_list() == [20, 21, 22]


def test_get_where_tuple_none_second(dataset: BaseDataSet):
    df = dataset.get(["a", "b", "x", "y"], y=(24, None))
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].to_list() == [6, 6]
    assert df["b"].to_list() == [9, 9]
    assert df["x"].to_list() == [14, 15]
    assert df["y"].to_list() == [24, 25]


def test_get_error(dataset: BaseDataSet):
    m = "No data was found."
    with pytest.raises(ValueError, match=m):
        dataset.get([])


def test_getitem_int(dataset: BaseDataSet):
    s = dataset[0]
    assert isinstance(s, Series)
    assert s.to_list() == [1, 2, 3]


def test_getitem_series(dataset: BaseDataSet):
    s = dataset["a"]
    assert isinstance(s, Series)
    assert s.to_list() == [4, 5, 6]


def test_getitem_frame(dataset: BaseDataSet):
    df = dataset[["a", "b"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].to_list() == [4, 5, 6]
    assert df["b"].to_list() == [7, 8, 9]


def test_getitem_merge(dataset: BaseDataSet):
    df = dataset[["a", "x"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_getitem_merge_tuple(dataset: BaseDataSet):
    df = dataset[["a", ("x",)]]
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_id_error(tmp_path):
    path = tmp_path / "test.h5"
    df1 = DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]})

    BaseDataSet.to_hdf(path, [df1, df2])

    with BaseDataSet(path) as dataset:
        assert dataset.id is None

        m = "The number of id columns is not equal to 1."
        with pytest.raises(ValueError, match=m):
            dataset.get_id_column()


def test_id_len_limit(tmp_path: Path):
    df1 = DataFrame({"id": range(2000), "a": range(2000)})
    df2 = DataFrame({"id": range(2000), "b": range(10000, 12000)})

    path = tmp_path / "test.h5"
    BaseDataSet.to_hdf(path, [df1, df2])

    with BaseDataSet(path) as dataset:
        df = dataset.get(["a", "b"], a=(0, 1500))
        assert df.shape == (1501, 2)
        assert df.iloc[-1].to_list() == [1500, 11500]
