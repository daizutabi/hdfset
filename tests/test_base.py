from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pandas import DataFrame, HDFStore, Series

from hdfset.base import BaseDataset, query_string
from hdfset.dataset import Dataset

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def dataframes() -> list[DataFrame | None]:
    df1 = DataFrame({"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = DataFrame({"id": [1, 1, 2, 2, 3, 3], "x": range(10, 16), "y": range(20, 26)})
    return [df1, None, df2]


@pytest.fixture(scope="module")
def path(dataframes: list[DataFrame | None], tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("test") / "test.h5"
    BaseDataset.to_hdf(path, dataframes)
    return path


@pytest.fixture(scope="module")
def store(path: Path):
    with HDFStore(path) as store:
        yield store


@pytest.fixture(params=[BaseDataset, Dataset])
def dataset(path: Path, request: pytest.FixtureRequest):
    cls = request.param

    with cls(path) as dataset:
        yield dataset


def test_id(dataset: BaseDataset):
    assert dataset.get_id_column() == "id"


def test_repr(dataset: BaseDataset):
    assert repr(dataset).endswith("Dataset('test')>")


def test_str(dataset: BaseDataset):
    assert str(dataset).endswith("Dataset((3, 6))")


def test_storers(dataset: BaseDataset):
    for storer in dataset.storers():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert storer.__class__.__name__ == "AppendableFrameTable"  # pyright: ignore[reportUnknownMemberType]


def test_len(dataset: BaseDataset):
    assert dataset.store.keys() == ["/_0", "/_2"]
    assert len(dataset) == 2


def test_iter(dataset: BaseDataset):
    assert list(dataset) == [["id", "a", "b"], ["id", "x", "y"]]


def test_columns(dataset: BaseDataset):
    assert dataset.columns == ["id", "a", "b", "id", "x", "y"]


def test_contains(dataset: BaseDataset):
    assert "id" in dataset
    assert "unknown" not in dataset


def test_length(dataset: BaseDataset):
    assert dataset.length == (3, 6)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [("id", "/_0"), (["a", "b"], "/_0"), ("x", "/_2"), (["x", "y"], "/_2")],
)
def test_index(dataset: BaseDataset, columns: str | list[str], expected: str):
    assert dataset.index(columns) == expected


def test_index_error(dataset: BaseDataset):
    with pytest.raises(IndexError):
        dataset.index(["a", "x"])


def test_index_dict(dataset: BaseDataset):
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
def test_query_string(where: dict[str, Any], expected: str):
    assert query_string(where) == expected


@pytest.mark.parametrize("where", [None, {}])
def test_query_string_none(where: dict[str, Any] | None):
    assert not query_string(where)


@pytest.mark.parametrize(("index", "i"), [(0, 0), (2, 2), ("/_0", 0), ("/_2", 2)])
def test_select(
    dataset: BaseDataset,
    dataframes: list[DataFrame],
    index: str | int,
    i: int,
):
    df = dataset.select(index)
    assert isinstance(df, DataFrame)
    assert df.equals(dataframes[i])


def test_select_columns_dataframe(dataset: BaseDataset):
    df = dataset.select(0, columns=["a", "b"])
    assert isinstance(df, DataFrame)


def test_select_error(dataset: BaseDataset):
    with pytest.raises(IndexError):
        dataset.select(1)


def test_get_series(dataset: BaseDataset):
    s = dataset.get("a")
    assert isinstance(s, Series)
    assert s.to_list() == [4, 5, 6]


def test_get_frame(dataset: BaseDataset):
    df = dataset.get(["a", "b"])
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].to_list() == [4, 5, 6]
    assert df["b"].to_list() == [7, 8, 9]


def test_get_merge(dataset: BaseDataset):
    df = dataset.get(["a", "x"])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_get_tuple(dataset: BaseDataset):
    df = dataset.get(["a", "b", ("x", "y")])
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 4)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["b"].to_list() == [7, 7, 8, 8, 9, 9]
    assert df["x"].to_list() == list(range(10, 16))
    assert df["y"].to_list() == list(range(20, 26))


def test_get_where_value(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], a=4)
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].to_list() == [4, 4]
    assert df["b"].to_list() == [7, 7]
    assert df["x"].to_list() == [10, 11]
    assert df["y"].to_list() == [20, 21]


def test_get_where_list(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], a=[4, 6])
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].to_list() == [4, 4, 6, 6]
    assert df["b"].to_list() == [7, 7, 9, 9]
    assert df["x"].to_list() == [10, 11, 14, 15]
    assert df["y"].to_list() == [20, 21, 24, 25]


def test_get_where_tuple(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 4)
    assert df["a"].to_list() == [5, 5, 6, 6]
    assert df["b"].to_list() == [8, 8, 9, 9]
    assert df["x"].to_list() == [12, 13, 14, 15]
    assert df["y"].to_list() == [22, 23, 24, 25]


def test_get_where_empty(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], a=4, b=(8, 9))
    assert isinstance(df, DataFrame)
    assert df.shape == (0, 4)


def test_get_where_tuple_none_first(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], x=(None, 12))
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 4)
    assert df["a"].to_list() == [4, 4, 5]
    assert df["b"].to_list() == [7, 7, 8]
    assert df["x"].to_list() == [10, 11, 12]
    assert df["y"].to_list() == [20, 21, 22]


def test_get_where_tuple_none_second(dataset: BaseDataset):
    df = dataset.get(["a", "b", "x", "y"], y=(24, None))
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 4)
    assert df["a"].to_list() == [6, 6]
    assert df["b"].to_list() == [9, 9]
    assert df["x"].to_list() == [14, 15]
    assert df["y"].to_list() == [24, 25]


def test_get_error(dataset: BaseDataset):
    m = "No data was found."
    with pytest.raises(ValueError, match=m):
        dataset.get([])


def test_getitem_int(dataset: BaseDataset):
    s = dataset[0]
    assert isinstance(s, Series)
    assert s.to_list() == [1, 2, 3]


def test_getitem_series(dataset: BaseDataset):
    s = dataset["a"]
    assert isinstance(s, Series)
    assert s.to_list() == [4, 5, 6]


def test_getitem_frame(dataset: BaseDataset):
    df = dataset[["a", "b"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].to_list() == [4, 5, 6]
    assert df["b"].to_list() == [7, 8, 9]


def test_getitem_merge(dataset: BaseDataset):
    df = dataset[["a", "x"]]
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_getitem_merge_tuple(dataset: BaseDataset):
    df = dataset[["a", ("x",)]]
    assert isinstance(df, DataFrame)
    assert df.shape == (6, 2)
    assert df["a"].to_list() == [4, 4, 5, 5, 6, 6]
    assert df["x"].to_list() == list(range(10, 16))


def test_id_error(tmp_path: Path):
    path = tmp_path / "test.h5"
    df1 = DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]})

    BaseDataset.to_hdf(path, [df1, df2])

    with BaseDataset(path) as dataset:
        assert dataset.id is None

        m = "The number of id columns is not equal to 1."
        with pytest.raises(ValueError, match=m):
            dataset.get_id_column()


def test_id_len_limit(tmp_path: Path):
    df1 = DataFrame({"id": range(2000), "a": range(2000)})
    df2 = DataFrame({"id": range(2000), "b": range(10000, 12000)})

    path = tmp_path / "test.h5"
    BaseDataset.to_hdf(path, [df1, df2])

    with BaseDataset(path) as dataset:
        df = dataset.get(["a", "b"], a=(0, 1500))
        assert df.shape == (1501, 2)
        assert df.iloc[-1].to_list() == [1500, 11500]
