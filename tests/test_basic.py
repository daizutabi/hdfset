from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame, HDFStore

from hdfset import DataSet


@pytest.fixture
def dataframes() -> list[DataFrame]:
    df1 = pd.DataFrame({"id": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})
    df2 = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3, 3], "x": range(10, 16), "y": range(20, 26)},
    )
    return [df1, df2]


@pytest.fixture
def path(dataframes, tmp_path):
    path = tmp_path / "test.h5"
    DataSet.to_hdf(path, dataframes)
    return path


@pytest.fixture
def store(path: Path):
    with HDFStore(path) as store:
        yield store


def test_path(path: Path):
    assert Path(path).exists()


@pytest.fixture
def dataset(path: Path):
    with DataSet(path) as dataset:
        yield dataset


def test_id(dataset: DataSet):
    assert dataset.id == "id"


def test_keys(dataset: DataSet):
    assert dataset.keys == ["/_0", "/_1"]


def test_iter(dataset: DataSet):
    assert list(dataset) == [["id", "a", "b"], ["id", "x", "y"]]


def test_index(dataset: DataSet):
    assert dataset.index(["id", "a"]) == "/_0"
    assert dataset.index(["x", "y"]) == "/_1"
