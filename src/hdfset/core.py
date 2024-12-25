from __future__ import annotations

import os
import re
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pandas as pd
from pandas import DataFrame, HDFStore

from hdfset.common import select
from hdfset.utils import get_id_column


class DataSet:
    path: Path
    store: HDFStore
    keys: list[str]
    id: str

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.store = HDFStore(self.path, mode="r")
        self.keys = self.store.keys()
        self.id = get_id_column(self.store)

    @staticmethod
    def to_hdf(path: str | Path, dataframes: list[DataFrame]) -> None:
        """Save a list of DataFrames to an HDF5 file.

        Args:
            path (str or Path): The file path where the data will be saved.
            dataframes (list of DataFrame): A list of DataFrames to be saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for k, df in enumerate(dataframes):
            df.to_hdf(
                path,
                key=f"_{k}",
                complevel=9,
                complib="blosc",
                format="table",
                data_columns=True,
            )

    # def __str__(self):
    #     return f"DataSet({self.length})"

    # def __repr__(self):
    #     basename = os.path.basename(self.path).replace(".h5", "")
    #     return f"<DataSet({basename})>"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.store.close()

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[list[str]]:
        for key in self.store:
            df = self.store.select(key, start=0, stop=0)
            yield df.columns.tolist()

    def index(self, columns: str | list[str]) -> str:
        if isinstance(columns, str):
            columns = [columns]

        for key, columns_ in zip(self.store, self, strict=True):
            if all(column in columns_ for column in columns):
                return key

        raise IndexError("The specified columns were not found.")

    #     def clone(self, df=None):
    #         """
    #         クローンを作成し，新しいDataSetインスタンスとして返す．

    #         Parameters
    #         ----------
    #         df : pd.DataFrame, optional
    #             dfを直接指定する．一部を抽出した場合に用いる．

    #         Returns
    #         -------
    #         DataSet
    #         """
    #         data = self.__class__(self.store)
    #         data.keys = self.keys
    #         data.df = df if isinstance(df, pd.DataFrame) else self.df.copy()
    #         data.id = self.id
    #         data.selected_ids = self.selected_ids
    #         return data


#     def __getitem__(self, index):
#         """
#         データの取得．

#         data[int]
#         data[column]
#         data[[column1, column2, ...]]

#         Parameters
#         ----------
#         index : int or list or tuple
#             インデックス指定
#         """
#         if isinstance(index, int):
#             return self.get(self.columns[index])
#         if isinstance(index, str) or isinstance(index, list):
#             return self.get(index)
#         if isinstance(index, tuple):
#             return self.get(index[0], **index[1])
#         raise IndexError

#     def get(self, columns, **kwargs):
#         """
#         カラム指定に応じて，複数のDataFrameから必要なデータを抽出し，
#         マージする．

#         Parameters
#         ----------
#         columns : str or list
#             データ選択リスト
#             複数のDataFrameにまたがってデータを取得する．
#             同一のDataFrameからまとめてデータを取得したい場合には
#             タプルでくくる． ['x', 'y', ('a', 'b')]
#         """
#         # カラムがどのindexに属するか探索
#         # カラム名からindexを求める辞書を作成．

#         if isinstance(columns, str):
#             return self.get([columns], **kwargs)[columns]

#         def get_index_dict(values):
#             index_dict = OrderedDict()
#             for value in values:
#                 index_ = self.index(value)
#                 if isinstance(value, tuple):
#                     for value_ in value:
#                         index_dict[value_] = index_
#                 else:
#                     index_dict[value] = index_
#             return index_dict

#         column_indexes = get_index_dict(columns)
#         kwarg_indexes = get_index_dict(kwargs.keys())
#         indexes = set(list(column_indexes.values()) + list(kwarg_indexes.values()))

#         df = None
#         for index in indexes:
#             subcolumns = [
#                 column
#                 for column in column_indexes.keys()
#                 if column_indexes[column] == index
#             ]
#             subkwargs = {
#                 key: value
#                 for key, value in kwargs.items()
#                 if kwarg_indexes[key] == index
#             }
#             if self.id not in subcolumns:
#                 subcolumns = [self.id] + subcolumns
#             if index == 0:
#                 sub = self.df
#                 if self.selected_ids is not None:
#                     sub = sub[sub[self.id].isin(self.selected_ids)]
#                 sub = select(sub, subkwargs)
#                 sub = sub[subcolumns]
#             else:
#                 if self.selected_ids is not None:
#                     query = {self.id: self.selected_ids}
#                 else:
#                     query = {}
#                 query.update(subkwargs)

#                 if query:
#                     query = query_hdf(subcolumns, **query)
#                     try:
#                         sub = self.store.select(index, query)
#                     except Exception:  # どんな例外が発生するか？
#                         sub = self.store.select(index)
#                         if self.selected_ids is not None:
#                             sub = sub[sub[self.id].isin(self.selected_ids)]
#                         sub = select(sub, subkwargs)
#                 else:
#                     sub = self.store.select(index)
#             how = "inner" if kwargs else "left"
#             df = sub if df is None else df.merge(sub, how=how)

#         columns = list(OrderedDict.fromkeys(flatten(columns)))
#         return df[columns]


#     @property
#     def length(self):
#         list_ = []
#         for key in self.keys:
#             list_.append(self.store.get_storer(key).nrows)
#         return tuple(list_)

#     @property
#     def shape(self):
#         list_ = []
#         for key in self.keys:
#             storer = self.store.get_storer(key)
#             list_.append((storer.nrows, len(storer.data_columns)))
#         return tuple(list_)

#     @property
#     def columns(self):
#         columns_ = [list(self.df.columns)]
#         for key in self.keys[1:]:
#             storer = self.store.get_storer(key)
#             columns_.append(storer.data_columns)
#         return columns_


#     def select(self, **kwargs):
#         """
#         dataを選択して新しいDataSetインスタンスを返す．

#         Parameters
#         ----------
#         **kwargs : dict
#             キーがカラム名．値が選択値
#             see also: spd.select

#         Returns
#         -------
#         self
#             戻り値は自分自身．
#         """
#         columns = list(kwargs.keys())
#         if self.id not in columns:
#             columns.append(self.id)

#         selected_ids = self.get(columns, **kwargs)[self.id]
#         df = self.df[self.df[self.id].isin(selected_ids)]
#         data = self.clone(df)
#         data.selected_ids = selected_ids
#         return data

#     @staticmethod
#     def required_columns(*args, **kwargs):
#         return required_columns(*args, **kwargs)


# class DataSets:
#     def __init__(self, paths=None, index="index", names=None, cls=DataSet, **kwargs):
#         if paths is not None:
#             self.paths = list(paths)
#             self.datasets = [cls(path, **kwargs) for path in paths]
#             for k, dataset in enumerate(self.datasets):
#                 dataset.df[index] = names[k] if names else k

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         for dataset in self.datasets:
#             dataset.close()

#     def __getitem__(self, index):
#         data = []
#         for k, dataset in enumerate(self.datasets):
#             df = dataset[index]
#             # if (isinstance(df, pd.DataFrame) and len(df) > 0
#             #         and self.index in index):
#             #     df[self.index] = self.names[k]
#             data.append(df)
#         return pd.concat(data, ignore_index=True)

#     def __len__(self):
#         return tuple(len(dataset) for dataset in self.datasets)

#     def __str__(self):
#         return f"DataSets({self.length})"

#     def __repr__(self):
#         basenames = [
#             repr(dataset).replace("<DataSet(", "").replace(")>", "")
#             for dataset in self.datasets
#         ]
#         return f"<DataSets({basenames})>"

#     def select(self, **kwargs):
#         datasets = self.__class__()
#         datasets.paths = self.paths
#         datasets.datasets = [dataset.select(**kwargs) for dataset in self.datasets]
#         datasets.index = self.index
#         datasets.names = self.names
#         return datasets

#     def __getattr__(self, name):
#         """未知の属性は，元データに委譲する．関数のみ対応"""

#         def func(*args, **kwargs):
#             rets = []
#             for dataset in self.datasets:
#                 attr = getattr(dataset, name)
#                 rets.append(attr(*args, **kwargs))
#             if any(rets):
#                 return rets

#         return func

#     @property
#     def length(self):
#         return [dataset.length for dataset in self.datasets]

#     @property
#     def shape(self):
#         return [dataset.shape for dataset in self.datasets]

#     @property
#     def path(self):
#         return self.paths

#     def get(self, *args, **kwargs):
#         data = []
#         for k, dataset in enumerate(self.datasets):
#             df = dataset.get(*args, **kwargs)
#             data.append(df)
#         return pd.concat(data, ignore_index=True)


# def query_hdf(columns, **kwargs):
#     """
#     HDF5のクエリ文字列を返す．
#     """
#     queries = [f"columns={columns}"]
#     for key, value in kwargs.items():
#         if isinstance(value, tuple):
#             queries.append(f"({key} >= {value[0]} and {key} <= {value[1]})")
#         else:
#             queries.append(f"{key}={value}")
#     return " and ".join(queries)


# def flatten(columns):
#     """
#     ['a', 'b', ('c', 'd')] -> ['a', 'b', 'c', 'd']
#     """
#     for column in columns:
#         if not isinstance(column, str):
#             yield from column
#         else:
#             yield column


# def required_columns(*args, fstring=None, **kwargs):
#     """
#     DataSetで必要なカラムを抽出する．

#     Parameters
#     ----------
#     args :
#         直接指定
#     fstring :
#         f-string指定
#     kwargs :
#         キーワード引数指定

#     Returns
#     -------
#     columns : list
#         カラム名のリスト
#     """
#     columns = list(args)
#     columns.extend(kwargs.keys())
#     if fstring:
#         columns.extend(re.findall(r"{(\w+?)}", fstring))
#     return list(set(columns))
