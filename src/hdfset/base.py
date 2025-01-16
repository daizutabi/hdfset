from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, overload

from pandas import DataFrame, HDFStore, Series

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Self

NUM_ID_LIMIT = 1000


class BaseDataSet:
    path: Path
    store: HDFStore
    id: str | None

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.store = HDFStore(self.path, mode="r")
        self.id = None

    @staticmethod
    def key(index: int) -> str:
        return f"/_{index}"

    @staticmethod
    def to_hdf(path: str | Path, dataframes: list[DataFrame | None]) -> None:
        """Save a list of DataFrames to an HDF5 file.

        Args:
            path (str or Path): The file path where the data will be saved.
            dataframes (list of DataFrame or None): A list of DataFrames to be saved.
                If a DataFrame is None, it will be skipped.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for k, df in enumerate(dataframes):
            if df is None:
                continue

            df.to_hdf(
                path,
                key=BaseDataSet.key(k),
                complevel=9,
                complib="blosc",
                format="table",
                data_columns=True,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.path.stem!r})>"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.store.close()

    def __len__(self) -> int:
        return len(self.store.keys())

    def storers(self) -> Iterator:
        for key in self.store:
            yield self.store.get_storer(key)  # type: ignore

    def __iter__(self) -> Iterator[list[str]]:
        return (storer.data_columns for storer in self.storers())

    @property
    def columns(self) -> list[str]:
        return list(chain.from_iterable(self))

    def __contains__(self, column: str) -> bool:
        return any(column in columns for columns in self)

    @property
    def length(self) -> tuple[int, ...]:
        return tuple(int(storer.nrows) for storer in self.storers())

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.length})"

    def index(self, columns: str | Iterable[str]) -> str:
        if isinstance(columns, str):
            columns = [columns]

        for key, columns_ in zip(self.store, self, strict=True):
            if all(column in columns_ for column in columns):
                return key

        raise IndexError("The specified columns were not found.")

    def get_index_dict(
        self,
        columns: Iterable[str | tuple[str, ...]],
    ) -> dict[str, str]:
        index_dict: dict[str, str] = {}

        for column in columns:
            index = self.index(column)
            if isinstance(column, tuple):
                for c in column:
                    index_dict[c] = index
            else:
                index_dict[column] = index

        return index_dict

    def select(
        self,
        index: int | str,
        where: str | dict | None = None,
        *args,
        **kwargs,
    ) -> DataFrame | Series:
        if isinstance(index, int):
            index = self.key(index)

            if index not in self.store:
                msg = "The specified index was not found."
                raise IndexError(msg)

        if isinstance(where, dict):
            where = query_string(where)

        return self.store.select(index, where, *args, **kwargs)

    def get_id_column(self) -> str:
        columns = set.intersection(*[set(x) for x in self])

        if len(columns) != 1:
            self.store.close()
            msg = "The number of id columns is not equal to 1."
            raise ValueError(msg)

        return columns.pop()

    @overload
    def get(self, columns: str, **kwargs) -> Series: ...

    @overload
    def get(self, columns: Sequence[str | tuple[str, ...]], **kwargs) -> DataFrame: ...

    def get(
        self,
        columns: str | Sequence[str | tuple[str, ...]],
        **kwargs,
    ) -> DataFrame | Series:
        """Extract necessary data from multiple DataFrames.

        Args:
            columns: Data selection list. Retrieve data across multiple DataFrames.
                If you want to retrieve data collectively from the same DataFrame,
                enclose it in a tuple. ['x', 'y', ('a', 'b')]
        """
        if isinstance(columns, str):
            return self.get([columns], **kwargs)[columns]

        if self.id is None:
            self.id = self.get_id_column()

        column_indexes = self.get_index_dict(columns)
        kwarg_indexes = self.get_index_dict(kwargs.keys())
        indexes = sorted(set(column_indexes.values()).union(kwarg_indexes.values()))

        df = None
        selected_ids = None

        for index in indexes:
            subcolumns = [c for c in column_indexes if column_indexes[c] == index]
            subkwargs = {k: v for k, v in kwargs.items() if kwarg_indexes[k] == index}

            if self.id not in subcolumns:
                subcolumns = [self.id, *subcolumns]

            where = {self.id: selected_ids} if selected_ids else {}
            where.update(subkwargs)

            sub = self.select(index, where, columns=subcolumns)

            if where:
                selected_ids = sub[self.id].drop_duplicates().to_list()
                if len(selected_ids) > NUM_ID_LIMIT:
                    selected_ids = None

            how = "inner" if kwargs else "left"
            df = sub if df is None else df.merge(sub, how=how)

        if df is None:
            raise ValueError("No data was found.")

        return df[list(flatten(columns))]

    @overload
    def __getitem__(self, index: int | str) -> Series: ...

    @overload
    def __getitem__(self, index: Sequence[str | tuple[str, ...]]) -> DataFrame: ...

    def __getitem__(
        self,
        index: int | str | Sequence[str | tuple[str, ...]],
    ) -> DataFrame | Series:
        if isinstance(index, int):
            return self.get(self.columns[index])

        if isinstance(index, str | list):
            return self.get(index)

        raise NotImplementedError


def query_string(where: dict | None = None) -> str:
    """Return the query string for HDF5."""
    if where is None:
        return ""

    queries = []

    for key, value in where.items():
        if isinstance(value, tuple):
            if value[0] is None:
                queries.append(f"{key}<={value[1]}")
            elif value[1] is None:
                queries.append(f"{key}>={value[0]}")
            else:
                queries.append(f"({key}>={value[0]} and {key}<={value[1]})")
        else:
            queries.append(f"{key}={value}")

    return " and ".join(queries)


def flatten(columns: Iterable[str | tuple[str, ...]]) -> Iterator[str]:
    """
    Example:
        >>> list(flatten(['a', 'b', ('c', 'd')]))
        ['a', 'b', 'c', 'd']
    """
    for column in columns:
        if not isinstance(column, str):
            yield from column
        else:
            yield column
