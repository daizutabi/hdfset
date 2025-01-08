from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame, Series

from hdfset.base import BaseDataSet

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


class DataSet(BaseDataSet):
    df: DataFrame

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

        df = self.store.select(self.key(0))

        if not isinstance(df, DataFrame):
            self.store.close()
            msg = "The first DataFrame is not a valid DataFrame."
            raise TypeError(msg)

        self.df = df

    def __iter__(self) -> Iterator[list[str]]:
        it = super().__iter__()
        next(it)

        yield self.df.columns.to_list()
        yield from it

    def merge(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        self.df = self.df.merge(df, *args, **kwargs)

        return self.df

    def select(
        self,
        index: int | str,
        where: str | dict | None = None,
        *args,
        columns: list[str] | None = None,
        **kwargs,
    ) -> DataFrame | Series:
        if index == 0 or index == self.key(0):
            return select(self.df, where, columns)

        return super().select(index, where, *args, columns=columns, **kwargs)


def select(
    df: DataFrame,
    where: dict | str | None = None,
    columns: list[str] | None = None,
) -> DataFrame:
    if isinstance(where, str):
        df = df.query(where)

    elif isinstance(where, dict):
        for column, value in where.items():
            df = select_by_column(df, column, value)

    if columns:
        df = df[columns]

    return df


def select_by_column(df: DataFrame, column: str, value: Any) -> DataFrame:
    s = df[column]

    if isinstance(value, list):
        return df.loc[s.isin(value)]

    if isinstance(value, tuple):
        return df.loc[(s >= value[0]) & (s <= value[1])]

    return df.loc[s == value]
