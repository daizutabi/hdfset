from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, overload

from pandas import DataFrame, HDFStore, Series

from hdfset.base import BaseDataSet

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self


class DataSet(BaseDataSet):
    df: DataFrame

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

        df = self.select(0)

        if not isinstance(df, DataFrame):
            msg = "The first DataFrame is not a valid DataFrame."
            raise TypeError(msg)

        self.df = df

    def __iter__(self) -> Iterator[list[str]]:
        it = super().__iter__()
        next(it)

        yield self.df.columns.to_list()
        yield from it
