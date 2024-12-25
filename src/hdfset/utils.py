from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas.io.pytables import HDFStore


def get_id_column(store: HDFStore) -> str:
    columns = set()

    for key in store:
        df = store.select(key, start=0, stop=0)
        if len(columns) == 0:
            columns = columns.union(df.columns)
        else:
            columns = columns.intersection(df.columns)

    if len(columns) != 1:
        raise ValueError("The number of id columns is not equal to 1.")

    return columns.pop()
