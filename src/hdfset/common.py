from collections.abc import Iterable

import pandas as pd


def select(df: pd.DataFrame, *sel, **kwargs) -> pd.DataFrame:
    """データフレームから条件に合う行を選択する．

    Parameters
    ----------
    df : DataFrame or list
        データフレームかデータフレームのリスト
    sel : dict, str, pandas.Series, list, callable
        行選別辞書．{'カラム': '値', ...}．
        値は，数値/文字列: 一致， リスト:含有， タプル: (最小，最大)
        文字列の場合がquery

    Returns
    -------
    selected_df : DataFrame
        選択されたデータフレーム

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 5, 6],
    ...                   [2, 3, 7, 8], [2, 3, 9, 10]],
    ...                   columns=list('abcd'))

    値を指定すると，一致する行を選択する．

    >>> select(df, {'a': 1})
       a  b  c  d
    0  1  2  3  4
    1  1  2  5  6

    >>> select(df, [{'a': 1}, 'c == 5'])
       a  b  c  d
    1  1  2  5  6

    >>> select(df, a=1)
       a  b  c  d
    0  1  2  3  4
    1  1  2  5  6

    リストを指定すると，要素を含む行を選択する．

    >>> select(df, {'c': [3, 7, 9]})
       a  b  c   d
    0  1  2  3   4
    2  2  3  7   8
    3  2  3  9  10

    タプルを指定すると，その範囲にある行を選択する．

    >>> select(df, {'c': (5, 7)})
       a  b  c  d
    1  1  2  5  6
    2  2  3  7  8

    複数のカラムを指定すると，ANDで真となる行を選択する．

    >>> select(df, {'c': (5, 7), 'a': 1})
       a  b  c  d
    1  1  2  5  6

    文字列を与えるとqueryを実行する．

    >>> select(df, 'c > 3')
       a  b  c   d
    1  1  2  5   6
    2  2  3  7   8
    3  2  3  9  10

    関数

    >>> select(df, {'c': lambda x: x > 3})
       a  b  c   d
    1  1  2  5   6
    2  2  3  7   8
    3  2  3  9  10

    >>> select(df, lambda df: (df['a'] == 1) & (df['c'] == 3))
       a  b  c  d
    0  1  2  3  4
    """
    if len(sel) == 0:
        if kwargs:
            selection = kwargs
        else:
            return df
    elif len(sel) == 1:
        selection = sel[0]
    else:
        raise ValueError("Invalid argument length")

    if isinstance(selection, list):
        for s in selection:
            df = select(df, s)
        return df
    if selection is None:
        return df
    if isinstance(df, list):
        return [select(df_, selection) for df_ in df]
    if isinstance(selection, str):
        return df.query(selection)
    if callable(selection):
        return df[selection(df)]
    if isinstance(selection, pd.Series):
        selection = dict(selection)
    elif not isinstance(selection, dict):
        raise TypeError("Invalid value of selection")

    for key in selection:
        if key in df:
            arg = selection[key]
            if isinstance(arg, tuple):
                df = df[(df[key] >= arg[0]) & (df[key] <= arg[1])]
            elif isinstance(arg, Iterable) and not isinstance(arg, str):
                df = df[df[key].isin(arg)]
            elif callable(arg):
                df = df[df[key].map(arg)]
            else:
                df = df[df[key] == arg]
    return df
