"""Find numeric columns that lost leading zeros when a file was read.

A column of codes such as ``01234`` is read as the number 1234, and the zeros are gone from the
loaded data. Only the file's own text can show they were there, so the columns that could have
lost zeros are read again as untouched text and compared. The comparison is a fact about the
file. Whether the column really holds codes is a heuristic, and is left to the diagnostics.
"""

from collections.abc import Callable

import pandas as pd
from pandas.api import types as pt

# Rows examined per column. A file with more rows is checked on its first rows only, and the
# record says how many were checked. Reading every row would cost about as much as the file.
CHECK_ROWS = 100_000

_LEADING_ZERO = r"0\d"


def find_leading_zeros(
    frame: pd.DataFrame, read_raw: Callable[[list[int], int], pd.DataFrame]
) -> dict[str, dict[str, int]]:
    """Count the values with a leading zero in the columns that were read as numbers.

    ``read_raw(positions, n_rows)`` must return the first ``n_rows`` rows of those columns as
    untouched text, with no missing-value conversion. Columns are matched by position because
    labels can be numbers.

    Returns ``{column: {"values": ..., "checked": ...}}`` for each column where at least one raw
    value has a leading zero. ``values`` counts them, ``checked`` is the number of rows
    examined, and ``width`` is present when every non-empty value in those rows had the same
    length, which is typical of codes.
    """
    positions = [i for i in range(frame.shape[1]) if _could_hold_codes(frame.iloc[:, i])]
    if not positions:
        return {}
    raw = read_raw(positions, CHECK_ROWS)
    found: dict[str, dict[str, int]] = {}
    for offset, position in enumerate(positions):
        text = raw.iloc[:, offset]
        count = int(text.str.match(_LEADING_ZERO).sum())
        if not count:
            continue
        entry = {"values": count, "checked": len(text)}
        widths = text[text != ""].str.len().unique()
        if len(widths) == 1:
            entry["width"] = int(widths[0])
        found[str(frame.columns[position])] = entry
    return found


def _could_hold_codes(series: pd.Series) -> bool:
    """Whether a numeric column holds only whole, non-negative numbers, as a code column would."""
    if not pt.is_numeric_dtype(series) or pt.is_bool_dtype(series):
        return False
    values = series.dropna()
    return bool(not values.empty and (values >= 0).all() and (values % 1 == 0).all())
