"""One rule for when two values are the same, shared by the duplicate and constant checks.

Each column is reduced to integer codes, one per distinct value. Every missing value (``None``,
``NaN``, ``NaT`` and ``pd.NA`` alike) gets the code ``-1``, so missing values are equal to each
other, and ``0.0`` equals ``-0.0``. This gives the same answer for every dtype and for every
supported pandas version, which comparing the raw values does not: in an object column pandas
treats ``None`` and ``NaN`` as different values.
"""

import numpy as np
import pandas as pd

MISSING = -1


def column_codes(series: pd.Series) -> np.ndarray | None:
    """Integer codes for the values of a column, or ``None`` if its values cannot be compared.

    Values that cannot be hashed, such as lists or dicts inside cells, cannot be compared. The
    caller must say so and leave the column out, never guess.
    """
    try:
        codes, _ = pd.factorize(series)
    except TypeError:
        return None
    return codes.astype(np.int32)
