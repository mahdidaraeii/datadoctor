"""The dataset abstraction every analyzer works on."""

from dataclasses import dataclass

import pandas as pd

from datadoctor.core.exceptions import DatasetError
from datadoctor.core.provenance import Provenance


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Dataset:
    """A dataframe together with the metadata describing where it came from.

    The dataframe is stored as given and is never modified or copied. ``eq=False`` keeps
    identity equality, because comparing dataframes with ``==`` does not yield a single bool.

    Attributes:
        data: The tabular data.
        name: Human-readable name of the dataset.
        source: Where the data was loaded from, if it came from a file.
        target: Name of the column to be predicted, if there is one.
        provenance: How the data was obtained. Always set: when not given, it records the
            frame's shape and the current versions, with no file.
    """

    data: pd.DataFrame
    name: str
    source: str | None = None
    target: str | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(self.data).__name__}")
        if self.target is not None and self.target not in self.data.columns:
            raise DatasetError(f"target column {self.target!r} is not in the dataset")
        if self.provenance is None:
            object.__setattr__(self, "provenance", Provenance.capture(self.data))
