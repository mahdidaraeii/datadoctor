"""Load a dataset from a csv file."""

import csv
import os
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DataLoadError

_COMMON_DELIMITERS = ",;\t|"


def load_dataset(
    path: str | os.PathLike[str],
    *,
    target: str | None = None,
    separator: str = ",",
    encoding: str = "utf-8-sig",
) -> Dataset:
    """Read a csv file into a ``Dataset``.

    The file is read as written: no rows are dropped, no values are filled and no column is
    renamed. Anything the loader cannot represent faithfully raises ``DataLoadError`` instead of
    being repaired, so a problem in the file is never hidden.

    pandas defaults that still apply, and that a caller should know about:

    - Empty cells and the standard missing-value tokens (``NA``, ``N/A``, ``NaN``, ``null``,
      ``None`` and similar) are read as missing. A genuine value spelled like one of these, such
      as the country code ``NA``, is also read as missing.
    - Column types are inferred from the whole file. Nothing is coerced beyond that.
    - A row with fewer fields than the header is padded with missing values.
    - An empty header cell becomes ``Unnamed: <position>``.

    Args:
        path: The csv file to read.
        target: Name of the column to be predicted, if there is one.
        separator: The single character that separates fields. It is never guessed.
        encoding: The file's text encoding. The default also accepts a UTF-8 byte order mark.

    Returns:
        A ``Dataset`` named after the file, with ``source`` set to the path as given.

    Raises:
        DataLoadError: The file is missing, empty, has no data rows, cannot be decoded with
            ``encoding``, cannot be parsed, has duplicate column names, has rows longer than the
            header, or does not appear to use ``separator``.
        DatasetError: ``target`` is not a column of the file.
    """
    file = Path(path)
    if len(separator) != 1:
        raise DataLoadError(f"separator must be a single character, got {separator!r}")
    if not file.exists():
        raise DataLoadError(f"file not found: {file}")
    if not file.is_file():
        raise DataLoadError(f"not a file: {file}")

    header = _read_header(file, separator, encoding)
    if not header:
        raise DataLoadError(f"{file}: the file is empty")
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise DataLoadError(f"{file}: duplicate column names: {', '.join(duplicates)}")

    frame = _read_frame(file, separator, encoding)
    _check_separator(file, frame, separator)
    if frame.empty:
        raise DataLoadError(f"{file}: the file has a header but no data rows")

    return Dataset(data=frame, name=file.stem, source=file.as_posix(), target=target)


def _read_header(file: Path, separator: str, encoding: str) -> list[str]:
    try:
        with file.open(encoding=encoding, newline="") as handle:
            return next((row for row in csv.reader(handle, delimiter=separator) if row), [])
    except UnicodeDecodeError as exc:
        raise _decode_error(file, encoding, exc) from exc


def _read_frame(file: Path, separator: str, encoding: str) -> pd.DataFrame:
    # index_col=False stops pandas from silently promoting the first column to the index when
    # rows are longer than the header. It then truncates those rows and emits a ParserWarning,
    # which is turned into an error so that no data is lost quietly.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                file,
                sep=separator,
                encoding=encoding,
                index_col=False,
                low_memory=False,
            )
    except UnicodeDecodeError as exc:
        raise _decode_error(file, encoding, exc) from exc
    except pd.errors.ParserWarning as exc:
        raise DataLoadError(f"{file}: data rows have more fields than the header ({exc})") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"{file}: could not parse the file as csv: {exc}") from exc


def _check_separator(file: Path, frame: pd.DataFrame, separator: str) -> None:
    if frame.shape[1] != 1:
        return
    name = str(frame.columns[0])
    for other in _COMMON_DELIMITERS:
        if other != separator and other in name:
            raise DataLoadError(
                f"{file}: only one column was found with separator {separator!r}, but its name "
                f"contains {other!r}. If {other!r} is the delimiter, pass separator={other!r}."
            )


def _decode_error(file: Path, encoding: str, exc: UnicodeDecodeError) -> DataLoadError:
    return DataLoadError(
        f"{file}: cannot decode the file as {encoding!r} ({exc.reason} at byte {exc.start}). "
        "Pass the file's real encoding, for example encoding='latin-1' or 'cp1252'."
    )
