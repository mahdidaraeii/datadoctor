"""One reader per file format.

Each reader returns the frame as the file holds it. Anything a reader cannot represent
faithfully raises ``DataLoadError`` instead of being repaired.
"""

import csv
import json
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from datadoctor.core.exceptions import DataLoadError

_COMMON_DELIMITERS = ",;\t|"


def read_delimited(file: Path, *, separator: str, encoding: str) -> pd.DataFrame:
    """Read a csv or tsv file."""
    if len(separator) != 1:
        raise DataLoadError(f"separator must be a single character, got {separator!r}")

    header = _read_header(file, separator, encoding)
    if not header:
        raise DataLoadError(f"{file}: the file is empty")
    _reject_duplicates(file, header)

    frame = _read_frame(file, separator, encoding)
    _check_separator(file, frame, separator)
    return frame


def read_json(file: Path, *, encoding: str) -> pd.DataFrame:
    """Read a json file holding an array of objects.

    The stdlib parser is used, not ``pd.read_json``, because pandas converts numeric strings to
    numbers and date-like column names to datetimes.
    """
    try:
        with file.open(encoding=encoding) as handle:
            data = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise _decode_error(file, encoding, exc) from exc
    except ValueError as exc:
        raise DataLoadError(f"{file}: invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise DataLoadError(f"{file}: expected a JSON array of objects, got {type(data).__name__}")
    for position, record in enumerate(data):
        if not isinstance(record, dict):
            raise DataLoadError(f"{file}: element {position} of the array is not an object")
    return pd.DataFrame(data)


def read_parquet(file: Path) -> pd.DataFrame:
    """Read a parquet file. A stored index that is not a plain row range becomes columns."""
    try:
        frame = pd.read_parquet(file, engine="pyarrow")
    except ImportError as exc:
        raise _missing_engine(file, "parquet", "pyarrow") from exc
    except (ValueError, OSError) as exc:
        raise DataLoadError(f"{file}: could not read the file as parquet: {exc}") from exc

    if not isinstance(frame.index, pd.RangeIndex):
        try:
            frame = frame.reset_index()
        except ValueError as exc:
            raise DataLoadError(
                f"{file}: the stored index cannot be turned into a column: {exc}"
            ) from exc
    return frame


def read_excel(file: Path, *, sheet: str | int | None) -> pd.DataFrame:
    """Read one sheet of an xlsx workbook. Several sheets require an explicit choice."""
    try:
        workbook = pd.ExcelFile(file, engine="openpyxl")
    except ImportError as exc:
        raise _missing_engine(file, "excel", "openpyxl") from exc
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DataLoadError(f"{file}: could not read the file as an Excel workbook: {exc}") from exc

    with workbook:
        name = _select_sheet(file, workbook.sheet_names, sheet)
        header = workbook.parse(name, header=None, nrows=1)
        if header.empty:
            raise DataLoadError(f"{file}: sheet {name!r} is empty")
        _reject_duplicates(file, header.iloc[0].tolist())
        return workbook.parse(name)


def _select_sheet(file: Path, names: list[str], sheet: str | int | None) -> str:
    if sheet is None:
        if len(names) > 1:
            raise DataLoadError(
                f"{file}: the workbook has {len(names)} sheets ({', '.join(names)}); "
                "pass sheet= to choose one"
            )
        return names[0]
    if isinstance(sheet, int):
        if not 0 <= sheet < len(names):
            raise DataLoadError(f"{file}: no sheet at position {sheet}; sheets: {', '.join(names)}")
        return names[sheet]
    if sheet not in names:
        raise DataLoadError(f"{file}: no sheet named {sheet!r}; sheets: {', '.join(names)}")
    return sheet


def _reject_duplicates(file: Path, names: list[Any]) -> None:
    duplicates = sorted(str(name) for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise DataLoadError(f"{file}: duplicate column names: {', '.join(duplicates)}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    repeated = sorted(key for key, count in Counter(keys).items() if count > 1)
    if repeated:
        raise ValueError(f"duplicate keys in an object: {', '.join(repeated)}")
    return dict(pairs)


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


def _missing_engine(file: Path, extra: str, package: str) -> DataLoadError:
    return DataLoadError(
        f"{file}: reading {extra} files needs the {package} package. "
        f"Install it with: uv pip install 'datadoctor[{extra}]'"
    )
