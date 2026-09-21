"""One reader per file format.

Each reader returns the frame as the file holds it. Anything a reader cannot represent
faithfully raises ``DataLoadError`` instead of being repaired.
"""

import csv
import json
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from datadoctor.core.exceptions import DataLoadError
from datadoctor.io.leading_zeros import find_leading_zeros

_COMMON_DELIMITERS = ",;\t|"


class ReadResult(NamedTuple):
    """A frame, and what the reader changed or could not keep exactly as written.

    ``converted_tokens`` maps a column to the literal cell texts read as missing, with counts.
    ``unnamed_columns`` are columns whose empty header cell was named ``Unnamed: <position>``.
    ``promoted_index`` are columns created from a stored index.
    ``leading_zeros`` maps a column that was read as numbers to how many of its values had a
    leading zero in the file, which the numbers no longer show. See ``find_leading_zeros``.
    """

    frame: pd.DataFrame
    converted_tokens: dict[str, dict[str, int]]
    unnamed_columns: tuple[str, ...]
    promoted_index: tuple[str, ...]
    leading_zeros: dict[str, dict[str, int]]


def read_delimited(
    file: Path, *, separator: str, encoding: str, text_columns: tuple[str, ...] = ()
) -> ReadResult:
    """Read a csv or tsv file. ``text_columns`` are read as text, not as inferred types."""
    if len(separator) != 1:
        raise DataLoadError(f"separator must be a single character, got {separator!r}")

    header = _read_header(file, separator, encoding)
    if not header:
        raise DataLoadError(f"{file}: the file is empty")
    _reject_duplicates(file, header)
    labels = [name or f"Unnamed: {position}" for position, name in enumerate(header)]
    as_text = _text_dtypes(file, labels, text_columns)

    frame = _read_frame(file, separator, encoding, dtype=as_text)
    _check_separator(file, frame, separator)
    tokens = _converted_tokens(
        frame, lambda positions: _read_frame(file, separator, encoding, raw_positions=positions)
    )
    zeros = find_leading_zeros(
        frame,
        lambda positions, n_rows: _read_frame(
            file, separator, encoding, raw_positions=positions, n_rows=n_rows
        ),
    )
    pairs = zip(frame.columns, header, strict=False)
    unnamed = tuple(str(column) for column, name in pairs if name == "")
    return ReadResult(frame, tokens, unnamed, (), zeros)


def read_json(file: Path, *, encoding: str) -> ReadResult:
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
    return ReadResult(pd.DataFrame(data), {}, (), (), {})


def read_parquet(file: Path) -> ReadResult:
    """Read a parquet file. A stored index that is not a plain row range becomes columns."""
    try:
        frame = pd.read_parquet(file, engine="pyarrow")
    except ImportError as exc:
        raise _missing_engine(file, "parquet", "pyarrow") from exc
    except (ValueError, OSError) as exc:
        raise DataLoadError(f"{file}: could not read the file as parquet: {exc}") from exc

    promoted: tuple[str, ...] = ()
    if not isinstance(frame.index, pd.RangeIndex):
        stored = set(frame.columns)
        try:
            frame = frame.reset_index()
        except ValueError as exc:
            raise DataLoadError(
                f"{file}: the stored index cannot be turned into a column: {exc}"
            ) from exc
        promoted = tuple(str(column) for column in frame.columns if column not in stored)
    return ReadResult(frame, {}, (), promoted, {})


def read_excel(
    file: Path, *, sheet: str | int | None, text_columns: tuple[str, ...] = ()
) -> ReadResult:
    """Read one sheet of an xlsx workbook. Several sheets require an explicit choice.

    ``text_columns`` are read as text, not as inferred types. Without that, pandas turns even
    cells stored as text, such as ``01234``, into numbers.
    """
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
        names = header.iloc[0].tolist()
        _reject_duplicates(file, names)
        labels = [f"Unnamed: {position}" if pd.isna(n) else n for position, n in enumerate(names)]
        as_text = _text_dtypes(file, labels, text_columns)

        frame = workbook.parse(name, dtype=as_text or None)
        tokens = _converted_tokens(
            frame,
            lambda positions: workbook.parse(
                name, usecols=positions, dtype=str, keep_default_na=False, na_filter=False
            ),
        )
        zeros = find_leading_zeros(
            frame,
            lambda positions, n_rows: workbook.parse(
                name,
                usecols=positions,
                nrows=n_rows,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            ),
        )
        pairs = zip(frame.columns, names, strict=False)
        unnamed = tuple(str(column) for column, value in pairs if pd.isna(value))
        return ReadResult(frame, tokens, unnamed, (), zeros)


_LISTED_COLUMNS = 20


def _text_dtypes(file: Path, labels: list[Any], text_columns: tuple[str, ...]) -> dict[Any, type]:
    """Map each requested column to ``str``, or raise if the file has no such column.

    Requested names are matched against ``str(label)`` because Excel headers can be numbers. The
    returned keys are the labels themselves, which is what pandas matches ``dtype`` against.
    """
    by_name = {str(label): label for label in labels}
    missing = [name for name in text_columns if name not in by_name]
    if missing:
        shown = ", ".join(list(by_name)[:_LISTED_COLUMNS])
        more = len(by_name) - _LISTED_COLUMNS
        raise DataLoadError(
            f"{file}: no column named {', '.join(repr(name) for name in missing)}; "
            f"columns: {shown}" + (f" and {more} more" if more > 0 else "")
        )
    return {by_name[name]: str for name in text_columns}


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


def _converted_tokens(
    frame: pd.DataFrame, read_raw: Callable[[list[int]], pd.DataFrame]
) -> dict[str, dict[str, int]]:
    """Count the literal texts that pandas read as missing, per column.

    The file is read again, only for the columns that have missing values, with no missing-value
    conversion at all. A cell that is missing in ``frame`` but not empty in that raw read was
    converted from a literal token. Comparing against what pandas actually did, and not against a
    list of tokens, stays right if pandas changes its defaults. Columns are matched by position
    because labels can be numbers, which pandas would treat as positions anyway.
    """
    positions = [i for i in range(frame.shape[1]) if frame.iloc[:, i].isna().any()]
    if not positions:
        return {}
    raw = read_raw(positions)
    found: dict[str, dict[str, int]] = {}
    for offset, position in enumerate(positions):
        literal = raw.iloc[:, offset]
        hits = literal[frame.iloc[:, position].isna() & literal.ne("")]
        if not hits.empty:
            counts = hits.value_counts().items()
            found[str(frame.columns[position])] = {
                str(token): int(count) for token, count in sorted(counts)
            }
    return found


def _read_frame(
    file: Path,
    separator: str,
    encoding: str,
    *,
    raw_positions: list[int] | None = None,
    n_rows: int | None = None,
    dtype: dict[Any, type] | None = None,
) -> pd.DataFrame:
    # index_col=False stops pandas from silently promoting the first column to the index when
    # rows are longer than the header. It then truncates those rows and emits a ParserWarning,
    # which is turned into an error so that no data is lost quietly.
    # raw_positions selects columns to read as untouched text, for counting converted tokens and
    # leading zeros. n_rows limits how many rows that read returns. dtype names the columns of the
    # main read that are to be kept as text.
    raw = (
        {"dtype": dtype or None}
        if raw_positions is None
        else {
            "usecols": raw_positions,
            "dtype": str,
            "keep_default_na": False,
            "na_filter": False,
            "nrows": n_rows,
        }
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                file,
                sep=separator,
                encoding=encoding,
                index_col=False,
                low_memory=False,
                **raw,
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
