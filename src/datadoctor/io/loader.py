"""Load a dataset from a file."""

import os
from pathlib import Path

import pandas as pd

from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DataLoadError
from datadoctor.io.readers import read_delimited, read_excel, read_json, read_parquet

_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".json": "json",
}

# The options each format accepts. Passing any other option is an error, never ignored.
_OPTIONS = {
    "csv": {"separator", "encoding"},
    "tsv": {"encoding"},
    "json": {"encoding"},
    "parquet": set(),
    "excel": {"sheet"},
}


def load_dataset(
    path: str | os.PathLike[str],
    *,
    file_format: str | None = None,
    target: str | None = None,
    separator: str | None = None,
    encoding: str | None = None,
    sheet: str | int | None = None,
) -> Dataset:
    """Read a data file into a ``Dataset``.

    The file is read as written: no rows are dropped, no values are filled and no column is
    renamed. Anything the loader cannot represent faithfully raises ``DataLoadError`` instead of
    being repaired, so a problem in the file is never hidden.

    The format is taken from the file extension (``.csv``, ``.tsv``, ``.parquet``, ``.xlsx``,
    ``.json``) and is never guessed. Use ``file_format`` for any other extension. Parquet needs
    the ``parquet`` extra and Excel needs the ``excel`` extra.

    Behavior that a caller should know about:

    - csv, tsv and Excel: empty cells and the standard missing-value tokens (``NA``, ``N/A``,
      ``NaN``, ``null``, ``None`` and similar) are read as missing. A genuine value spelled like
      one of these, such as the country code ``NA``, is also read as missing.
    - csv and tsv: column types are inferred from the whole file. A row with fewer fields than
      the header is padded with missing values, and an empty header cell becomes
      ``Unnamed: <position>``. Excel also names an empty header cell ``Unnamed: <position>``.
    - json: the file must be an array of objects. Values keep their JSON types, so ``"01234"``
      stays text. A key missing from some objects is filled with missing values.
    - parquet: a stored index that is not a plain row range is turned into ordinary columns
      with ``reset_index``. It keeps its name, or becomes ``index`` if it had none.
    - Excel: only one sheet is read. A workbook with several sheets requires ``sheet``.

    Args:
        path: The file to read.
        file_format: One of ``"csv"``, ``"tsv"``, ``"parquet"``, ``"excel"`` or ``"json"``.
            Overrides the extension.
        target: Name of the column to be predicted, if there is one.
        separator: csv only. The single character that separates fields, ``","`` by default.
            It is never guessed.
        encoding: csv, tsv and json only. The file's text encoding, ``"utf-8-sig"`` by default,
            which also accepts a UTF-8 byte order mark.
        sheet: Excel only. The sheet name, or its zero-based position.

    Returns:
        A ``Dataset`` named after the file, with ``source`` set to the path as given.

    Raises:
        DataLoadError: The file is missing, has an unknown format, is given an option that does
            not apply to its format, is empty or has no data rows, cannot be decoded or parsed,
            has duplicate column names, or violates a rule of its format (see above and the
            error message).
        DatasetError: ``target`` is not a column of the file.
    """
    file = Path(path)
    if not file.exists():
        raise DataLoadError(f"file not found: {file}")
    if not file.is_file():
        raise DataLoadError(f"not a file: {file}")

    fmt = _resolve_format(file, file_format)
    given = {
        name
        for name, value in (("separator", separator), ("encoding", encoding), ("sheet", sheet))
        if value is not None
    }
    inapplicable = sorted(given - _OPTIONS[fmt])
    if inapplicable:
        raise DataLoadError(f"{file}: {', '.join(inapplicable)} does not apply to {fmt} files")

    frame = _read(fmt, file, separator, encoding, sheet)
    if frame.empty:
        raise DataLoadError(f"{file}: the file has no data rows")
    return Dataset(data=frame, name=file.stem, source=file.as_posix(), target=target)


def _resolve_format(file: Path, file_format: str | None) -> str:
    if file_format is not None:
        if file_format not in _OPTIONS:
            expected = ", ".join(sorted(_OPTIONS))
            raise DataLoadError(f"unknown file_format {file_format!r}; expected one of: {expected}")
        return file_format
    extension = file.suffix.lower()
    if extension not in _EXTENSIONS:
        raise DataLoadError(
            f"{file}: cannot tell the file format from the extension {extension!r}. Supported "
            f"extensions: {', '.join(sorted(_EXTENSIONS))}. For any other, pass file_format=."
        )
    return _EXTENSIONS[extension]


def _read(
    fmt: str,
    file: Path,
    separator: str | None,
    encoding: str | None,
    sheet: str | int | None,
) -> pd.DataFrame:
    encoding = "utf-8-sig" if encoding is None else encoding
    if fmt == "csv":
        return read_delimited(
            file, separator="," if separator is None else separator, encoding=encoding
        )
    if fmt == "tsv":
        return read_delimited(file, separator="\t", encoding=encoding)
    if fmt == "json":
        return read_json(file, encoding=encoding)
    if fmt == "parquet":
        return read_parquet(file)
    return read_excel(file, sheet=sheet)
