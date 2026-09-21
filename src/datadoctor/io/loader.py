"""Load a dataset from a file."""

import os
from collections.abc import Sequence
from pathlib import Path

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DataLoadError
from datadoctor.core.provenance import Provenance
from datadoctor.io.readers import (
    ReadResult,
    read_delimited,
    read_excel,
    read_json,
    read_parquet,
)

_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".json": "json",
}

# The options each format accepts. Passing any other option is an error, never ignored.
_OPTIONS = {
    "csv": {"separator", "encoding", "text_columns"},
    "tsv": {"encoding", "text_columns"},
    "json": {"encoding"},
    "parquet": set(),
    "excel": {"sheet", "text_columns"},
}


def load_dataset(
    path: str | os.PathLike[str],
    *,
    file_format: str | None = None,
    target: str | None = None,
    separator: str | None = None,
    encoding: str | None = None,
    sheet: str | int | None = None,
    text_columns: Sequence[str] | None = None,
    config: AnalysisConfig | None = None,
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
    - csv, tsv and Excel: a column of codes such as ``01234`` is read as the number 1234, and the
      zeros are lost. Excel does this even for cells stored as text. The file is checked for
      this, see below. json and parquet keep such values as text.
    - csv and tsv: column types are inferred from the whole file. A row with fewer fields than
      the header is padded with missing values, and an empty header cell becomes
      ``Unnamed: <position>``. Excel also names an empty header cell ``Unnamed: <position>``.
    - json: the file must be an array of objects. Values keep their JSON types, so ``"01234"``
      stays text. A key missing from some objects is filled with missing values.
    - parquet: a stored index that is not a plain row range is turned into ordinary columns
      with ``reset_index``. It keeps its name, or becomes ``index`` if it had none.
    - Excel: only one sheet is read. A workbook with several sheets requires ``sheet``.

    What the loader changed or could not keep as written is recorded in ``Dataset.provenance``:
    the literal texts read as missing (``converted_tokens``), empty header cells that were named
    ``Unnamed: <position>`` (``unnamed_columns``), a promoted parquet index (``promoted_index``)
    and numeric columns whose values had leading zeros in the file (``leading_zeros``). The last
    is checked on the first 100,000 rows, for csv, tsv and Excel files. Rows padded because they
    were shorter than the header are not recorded.

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
        text_columns: csv, tsv and Excel only. Names of columns to read as text instead of
            inferring their type, so that codes such as ``01234`` keep their leading zeros. An
            empty header cell can be named ``Unnamed: <position>``. The usual missing-value
            tokens are still read as missing. The request is recorded in the provenance.
        config: The analysis settings to record in the provenance, ``AnalysisConfig()`` by
            default. Loading itself does not depend on them.

    Returns:
        A ``Dataset`` named after the file, with ``source`` set to the path as given and a
        ``provenance`` that includes the SHA-256 of the file.

    Raises:
        DataLoadError: The file is missing, has an unknown format, is given an option that does
            not apply to its format, is empty or has no data rows, cannot be decoded or parsed,
            has duplicate column names, is asked to keep a column as text that it does not have,
            or violates a rule of its format (see above and the error message).
        TypeError: ``text_columns`` is a single string or holds something other than strings.
        DatasetError: ``target`` is not a column of the file.
    """
    file = Path(path)
    if not file.exists():
        raise DataLoadError(f"file not found: {file}")
    if not file.is_file():
        raise DataLoadError(f"not a file: {file}")

    requested = _requested_text_columns(text_columns)
    fmt = _resolve_format(file, file_format)
    given = {
        name
        for name, value in (
            ("separator", separator),
            ("encoding", encoding),
            ("sheet", sheet),
            ("text_columns", requested or None),
        )
        if value is not None
    }
    inapplicable = sorted(given - _OPTIONS[fmt])
    if inapplicable:
        raise DataLoadError(f"{file}: {', '.join(inapplicable)} does not apply to {fmt} files")

    read = _read(fmt, file, separator, encoding, sheet, requested)
    if read.frame.empty:
        raise DataLoadError(f"{file}: the file has no data rows")
    provenance = Provenance.capture(
        read.frame,
        config=config,
        file=file,
        converted_tokens=read.converted_tokens,
        unnamed_columns=read.unnamed_columns,
        promoted_index=read.promoted_index,
        leading_zeros=read.leading_zeros,
        text_columns=requested,
    )
    return Dataset(
        data=read.frame,
        name=file.stem,
        source=file.as_posix(),
        target=target,
        provenance=provenance,
    )


def _requested_text_columns(text_columns: Sequence[str] | None) -> tuple[str, ...]:
    if text_columns is None:
        return ()
    if isinstance(text_columns, str) or not all(isinstance(name, str) for name in text_columns):
        raise TypeError("text_columns must be a sequence of column names, not a single string")
    return tuple(dict.fromkeys(text_columns))


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
    text_columns: tuple[str, ...],
) -> ReadResult:
    encoding = "utf-8-sig" if encoding is None else encoding
    if fmt == "csv":
        return read_delimited(
            file,
            separator="," if separator is None else separator,
            encoding=encoding,
            text_columns=text_columns,
        )
    if fmt == "tsv":
        return read_delimited(file, separator="\t", encoding=encoding, text_columns=text_columns)
    if fmt == "json":
        return read_json(file, encoding=encoding)
    if fmt == "parquet":
        return read_parquet(file)
    return read_excel(file, sheet=sheet, text_columns=text_columns)
