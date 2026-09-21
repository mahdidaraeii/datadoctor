"""Dtype consistency: numbers stored as text, and columns that mix Python types.

Nothing is converted. Each column is described and the user decides.
"""

import numbers

import pandas as pd
from pandas.api import types as pt

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.quality import dtypes_findings as report

# A text column is reported as numbers when at least this share of its non-null values parse as
# numbers, and it has at least this many non-null values.
MIN_NUMERIC_SHARE = 0.95
MIN_VALUES = 20
# Non-numeric tokens are listed if they are at most this long: a longer one is probably a real
# value that failed to parse, not a placeholder.
MAX_TOKEN_LENGTH = 20
LISTED_TOKENS = 5

# Plain numbers, and US-style numbers with thousands commas. Currency symbols, percent signs and
# European decimal commas are not recognized: "1.234,5" is ambiguous, and is never guessed.
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_THOUSANDS = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_LEADING_ZERO = r"[+-]?0\d"

_MIXED_KINDS = {"mixed", "mixed-integer", "mixed-integer-float"}


def check_dtypes(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Find numbers stored as text, and columns that mix Python types.

    **Numbers stored as text.** In a column of strings, a value parses as a number if it is a plain
    number or a US-style number with thousands commas. A column with at least ``MIN_VALUES``
    non-null values of which at least ``MIN_NUMERIC_SHARE`` parse is reported, with the non-numeric
    tokens that keep it from being fully numeric. These are usually placeholders such as ``?`` or
    ``unknown``. At most five are listed, and any longer than ``MAX_TOKEN_LENGTH`` characters is
    left out, since it is probably a real value. If any parsed value has a leading zero, such as
    ``01234``, the column is reported separately as numeric-looking codes that must not be
    converted, because converting destroys the zeros.

    **Leading zeros lost at load.** A csv, tsv or Excel column of codes such as ``01234`` is read
    as the number 1234. The loader records such columns in ``dataset.provenance.leading_zeros``,
    from the file's own text, and each is reported here as a separate finding, because the zeros
    are already gone from the data. A column that is text in the frame is never reported this
    way, only as numeric-looking codes above. The finding recommends the ``text_columns`` option
    of ``load_dataset``.

    A column listed in ``dataset.provenance.text_columns`` was kept as text on request, so it is
    left out of the numbers-as-text and numeric-looking-codes checks.

    **Mixed types.** An object column whose non-null values are of more than one Python type, for
    example integers and strings from an Excel sheet or a JSON file. The finding gives the exact
    count of each type. A column of only lists or dicts is not mixed.

    Every row is used, so the size guardrails of ``config`` do not apply. ``config`` is recorded.
    Nothing is converted, per the rule against silent changes.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        ``metrics`` with one entry per reported column, and the findings above with ``config``
        recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot check column types in a dataset with no rows")

    # Columns the caller asked to keep as text are not reported as numbers stored as text: that was
    # their choice, and the advice to convert them would contradict it.
    requested = set(dataset.provenance.text_columns)
    reported = []
    for position in range(n_columns):
        series = frame.iloc[:, position]
        values = series.dropna()
        kind = pt.infer_dtype(values, skipna=True)
        entry = None
        if kind == "string" and str(frame.columns[position]) not in requested:
            entry = _numbers_as_text(str(frame.columns[position]), values)
        elif kind in _MIXED_KINDS:
            entry = _mixed_types(str(frame.columns[position]), values)
        if entry is not None:
            reported.append(entry)
    reported.extend(_lost_leading_zeros(dataset))

    def of_kind(kind: str) -> list[dict]:
        return [c for c in reported if c["kind"] == kind]

    findings: list[Finding] = []
    if of_kind("numbers_as_text"):
        findings.append(report.numbers_finding(of_kind("numbers_as_text")))
    if of_kind("numeric_codes"):
        findings.append(report.codes_finding(of_kind("numeric_codes")))
    if of_kind("codes_lost"):
        findings.append(report.lost_zeros_finding(of_kind("codes_lost"), n_rows))
    if of_kind("mixed_types"):
        findings.append(report.mixed_finding(of_kind("mixed_types")))

    metrics = {"n_rows": n_rows, "n_columns": n_columns, "columns": reported}
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _lost_leading_zeros(dataset: Dataset) -> list[dict]:
    """Columns the loader recorded as having lost leading zeros, and that are numeric now.

    A column that is text in the frame is left to the numbers-as-text check, which reports its
    leading zeros as codes. The two never overlap, so no column is reported twice.
    """
    frame = dataset.data
    numeric = {str(label) for label, dtype in frame.dtypes.items() if pt.is_numeric_dtype(dtype)}
    return [
        {
            "name": name,
            "kind": "codes_lost",
            "leading_zero": record["values"],
            "checked": record["checked"],
            "width": record.get("width"),
        }
        for name, record in dataset.provenance.leading_zeros.items()
        if name in numeric
    ]


def _numbers_as_text(name: str, values: pd.Series) -> dict | None:
    if len(values) < MIN_VALUES:
        return None
    text = values.str.strip()
    parses = text.str.fullmatch(_NUMBER) | text.str.fullmatch(_THOUSANDS)
    share = float(parses.mean())
    if share < MIN_NUMERIC_SHARE:
        return None

    leading_zero = int(text[parses].str.match(_LEADING_ZERO).sum())
    others = text[~parses]
    counts = others.value_counts()
    short = counts[counts.index.str.len() <= MAX_TOKEN_LENGTH]
    ranked = sorted(short.items(), key=lambda item: (-item[1], item[0]))[:LISTED_TOKENS]
    return {
        "name": name,
        "kind": "numeric_codes" if leading_zero else "numbers_as_text",
        "non_null": len(values),
        "numeric_share": share,
        "non_numeric": int(others.size),
        "leading_zero": leading_zero,
        "tokens": [{"token": str(token), "count": int(count)} for token, count in ranked],
        "tokens_not_listed": int(others.size - sum(count for _, count in ranked)),
    }


def _type_name(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    return type(value).__name__


def _mixed_types(name: str, values: pd.Series) -> dict | None:
    counts: dict[str, int] = {}
    for value in values:
        kind = _type_name(value)
        counts[kind] = counts.get(kind, 0) + 1
    if len(counts) < 2:
        return None
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return {"name": name, "kind": "mixed_types", "non_null": len(values), "types": ordered}
