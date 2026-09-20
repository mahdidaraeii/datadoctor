"""Schema detection: dtype, inferred semantic type, cardinality and null count per column.

Semantic types are heuristics. They are inferred from the values and never change them: a
column of ``"yes"``/``"no"`` strings stays categorical and numbers stored as text stay text.
Identifier detection is the one heuristic reported as a ``Finding``, with a confidence below 1.
"""

import re
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
from pandas.api import types as pt

from datadoctor.core.dataset import Dataset
from datadoctor.core.result import AnalysisResult, Finding, Severity

# Below this many non-null values, uniqueness says nothing: everything looks unique.
MIN_ROWS_FOR_UNIQUENESS = 20
# A column is "unique" when at least this share of its non-null values are distinct.
UNIQUE_RATIO = 0.95
# Strings with at most this many distinct values, or with a distinct share at or below the
# ratio, are categorical. Other strings are text.
CATEGORICAL_MAX_DISTINCT = 20
CATEGORICAL_MAX_RATIO = 0.5
# Identifier confidence tiers. A column is typed as an identifier from this confidence up.
CONFIDENCE_NAMED_AND_UNIQUE = 0.95
CONFIDENCE_UNIQUE_ONLY = 0.6
CONFIDENCE_NAMED_ONLY = 0.4
IDENTIFIER_MIN_CONFIDENCE = 0.6

_ID_NAME = re.compile(r"(?:^|[^a-z0-9])(?:id|uuid|guid|key)$", re.IGNORECASE)
_CAMEL_ID_NAME = re.compile(r"[a-z0-9](?:Id|ID)$")
_ISO_8601 = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.\d+)?)?"
    r"(?:Z|[+-](?P<offset_hour>\d{2}):?(?P<offset_minute>\d{2}))?)?"
)
# The largest value each clock part of an ISO 8601 text can have. A second of 60 is rejected too.
_TIME_LIMITS = {"hour": 23, "minute": 59, "second": 59, "offset_hour": 23, "offset_minute": 59}

_NUMERIC_KINDS = {"integer", "floating", "mixed-integer-float"}
_DATETIME_KINDS = {"datetime64", "datetime", "date"}
_CLASSIFIABLE_KINDS = {"boolean", "categorical", "string"} | _DATETIME_KINDS | _NUMERIC_KINDS


class SemanticType(str, Enum):
    """What a column appears to hold, inferred from its values."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class _Identifier(NamedTuple):
    confidence: float
    named: bool
    sequential: bool


def profile_schema(dataset: Dataset) -> AnalysisResult:
    """Describe every column of a dataset.

    ``metrics`` holds ``n_rows``, ``n_columns`` and ``columns``, a list in file order. Each entry
    has ``name`` (non-string labels are converted to text), ``dtype`` as pandas reports it,
    ``semantic_type``, ``cardinality`` (distinct non-null values, ``None`` when it cannot be
    counted), ``null_count`` and ``identifier_confidence`` (``None`` when the column is not an
    identifier candidate).

    Semantic types, tried in this order on the non-null values:

    - ``unknown``: no values, or mixed or nested values such as lists and dicts.
    - ``boolean``: real booleans only. Text such as ``"yes"``/``"no"`` and integer 0/1 columns
      are not reinterpreted.
    - ``datetime``: datetime values, or text where every value is a valid ISO 8601 date or
      datetime. Ambiguous forms such as ``03/04/2024`` are never guessed.
    - ``numeric``: numbers. Numbers stored as text stay text.
    - ``categorical``: the ``category`` dtype, or strings with few distinct values.
    - ``text``: other strings.

    A column that looks like an identifier is reported as a ``Finding`` with a confidence, and
    is typed ``identifier`` when the confidence reaches ``IDENTIFIER_MIN_CONFIDENCE``. Only
    strings and whole numbers are candidates, and the target column never is.

    Args:
        dataset: The dataset to describe.

    Returns:
        An ``AnalysisResult`` with the column metrics and the identifier findings.
    """
    frame = dataset.data
    columns = []
    findings = []
    for position, label in enumerate(frame.columns):
        series = frame.iloc[:, position]
        name = str(label)
        is_target = dataset.target is not None and label == dataset.target
        semantic_type, cardinality, identifier = _classify(series, name, is_target)
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "semantic_type": semantic_type.value,
                "cardinality": cardinality,
                "null_count": int(series.isna().sum()),
                "identifier_confidence": None if identifier is None else identifier.confidence,
            }
        )
        if identifier is not None:
            n_non_null = int(series.notna().sum())
            findings.append(_identifier_finding(name, identifier, cardinality, n_non_null))

    metrics = {"n_rows": len(frame), "n_columns": frame.shape[1], "columns": columns}
    return AnalysisResult(findings=findings, metrics=metrics)


def _classify(
    series: pd.Series, name: str, is_target: bool
) -> tuple[SemanticType, int | None, _Identifier | None]:
    non_null = series.dropna()
    if non_null.empty:
        return SemanticType.UNKNOWN, 0, None

    kind = pt.infer_dtype(non_null, skipna=True)
    if kind not in _CLASSIFIABLE_KINDS:
        return SemanticType.UNKNOWN, _try_cardinality(non_null), None

    cardinality = int(non_null.nunique())
    if kind == "boolean":
        return SemanticType.BOOLEAN, cardinality, None
    if kind in _DATETIME_KINDS:
        return SemanticType.DATETIME, cardinality, None
    if kind == "categorical":
        return SemanticType.CATEGORICAL, cardinality, None

    if kind == "string":
        if _is_iso_8601(non_null):
            return SemanticType.DATETIME, cardinality, None
        identifier = None if is_target else _string_identifier(name, non_null, cardinality)
        base_type = _categorical_or_text(len(non_null), cardinality)
    else:
        identifier = None if is_target else _number_identifier(name, non_null, kind, cardinality)
        base_type = SemanticType.NUMERIC

    if identifier is not None and identifier.confidence >= IDENTIFIER_MIN_CONFIDENCE:
        return SemanticType.IDENTIFIER, cardinality, identifier
    return base_type, cardinality, identifier


def _try_cardinality(non_null: pd.Series) -> int | None:
    try:
        return int(non_null.nunique())
    except TypeError:  # unhashable values such as lists or dicts
        return None


def _is_iso_8601(non_null: pd.Series) -> bool:
    # The pattern gate comes first because a parser alone accepts year-like text such as "1234".
    if not non_null.str.fullmatch(_ISO_8601).all():
        return False
    # The date and the time are checked here and not with pandas. pandas 2.x holds dates as
    # datetime64[ns] and turns anything outside roughly 1677 to 2262, such as 9999-12-31 or
    # 0001-01-01, into NaT, so the same column would be typed differently on pandas 2.x and 3.x.
    # numpy's day-resolution dates cover every year and reject dates that do not exist.
    try:
        np.array(non_null.str[:10], dtype="datetime64[D]")
    except ValueError:
        return False
    times = non_null.str.extract(_ISO_8601)
    return not any(
        (pd.to_numeric(times[part]) > limit).any() for part, limit in _TIME_LIMITS.items()
    )


def _categorical_or_text(n_non_null: int, cardinality: int) -> SemanticType:
    if cardinality <= CATEGORICAL_MAX_DISTINCT or cardinality / n_non_null <= CATEGORICAL_MAX_RATIO:
        return SemanticType.CATEGORICAL
    return SemanticType.TEXT


def _has_identifier_name(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID_NAME.search(name))


def _is_unique(n_non_null: int, cardinality: int) -> bool:
    return n_non_null >= MIN_ROWS_FOR_UNIQUENESS and cardinality / n_non_null >= UNIQUE_RATIO


def _number_identifier(
    name: str, non_null: pd.Series, kind: str, cardinality: int
) -> _Identifier | None:
    # Only whole numbers can identify rows. A real-valued measure is never an identifier.
    if kind != "integer" and not bool((non_null % 1 == 0).all()):
        return None
    n_non_null = len(non_null)
    unique = _is_unique(n_non_null, cardinality)
    span = int(non_null.max()) - int(non_null.min()) + 1
    sequential = unique and cardinality == n_non_null and span == n_non_null
    return _identifier(_has_identifier_name(name), unique, sequential, qualifies_unnamed=sequential)


def _string_identifier(name: str, non_null: pd.Series, cardinality: int) -> _Identifier | None:
    unique = _is_unique(len(non_null), cardinality)
    compact = unique and not non_null.str.contains(r"\s").any()
    return _identifier(_has_identifier_name(name), unique, False, qualifies_unnamed=compact)


def _identifier(
    named: bool, unique: bool, sequential: bool, *, qualifies_unnamed: bool
) -> _Identifier | None:
    if named and unique:
        return _Identifier(CONFIDENCE_NAMED_AND_UNIQUE, named, sequential)
    if named:
        return _Identifier(CONFIDENCE_NAMED_ONLY, named, sequential)
    if qualifies_unnamed:
        return _Identifier(CONFIDENCE_UNIQUE_ONLY, named, sequential)
    return None


def _identifier_finding(
    name: str, identifier: _Identifier, cardinality: int | None, n_non_null: int
) -> Finding:
    facts = [f"{cardinality} distinct values in {n_non_null} non-null rows"]
    if n_non_null < MIN_ROWS_FOR_UNIQUENESS:
        facts.append("too few rows to judge uniqueness")
    if identifier.named:
        facts.append("its name matches an identifier pattern")
    if identifier.sequential:
        facts.append("the values are consecutive integers")
    elif not identifier.named:
        facts.append("no value contains whitespace")

    if identifier.confidence == CONFIDENCE_NAMED_ONLY:
        interpretation = (
            "The name suggests an identifier, but the values repeat, so this may be a foreign "
            "key or grouping key rather than a row identifier. This is a heuristic."
        )
        recommendation = (
            "Confirm whether the column is a key. Exclude it from model features unless its "
            "values carry meaning."
        )
    else:
        interpretation = (
            "The column appears to identify rows rather than describe them, so it likely "
            "carries no predictive signal and can let a model memorize rows. This is a "
            "heuristic based on uniqueness and the column name."
        )
        recommendation = (
            "Exclude the column from model features unless it is known to be meaningful, and "
            "check it for duplicate values."
        )

    return Finding(
        category="schema",
        severity=Severity.LOW,
        confidence=identifier.confidence,
        title="Possible identifier column",
        evidence=f"Column {name}: " + "; ".join(facts) + ".",
        interpretation=interpretation,
        limitations=(
            "Uniqueness cannot separate an identifier from a measurement that happens to be "
            "unique, and the name patterns are English-centric and incomplete. The column's "
            "meaning was not checked against any documentation."
        ),
        affected_columns=(name,),
        recommendation=recommendation,
    )
