"""Impossible values: negative ages and quantities, and dates in the future.

Which columns to check is decided from the column name, which is a heuristic, and every finding
says so. The rules are deliberately few and conservative, to keep false alarms rare.
"""

import datetime
import re

import numpy as np
from pandas.api import types as pt

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import impossible_findings as report

# Name tokens, matched as whole words after splitting on non-letters and on camelCase, so that
# "customer_age" and "AgeYears" match but "page", "stage" and "average" do not.
AGE_TOKENS = frozenset({"age", "ages"})
AGE_MAXIMUM = 120
# Quantities that cannot be negative. Amounts, prices, quantities and balances are left out on
# purpose, because refunds, returns and debts make a negative value legitimate there.
NON_NEGATIVE_TOKENS = frozenset(
    {"height", "weight", "length", "width", "depth", "duration", "distance", "population"}
    | {"count", "counts"}
)
# Date columns whose names suggest that a later date is expected are not checked.
FUTURE_OK_TOKENS = frozenset(
    {"due", "expire", "expires", "expiry", "expiration", "end", "deadline", "next", "renew"}
    | {"renewal", "scheduled", "delivery", "forecast", "planned", "until", "expected", "eta"}
)

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", re.DOTALL)
# How many values are tested first when deciding whether a text column holds dates.
_PROBE = 1000
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", _CAMEL.sub(" ", name).lower()))


def check_impossible_values(
    dataset: Dataset, config: AnalysisConfig, *, as_of: datetime.date | None = None
) -> AnalysisResult:
    """Find negative ages and quantities, and dates later than a reference date.

    **Ages.** A numeric column whose name contains the word ``age`` is checked for negative values,
    which are impossible, and for values above 120, which are implausible.

    **Other quantities.** A numeric column named for a length, weight, duration, count and the like
    is checked for negative values. See ``NON_NEGATIVE_TOKENS``.

    **Future dates.** A column of datetimes, dates, or ISO text dates (``YYYY-MM-DD``, optionally
    followed by a time) is checked for dates later than ``as_of``, unless its name suggests that
    later dates are expected. Dates are compared as calendar dates, ignoring the time of day, and
    timezone-aware values are converted to UTC first. ISO text is parsed with numpy, not pandas,
    so a date such as ``9999-12-31`` behaves the same on every supported pandas version. A text
    column is a date column only if every non-null value is a valid ISO date.

    ``as_of`` defaults to the date in ``dataset.provenance.loaded_at``, never to the wall clock, so
    the same dataset gives the same answer whenever the analysis is run. It is recorded in the
    metrics.

    Every row is used, so the size guardrails of ``config`` do not apply. ``config`` is recorded.
    The findings give counts and never the offending values; the lowest and highest are in the
    metrics.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.
        as_of: The reference date for "the future". Defaults to the dataset's load date.

    Returns:
        ``metrics`` with one entry per checked column and rule, and the findings, with ``config``
        recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for impossible values in a dataset with no rows")
    reference = _reference_date(dataset, as_of)

    schema = profile_schema(dataset).metrics["columns"]
    checks: list[dict] = []
    for position in range(n_columns):
        name = str(frame.columns[position])
        tokens = _tokens(name)
        series = frame.iloc[:, position]
        if schema[position]["semantic_type"] == "numeric":
            if tokens & AGE_TOKENS:
                checks.append(_age(name, series))
            elif tokens & NON_NEGATIVE_TOKENS:
                checks.append(_non_negative(name, series))
        if not tokens & FUTURE_OK_TOKENS:
            future = _future(name, series, reference)
            if future is not None:
                checks.append(future)

    findings: list[Finding] = []
    for check in checks:
        findings.extend(report.for_check(check))

    metrics = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "as_of": reference.isoformat(),
        "columns": checks,
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _reference_date(dataset: Dataset, as_of: datetime.date | None) -> datetime.date:
    if as_of is None:
        return datetime.date.fromisoformat(dataset.provenance.loaded_at[:10])
    if isinstance(as_of, datetime.datetime):
        return as_of.date()
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be a date, got {type(as_of).__name__}")
    return as_of


def _finite(series) -> np.ndarray:
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return values[np.isfinite(values)]


def _age(name: str, series) -> dict:
    x = _finite(series)
    negative, too_high = x < 0, x > AGE_MAXIMUM
    return {
        "name": name,
        "rule": "age",
        "checked": int(x.size),
        "negative": int(negative.sum()),
        "above_maximum": int(too_high.sum()),
        "rate": float(negative.sum() / x.size) if x.size else 0.0,
        "above_maximum_rate": float(too_high.sum() / x.size) if x.size else 0.0,
        "min": float(x.min()) if x.size else None,
        "max": float(x.max()) if x.size else None,
    }


def _non_negative(name: str, series) -> dict:
    x = _finite(series)
    negative = x < 0
    return {
        "name": name,
        "rule": "non_negative",
        "checked": int(x.size),
        "negative": int(negative.sum()),
        "rate": float(negative.sum() / x.size) if x.size else 0.0,
        "min": float(x.min()) if x.size else None,
    }


def _future(name: str, series, reference: datetime.date) -> dict | None:
    values = series.dropna()
    parsed = _as_days(values)
    if parsed is None:
        return None
    days, invalid = parsed
    total = int(days.size + invalid)
    later = days > np.datetime64(reference, "D")
    return {
        "name": name,
        "rule": "future_date",
        "checked": total,
        "future": int(later.sum()),
        "rate": float(later.sum() / total),
        "invalid": invalid,
        "invalid_rate": float(invalid / total),
        "as_of": reference.isoformat(),
        "latest": str(days.max()) if days.size else None,
    }


def _as_days(values) -> tuple[np.ndarray, int] | None:
    """Calendar dates of a column as numpy days, and how many values are not real dates.

    Returns ``None`` if the column is not a date column. A text column is one only if every
    non-null value is shaped like an ISO date. Values with that shape that are not real calendar
    dates, such as 2024-02-30, are counted as invalid and left out of the days.
    """
    if values.empty:
        return None
    kind = pt.infer_dtype(values, skipna=True)
    if kind == "datetime64":
        if values.dt.tz is not None:
            values = values.dt.tz_convert("UTC").dt.tz_localize(None)
        return values.to_numpy().astype("datetime64[D]"), 0
    if kind in ("datetime", "date"):
        return np.array([_to_date(v) for v in values], dtype="datetime64[D]"), 0
    if kind != "string":
        return None
    text = values.str.strip()
    # A date column needs every value to look like a date. Testing the first few first is exact,
    # since one miss settles it, and it spares the full scan for columns that are plainly not dates.
    if not text.iloc[:_PROBE].str.fullmatch(_ISO_DATE).all():
        return None
    if not text.str.fullmatch(_ISO_DATE).all():
        return None
    prefix = text.str[:10]
    try:
        return np.array(prefix, dtype="datetime64[D]"), 0
    except ValueError:  # at least one value is not a real date: find which
        return _split_valid(prefix)


def _split_valid(prefix) -> tuple[np.ndarray, int]:
    valid = {}
    for value in prefix.unique():
        try:
            valid[value] = np.datetime64(value, "D")
        except ValueError:
            valid[value] = None
    days = [valid[value] for value in prefix if valid[value] is not None]
    return np.array(days, dtype="datetime64[D]"), len(prefix) - len(days)


def _to_date(value) -> datetime.date:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc)
        return value.date()
    return value
