"""Univariate exploration: the distribution of each column on its own, drawn and summarized.

Numeric columns are drawn as histograms and categorical columns as frequency bars, one figure of
small plots for each kind of column. Every column gets summary statistics in the metrics.
"""

from typing import Any

import numpy as np
import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.eda import univariate_findings as report
from datadoctor.eda import univariate_plots as plots
from datadoctor.privacy import check_pii
from datadoctor.profiling.schema import profile_schema

# Figures per kind of column. More columns than fit are named in a note and kept in the metrics.
MAX_PAGES = 4
# Categorical columns draw this many of their most frequent values, and one bar for the rest.
TOP_CATEGORIES = 10
# A numeric column with at most this many distinct whole numbers is drawn as one bar per value.
DISCRETE_MAX_DISTINCT = 15
# A histogram covers this many interquartile ranges beyond the quartiles. This is the "wide"
# fence of the extreme values in the outlier check. Values further out are drawn as one bar each.
FENCE_IQRS = 3.0


def check_univariate(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Summarize and draw every column on its own.

    **Which columns.** The schema profile decides the kind. Numeric columns are drawn as
    histograms, whole numbers with at most ``DISCRETE_MAX_DISTINCT`` distinct values as one bar per
    value. Categorical and boolean columns are drawn as horizontal bars of their
    ``TOP_CATEGORIES`` most frequent values and one bar for the rest. Identifiers, free text, dates
    and columns of mixed or no values are not drawn, since a chart of them says nothing here. They
    still get statistics, and a note names them and the reason.

    **Figures.** Each kind of column is drawn on figures of 12 small plots, in file order, at most
    ``MAX_PAGES`` figures per kind. The files are ``univariate_<kind>_<page>.png`` in
    ``config.output_dir``, and their paths are in ``artifacts``. A few extreme values would squeeze
    the rest of a histogram into a few bars, so its range is from ``FENCE_IQRS`` interquartile
    ranges below the first quartile to as many above the third, cut back to the data. Values
    beyond it are not dropped: each side has one orange bar at the edge, labelled with how many
    values it holds and the true extreme. The range is in ``range_shown``, ``None`` when the
    histogram covers every value. Statistics always use every value. Missing and infinite values
    are left out of a histogram and counted under it.

    **Privacy.** Category labels are values from the data. For a column that the privacy check
    flags as possibly holding personal data, the bars and counts are drawn and stored, but the
    labels are left out of the figure and of the metrics.

    **Statistics.** ``metrics["columns"]`` has one entry per column with ``statistics``. Numeric
    columns get ``count`` (finite values), ``missing``, ``infinite``, ``distinct``, ``mean``, the
    sample ``std``, ``min``, the quartiles ``q1``, ``median`` and ``q3``, ``max`` and ``skew``.
    Categorical columns get ``count``, ``missing``, ``distinct`` and ``top``, the counts (and the
    labels, unless hidden) of the most frequent values. Every row is used, so the size guardrails
    of ``config`` do not apply. The dataset is never changed.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded, and ``output_dir`` says where the
            figures go.

    Returns:
        The metrics, INFO notes about what was not drawn, and the figures under ``artifacts``,
        with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot explore a dataset with no rows")

    schema = profile_schema(dataset).metrics["columns"]
    personal = {d["name"] for d in check_pii(dataset, config).metrics["columns"]}

    entries: list[dict[str, Any]] = []
    drawable: dict[str, list[dict[str, Any]]] = {"numeric": [], "categorical": []}
    for position, column in enumerate(schema):
        entry, drawing = _describe(frame.iloc[:, position], column, column["name"] in personal)
        entries.append(entry)
        if drawing is not None:
            drawable[drawing["kind"]].append(drawing | {"entry": entry})

    artifacts: dict[str, Any] = {}
    figures = []
    not_drawn: dict[str, list[str]] = {}
    for kind, columns in drawable.items():
        pages = [columns[i : i + plots.PER_PAGE] for i in range(0, len(columns), plots.PER_PAGE)]
        for number, page in enumerate(pages[:MAX_PAGES], start=1):
            key = f"univariate_{kind}_{number}"
            artifacts[key] = plots.save_page(kind, number, min(len(pages), MAX_PAGES), page, config)
            for column in page:
                column["entry"]["figure"] = key
                column["entry"]["plot"] = column["plot"]
            figures.append({"key": key, "kind": kind, "columns": [c["name"] for c in page]})
        for column in [c for page in pages[MAX_PAGES:] for c in page]:
            column["entry"]["skipped_reason"] = "page_limit"
            not_drawn.setdefault(kind, []).append(column["name"])

    findings = _notes(entries, not_drawn)
    metrics = {"n_rows": n_rows, "n_columns": n_columns, "columns": entries, "figures": figures}
    return AnalysisResult(findings=findings, metrics=metrics, artifacts=artifacts, config=config)


def _notes(entries: list[dict[str, Any]], not_drawn: dict[str, list[str]]) -> list[Finding]:
    hidden = [entry["name"] for entry in entries if entry["labels_hidden"]]
    by_reason: dict[str, list[str]] = {}
    for entry in entries:
        reason = entry["skipped_reason"]
        if reason is not None and reason != "page_limit":
            by_reason.setdefault(reason, []).append(entry["name"])
    findings = []
    if by_reason:
        findings.append(report.not_plotted_finding(by_reason))
    if not_drawn:
        findings.append(report.page_limit_finding(not_drawn, MAX_PAGES))
    if hidden:
        findings.append(report.labels_hidden_finding(hidden))
    return findings


def _describe(
    series: pd.Series, column: dict[str, Any], personal: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The metrics entry of a column, and what to draw for it, or ``None`` if nothing is drawn."""
    kind = column["semantic_type"]
    entry: dict[str, Any] = {
        "name": column["name"],
        "semantic_type": kind,
        "skipped_reason": None,
        "figure": None,
        "plot": None,
        "labels_hidden": False,
        "range_shown": None,
        "statistics": {},
    }
    if kind == "numeric":
        return _numeric(series, entry)
    if kind in ("categorical", "boolean"):
        return _categorical(series, entry, personal)
    entry["skipped_reason"] = kind  # identifier, text, datetime or unknown
    entry["statistics"] = _basic(series, column)
    if kind == "datetime":
        values = series.dropna()
        entry["statistics"] |= {"min": str(values.min()), "max": str(values.max())}
    return entry, None


def _basic(series: pd.Series, column: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "distinct": column["cardinality"],
    }


def _range_shown(finite: np.ndarray, q1: float, q3: float) -> dict[str, Any] | None:
    """The range a histogram covers, if a few extreme values would squeeze the rest.

    The range is from ``FENCE_IQRS`` interquartile ranges below the first quartile to as many
    above the third, cut back to the data where the data ends sooner. Values outside it are
    counted and drawn as one bar at each end. ``None`` when nothing lies outside, and when the
    interquartile range is zero, since the fences then have no width.
    """
    spread = q3 - q1
    if spread <= 0:
        return None
    low = max(float(finite.min()), q1 - FENCE_IQRS * spread)
    high = min(float(finite.max()), q3 + FENCE_IQRS * spread)
    below, above = int((finite < low).sum()), int((finite > high).sum())
    if not below and not above:
        return None
    return {"low": low, "high": high, "below": below, "above": above}


def _number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _numeric(series: pd.Series, entry: dict[str, Any]) -> tuple[dict[str, Any], dict | None]:
    every = series.to_numpy(dtype="float64", na_value=np.nan)
    missing = int(np.isnan(every).sum())
    finite = every[np.isfinite(every)]
    infinite = int(every.size - missing - finite.size)
    if finite.size == 0:
        entry["skipped_reason"] = "no_finite_values"
        entry["statistics"] = {"count": 0, "missing": missing, "infinite": infinite}
        return entry, None
    q1, median, q3 = (float(q) for q in np.quantile(finite, [0.25, 0.5, 0.75]))
    distinct, counts = np.unique(finite, return_counts=True)
    entry["statistics"] = {
        "count": int(finite.size),
        "missing": missing,
        "infinite": infinite,
        "distinct": int(distinct.size),
        "mean": float(finite.mean()),
        "std": _number(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(finite.max()),
        # Skewness needs three values that differ. pandas answers 0 or NaN for less, depending on
        # its version, so the answer is fixed here.
        "skew": _number(pd.Series(finite).skew()) if distinct.size >= 3 else None,
    }
    whole = bool(np.all(finite % 1 == 0))
    discrete = (distinct, counts) if whole and distinct.size <= DISCRETE_MAX_DISTINCT else None
    range_shown = None if discrete is not None else _range_shown(finite, q1, q3)
    entry["range_shown"] = range_shown
    drawing = {
        "kind": "numeric",
        "plot": "value_bars" if discrete is not None else "histogram",
        "name": entry["name"],
        "values": finite,
        "discrete": discrete,
        "range_shown": range_shown,
        "count": int(finite.size),
        "missing": missing,
        "infinite": infinite,
    }
    return entry, drawing


def _categorical(
    series: pd.Series, entry: dict[str, Any], personal: bool
) -> tuple[dict[str, Any], dict]:
    values = series.dropna()
    counts = values.value_counts(sort=False)
    counts.index = counts.index.astype(str)
    # Most frequent first, equal counts by label. Sorting by label and then stably by count does
    # that without visiting every distinct value in Python, which matters for a column that has
    # hundreds of thousands of them.
    ranked = counts.sort_index().sort_values(ascending=False, kind="stable")
    top = [(label, int(count)) for label, count in ranked.iloc[:TOP_CATEGORIES].items()]
    other_count = int(ranked.iloc[TOP_CATEGORIES:].sum())
    shown = [
        {"count": count} if personal else {"label": label, "count": count} for label, count in top
    ]
    entry["labels_hidden"] = personal
    entry["statistics"] = {
        "count": int(values.size),
        "missing": int(series.isna().sum()),
        "distinct": len(ranked),
        "top": shown,
        "other_count": other_count,
    }
    drawing = {
        "kind": "categorical",
        "plot": "category_bars",
        "name": entry["name"],
        "top": shown,
        "other_count": other_count,
        "other_distinct": len(ranked) - TOP_CATEGORIES,  # only drawn when there are more
        "labels_hidden": personal,
        "count": int(values.size),
        "missing": int(series.isna().sum()),
    }
    return entry, drawing
