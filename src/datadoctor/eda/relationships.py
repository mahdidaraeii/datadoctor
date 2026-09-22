"""Relationship exploration: correlation between columns, and each feature's tie to the target.

A correlation matrix and heatmap cover the numeric columns. When the dataset has a usable target
(numeric, so a regression target, or categorical/boolean, so a classification one), every other
numeric or categorical column is compared against it, and the target's own distribution — the
class balance, for a classification target — is drawn and summarized.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.guardrails import apply_guardrails
from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.eda import relationships_findings as report
from datadoctor.eda import relationships_plots as plots
from datadoctor.eda import relationships_stats as stat
from datadoctor.privacy import check_pii
from datadoctor.profiling.schema import profile_schema

# An association with the target is reported only from this effect size up: a conventional
# "small" effect, and Pearson correlation, the correlation ratio and Cramer's V all share this
# 0 to 1 scale for their absolute value.
MIN_EFFECT = 0.1
ALPHA = 0.05
# The target's most frequent classes are drawn and listed by name; the rest are pooled.
TOP_CLASSES = 10
# A numeric target with at most this many distinct whole numbers gets the class-balance view
# rather than a histogram, the same rule the univariate histograms use for a discrete numeric
# column such as a 0/1 target stored as numbers rather than booleans.
DISCRETE_MAX_DISTINCT = 15
# Minority-class share below the first value is graded MEDIUM, below the second LOW, and 10% or
# more is not reported: most real classification targets are imbalanced past that point.
IMBALANCE_MEDIUM = 0.05
IMBALANCE_LOW = 0.10

_TASKS = {"numeric": "regression", "categorical": "classification", "boolean": "classification"}
_ASSOCIATION_TYPES = {"numeric", "categorical", "boolean"}


def check_relationships(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Correlate the numeric columns, and compare every feature against the target.

    **Correlation matrix.** Pearson correlation between every pair of numeric columns, apart from
    the target. It is skipped above ``config.column_threshold``, per the column guardrail, and
    computed on the rows chosen by the row guardrail otherwise. Pairs at or above
    ``relationships_stats.MIN_ABS_CORRELATION`` are reported as a finding, a candidate for
    redundancy in a model. Saved as ``correlation_heatmap``.

    **Feature-target relationships.** Usable when the target's schema type is ``numeric``
    (regression) or ``categorical``/``boolean`` (classification); anything else, or no target at
    all, skips this part. Every other numeric or categorical column is compared against it, on
    the row guardrail's sample: Pearson correlation between two numeric columns, the correlation
    ratio (the square root of eta-squared, from a one-way ANOVA) between a numeric column and a
    categorical one, and Cramer's V between two categorical columns. A categorical column with
    more than ``relationships_stats.MAX_CATEGORIES`` distinct values is not tested. p-values are
    Bonferroni-adjusted over the comparisons that ran. The strongest, past ``MIN_EFFECT`` and
    significant at ``ALPHA``, are reported as a finding.

    **Class balance.** For a categorical or boolean target, or a numeric one with at most
    ``DISCRETE_MAX_DISTINCT`` distinct whole numbers (a 0/1-coded classification target, for
    instance), the count and share of every class, most frequent first, computed on every row.
    This is independent of ``task``: such a target's associations above are still Pearson
    correlations, a point-biserial correlation against a 0/1 outcome being a legitimate use of
    one, only its own display is classification-style. A minority class under ``IMBALANCE_LOW``
    is reported, at MEDIUM below ``IMBALANCE_MEDIUM`` and LOW otherwise: imbalance is expected for
    many real problems and is not a defect by itself. For every other numeric target, the
    distribution instead: count, mean, standard deviation, quartiles and the ``range_shown`` a
    histogram of it would cover, the same wide-fence rule the univariate histograms use (``None``
    when nothing lies beyond the fences). Saved as ``target_distribution``.

    **Privacy.** A column the privacy check flags as possibly holding personal data keeps its
    counts but not its labels, in the class balance and wherever else its values would otherwise
    be shown, the same rule the univariate module uses.

    Identifiers, free text, dates and columns of mixed or no values take no part in any of this,
    and are named in a note.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded, and ``output_dir`` says where the
            figures go.

    Returns:
        The metrics and findings above, with the figures under ``artifacts`` and ``config``
        recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for relationships in a dataset with no rows")
    guard = apply_guardrails(dataset, config)
    sample = guard.frame

    schema = profile_schema(dataset).metrics["columns"]
    kinds = [column["semantic_type"] for column in schema]
    names = [column["name"] for column in schema]
    personal = {d["name"] for d in check_pii(dataset, config).metrics["columns"]}
    target_position = None if dataset.target is None else list(frame.columns).index(dataset.target)

    findings: list[Finding] = list(guard.findings)
    artifacts: dict[str, Any] = {}

    numeric_positions = [
        i for i, kind in enumerate(kinds) if kind == "numeric" and i != target_position
    ]

    correlation_metrics = _correlation(
        sample, names, numeric_positions, guard.pairwise, config, artifacts
    )
    if correlation_metrics["pairs"]:
        findings.append(
            report.correlated_columns_finding(
                [(p["a"], p["b"], p["r"]) for p in correlation_metrics["pairs"]],
                stat.MIN_ABS_CORRELATION,
            )
        )

    target_metrics, target_findings = _target(
        dataset, target_position, kinds, names, sample, personal, config, artifacts
    )
    findings.extend(target_findings)

    not_used = _not_used(names, kinds, target_position)
    if not_used:
        findings.append(report.not_used_finding(not_used))

    metrics = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "correlation": correlation_metrics,
        "target": target_metrics,
    }
    return AnalysisResult(findings=findings, metrics=metrics, artifacts=artifacts, config=config)


def _correlation(
    sample: pd.DataFrame,
    names: list[str],
    numeric_positions: list[int],
    pairwise: bool,
    config: AnalysisConfig,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    if not pairwise or len(numeric_positions) < 2:
        return {"computed": False, "columns": [], "matrix": None, "pairs": []}

    numeric_names = [names[i] for i in numeric_positions]
    matrix = stat.correlation_matrix(sample.iloc[:, numeric_positions])
    pairs = stat.correlated_pairs(numeric_names, matrix)
    artifacts["correlation_heatmap"] = plots.save_heatmap(numeric_names, matrix, config)
    return {
        "computed": True,
        "columns": numeric_names,
        "matrix": [[None if np.isnan(v) else float(v) for v in row] for row in matrix],
        "pairs": [{"a": a, "b": b, "r": r} for a, b, r in pairs],
    }


def _target(
    dataset: Dataset,
    target_position: int | None,
    kinds: list[str],
    names: list[str],
    sample: pd.DataFrame,
    personal: set[str],
    config: AnalysisConfig,
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[Finding]]:
    if target_position is None:
        return None, []
    kind = kinds[target_position]
    name = names[target_position]
    if kind not in _TASKS:
        return {"name": name, "usable": False, "reason": kind}, [
            report.target_unusable_finding(name, kind)
        ]

    task = _TASKS[kind]
    associations, not_tested = _associations(sample, target_position, task, kinds, names)
    findings: list[Finding] = []
    detected = [
        a for a in associations if abs(a["effect"]) >= MIN_EFFECT and a["p_adjusted"] < ALPHA
    ]
    if detected:
        detected.sort(key=lambda a: -abs(a["effect"]))
        findings.append(report.top_predictors_finding(detected))
    by_reason: dict[str, list[str]] = {}
    for entry in not_tested:
        by_reason.setdefault(entry["reason"], []).append(entry["name"])
    if by_reason:
        findings.append(report.not_tested_finding(by_reason))

    frame = dataset.data
    target_series = frame.iloc[:, target_position]
    hidden = name in personal
    distribution = None
    class_balance = None
    # The class-balance view is not tied to ``task``: a 0/1-coded classification target is typed
    # numeric (schema never reinterprets a whole-number column as boolean), and Pearson is still a
    # legitimate association for it, but it should still get the classification-style view, the
    # same "whole number, at most DISCRETE_MAX_DISTINCT distinct values" convention the univariate
    # histograms use for a discrete numeric column.
    if kind in ("categorical", "boolean") or _is_discrete_numeric(target_series):
        class_balance, drawing = _class_balance(target_series, name, hidden)
        artifacts["target_distribution"] = plots.save_target_distribution(
            "categorical", drawing, config
        )
        severity = _imbalance_severity(class_balance["minority_share"])
        if severity is not None:
            findings.append(
                report.class_imbalance_finding(
                    class_balance["minority_label"], class_balance["minority_share"], severity
                )
            )
    else:
        distribution, drawing = _numeric_target(target_series, name)
        if drawing is not None:
            artifacts["target_distribution"] = plots.save_target_distribution(
                "numeric", drawing, config
            )

    target_metrics = {
        "name": name,
        "usable": True,
        "task": task,
        "associations": sorted(associations, key=lambda a: -abs(a["effect"])),
        "not_tested": not_tested,
        "distribution": distribution,
        "class_balance": class_balance,
    }
    return target_metrics, findings


def _imbalance_severity(minority_share: float) -> Severity | None:
    if minority_share < IMBALANCE_MEDIUM:
        return Severity.MEDIUM
    if minority_share < IMBALANCE_LOW:
        return Severity.LOW
    return None


def _associations(
    sample: pd.DataFrame, target_position: int, task: str, kinds: list[str], names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    target_series = sample.iloc[:, target_position]
    records: list[tuple[str, str, float, float]] = []  # name, kind, effect, p
    not_tested: list[dict[str, str]] = []

    if task == "regression":
        target_values = target_series.to_numpy(dtype="float64", na_value=np.nan)
    else:
        target_codes, _ = pd.factorize(target_series)

    for position, name in enumerate(names):
        if position == target_position or kinds[position] not in _ASSOCIATION_TYPES:
            continue
        if kinds[position] == "numeric":
            feature = sample.iloc[:, position].to_numpy(dtype="float64", na_value=np.nan)
            if task == "regression":
                pearson = _pearson(feature, target_values)
                if pearson is None:
                    not_tested.append({"name": name, "reason": "not_enough_data"})
                else:
                    records.append((name, "pearson", pearson[0], pearson[1]))
            else:
                mask = np.isfinite(feature) & (target_codes >= 0)
                result = stat.eta_squared(feature[mask], target_codes[mask])
                if result is None:
                    not_tested.append({"name": name, "reason": "not_enough_data"})
                else:
                    records.append((name, "eta_squared", float(np.sqrt(result.eta2)), result.p))
        else:
            distinct = sample.iloc[:, position].nunique(dropna=True)
            if distinct > stat.MAX_CATEGORIES:
                not_tested.append({"name": name, "reason": "too_many_categories"})
                continue
            codes, _ = pd.factorize(sample.iloc[:, position])
            if task == "regression":
                mask = (codes >= 0) & np.isfinite(target_values)
                result = stat.eta_squared(target_values[mask], codes[mask])
                if result is None:
                    not_tested.append({"name": name, "reason": "not_enough_data"})
                else:
                    records.append((name, "eta_squared", float(np.sqrt(result.eta2)), result.p))
            else:
                mask = (codes >= 0) & (target_codes >= 0)
                result = stat.cramers_v(codes[mask], target_codes[mask])
                if result is None:
                    not_tested.append({"name": name, "reason": "not_enough_data"})
                else:
                    records.append((name, "cramers_v", result.v, result.p))

    tests_run = len(records)
    associations = [
        {"name": name, "kind": kind, "effect": effect, "p_adjusted": min(1.0, p * tests_run)}
        for name, kind, effect, p in records
    ]
    return associations, not_tested


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """A feature's Pearson correlation and p-value against the target, or ``None`` if unreliable.

    ``None`` below ``stat.MIN_PAIRS`` rows with both values present, or if either side has no
    variance there, since the correlation is undefined.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < stat.MIN_PAIRS:
        return None
    x, y = a[mask], b[mask]
    if x.std() == 0 or y.std() == 0:
        return None
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _is_discrete_numeric(series: pd.Series) -> bool:
    """Whether a numeric target has few enough distinct whole numbers to show as class balance.

    The same convention the univariate histograms use for a discrete numeric column, such as a
    0/1-coded classification target: whole numbers, at most ``DISCRETE_MAX_DISTINCT`` of them.
    ``False`` when there is no finite data to judge.
    """
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    whole = bool(np.all(finite % 1 == 0))
    return whole and len(np.unique(finite)) <= DISCRETE_MAX_DISTINCT


def _numeric_target(series: pd.Series, name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """A regression target's own distribution: summary statistics and its histogram range.

    Called only once ``_is_discrete_numeric`` has ruled out the class-balance view, so the target
    always draws as a continuous histogram here, never one bar per value.
    """
    every = series.to_numpy(dtype="float64", na_value=np.nan)
    missing = int(np.isnan(every).sum())
    finite = every[np.isfinite(every)]
    infinite = int(every.size - missing - finite.size)
    if finite.size == 0:
        return {"count": 0, "missing": missing, "infinite": infinite, "range_shown": None}, None
    q1, median, q3 = (float(q) for q in np.quantile(finite, [0.25, 0.5, 0.75]))
    range_shown = stat.wide_fence_range(finite, q1, q3)
    distribution = {
        "count": int(finite.size),
        "missing": missing,
        "infinite": infinite,
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(finite.max()),
        "range_shown": range_shown,
    }
    drawing = {
        "name": name,
        "discrete": None,
        "values": finite,
        "range_shown": range_shown,
        "count": int(finite.size),
        "missing": missing,
        "infinite": infinite,
    }
    return distribution, drawing


def _class_balance(
    series: pd.Series, name: str, hidden: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = series.dropna()
    counts = values.value_counts(sort=False)
    counts.index = counts.index.astype(str)
    ranked = counts.sort_index().sort_values(ascending=False, kind="stable")
    total = int(values.size)
    top = [(label, int(count)) for label, count in ranked.iloc[:TOP_CLASSES].items()]
    other_count = int(ranked.iloc[TOP_CLASSES:].sum())
    for_drawing = [
        {"count": count} if hidden else {"label": label, "count": count} for label, count in top
    ]
    classes = [
        entry | {"share": count / total} for entry, (_, count) in zip(for_drawing, top, strict=True)
    ]
    minority_label, minority_count = min(ranked.items(), key=lambda item: item[1])
    class_balance = {
        "classes": classes,
        "other_count": other_count,
        "minority_label": None if hidden else str(minority_label),
        "minority_share": minority_count / total,
    }
    drawing = {
        "name": name,
        "top": for_drawing,
        "other_count": other_count,
        "other_distinct": len(ranked) - TOP_CLASSES,
        "labels_hidden": hidden,
        "count": total,
        "missing": int(series.isna().sum()),
    }
    return class_balance, drawing


def _not_used(
    names: list[str], kinds: list[str], target_position: int | None
) -> dict[str, list[str]]:
    by_reason: dict[str, list[str]] = {}
    for position, (name, kind) in enumerate(zip(names, kinds, strict=True)):
        if position == target_position or kind in _ASSOCIATION_TYPES:
            continue
        by_reason.setdefault(kind, []).append(name)
    return by_reason
