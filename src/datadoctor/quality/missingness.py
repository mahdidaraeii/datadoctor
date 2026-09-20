"""Missing-value diagnostics: how much is missing, where, and whether it looks random."""

from typing import NamedTuple

import numpy as np
from pandas.api import types as pt

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.guardrails import Guardrails, apply_guardrails
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import dependence as stat
from datadoctor.quality import missingness_findings as report

_PREDICTOR_TYPES = {"numeric": "numeric", "categorical": "categorical", "boolean": "categorical"}


def check_missingness(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Describe the missing values of a dataset and look for structure in them.

    Rates use every row. The pairwise parts (correlation between columns, and whether a column's
    missingness depends on the values of others) run on the rows chosen by the size guardrails
    of ``config``, and are skipped when there are too many columns.

    ``metrics`` holds ``overall`` (cell and row counts and rates), ``columns`` (count, rate and the
    dependence label of each column: ``detected``, ``not_detected``, ``not_tested`` or
    ``no_missing``), ``pairwise``, ``pairs`` and ``dependence``.

    Findings, all with separate evidence, interpretation and limitations:

    - a severity-graded finding per column with a missing rate of 5% or more
      (``missingness_findings.GRADES``), and one for the share of rows with any missing value. A
      target column with any missing value is at least HIGH;
    - one finding when missingness is correlated between columns;
    - one per column whose missingness depends on other columns, which rules out "missing
      completely at random". It cannot tell "at random" from "not at random";
    - one INFO finding when some columns could not be tested, so silence is never read as
      "no dependence found". It covers columns whose missing rate is high enough to get a rate
      finding; every column still has its label in ``metrics``;
    - one finding per column where the token ``NA`` was read as missing but looks like a
      genuine 2 or 3 letter code, such as Namibia's country code;
    - the guardrail findings for whatever limit applied to the pairwise parts.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot assess missing values in a dataset with no rows")
    guard = apply_guardrails(dataset, config)

    names = [str(label) for label in frame.columns]
    types = [column["semantic_type"] for column in profile_schema(dataset).metrics["columns"]]
    target = None if dataset.target is None else list(frame.columns).index(dataset.target)
    mask = frame.isna()
    counts = [int(count) for count in mask.sum()]
    rows_with_missing = int(mask.any(axis=1).sum())

    pairwise = _pairwise(dataset, guard, types)
    labels = _labels(counts, pairwise)

    findings: list[Finding] = []
    for position, (name, count) in enumerate(zip(names, counts, strict=True)):
        finding = report.rate_finding(name, count, n_rows, position == target)
        if finding is not None:
            findings.append(finding)
    row_rate = rows_with_missing / n_rows
    if report.grade(row_rate) is not None:
        findings.append(report.rows_finding(rows_with_missing, n_rows, row_rate))

    findings.extend(_pair_findings(pairwise, names))
    findings.extend(_dependence_findings(pairwise, labels, names))
    findings.extend(_untested_findings(pairwise, labels, names, counts, n_rows))
    findings.extend(_token_findings(dataset, names))
    findings.extend(_guardrail_findings(guard, dataset))

    cells = n_rows * n_columns
    metrics = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "overall": {
            "cells": cells,
            "missing_cells": sum(counts),
            "cell_rate": sum(counts) / cells,
            "rows_with_missing": rows_with_missing,
            "row_rate": row_rate,
        },
        "columns": _column_metrics(names, counts, n_rows, labels, pairwise),
        "pairwise": {
            "performed": pairwise.performed,
            "rows_used": pairwise.rows_used,
            "sampled": pairwise.sampled,
        },
        "pairs": _pair_metrics(pairwise, names),
        "dependence": {"tests_run": pairwise.tests_run},
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


class _Pairwise(NamedTuple):
    performed: bool
    sampled: bool
    rows_used: int | None
    missing_in_sample: np.ndarray | None
    partial: list[int]
    index_of: dict[int, int]
    pairs: stat.PairSummary
    dependence: stat.DependenceSummary
    tests_run: int


class _Label(NamedTuple):
    label: str
    reason: str | None
    comparisons: int


def _pairwise(dataset: Dataset, guard: Guardrails, types: list[str]) -> _Pairwise:
    none = stat.PairSummary(0, 0, 0, [])
    empty = stat.DependenceSummary(0, [])
    if not guard.pairwise:
        return _Pairwise(False, False, None, None, [], {}, none, empty, 0)

    sample = guard.frame
    n = len(sample)
    missing = sample.isna().to_numpy()
    in_sample = missing.sum(axis=0)
    partial = [i for i in range(missing.shape[1]) if 0 < in_sample[i] < n]
    sampled = sample is not dataset.data
    index_of = {position: j for j, position in enumerate(partial)}
    if not partial:
        return _Pairwise(True, sampled, n, in_sample, partial, index_of, none, empty, 0)

    indicators = missing[:, partial].astype(np.float64)
    numeric = [i for i, kind in enumerate(types) if _PREDICTOR_TYPES.get(kind) == "numeric"]
    categorical = [i for i, kind in enumerate(types) if _PREDICTOR_TYPES.get(kind) == "categorical"]
    dependence = stat.dependence_tests(sample, indicators, partial, numeric, categorical)
    pairs = stat.correlated_pairs(indicators)
    return _Pairwise(
        True, sampled, n, in_sample, partial, index_of, pairs, dependence, dependence.tests_run
    )


def _labels(counts: list[int], pairwise: _Pairwise) -> list[_Label]:
    labels = []
    for position, count in enumerate(counts):
        if count == 0:
            labels.append(_Label("no_missing", None, 0))
        elif not pairwise.performed:
            labels.append(_Label("not_tested", "pairwise_skipped", 0))
        elif position not in pairwise.index_of:
            reason = "all_missing" if pairwise.missing_in_sample[position] else "none_in_sample"
            labels.append(_Label("not_tested", reason, 0))
        else:
            column = pairwise.dependence.columns[pairwise.index_of[position]]
            if column.detected:
                labels.append(_Label("detected", None, column.tested))
            elif column.tested:
                labels.append(_Label("not_detected", None, column.tested))
            else:
                labels.append(_Label("not_tested", _main_reason(column.skipped), 0))
    return labels


def _main_reason(skipped: dict[str, int]) -> str:
    if not skipped:
        return "no_predictors"
    order = [stat.SMALL_GROUP, stat.SMALL_EXPECTED, stat.NO_VARIATION, stat.TOO_MANY_CATEGORIES]
    return max(skipped, key=lambda reason: (skipped[reason], -order.index(reason)))


def _pair_findings(pairwise: _Pairwise, names: list[str]) -> list[Finding]:
    summary = pairwise.pairs
    if summary.above_threshold == 0:
        return []
    pairs = [
        (
            names[pairwise.partial[pair.a]],
            names[pairwise.partial[pair.b]],
            pair.phi,
            pair.n_both,
        )
        for pair in summary.listed
    ]
    return [
        report.pairs_finding(
            pairs,
            above_threshold=summary.above_threshold,
            tested=summary.tested,
            skipped=summary.skipped,
            sampled=pairwise.sampled,
        )
    ]


def _dependence_findings(
    pairwise: _Pairwise, labels: list[_Label], names: list[str]
) -> list[Finding]:
    findings = []
    for position, label in enumerate(labels):
        if label.label == "detected":
            column = pairwise.dependence.columns[pairwise.index_of[position]]
            findings.append(
                report.dependence_finding(
                    names[position],
                    column.detected,
                    comparisons=label.comparisons,
                    tests_run=pairwise.tests_run,
                    sampled=pairwise.sampled,
                )
            )
    return findings


def _untested_findings(
    pairwise: _Pairwise,
    labels: list[_Label],
    names: list[str],
    counts: list[int],
    n_rows: int,
) -> list[Finding]:
    # If pairwise work was skipped altogether, the guardrail finding already says so.
    if not pairwise.performed:
        return []
    # Only columns whose missing rate is high enough to get a rate finding are worth naming.
    # A column with a couple of missing values is expected to be untestable.
    considered = [i for i, count in enumerate(counts) if report.grade(count / n_rows) is not None]
    by_reason: dict[str, list[str]] = {}
    for position in considered:
        if labels[position].label == "not_tested":
            by_reason.setdefault(labels[position].reason, []).append(names[position])
    if not by_reason:
        return []
    return [report.untested_finding(by_reason, len(considered))]


def _token_findings(dataset: Dataset, names: list[str]) -> list[Finding]:
    findings = []
    tokens = dataset.provenance.converted_tokens
    for position, name in enumerate(names):
        count = tokens.get(name, {}).get("NA")
        if not count:
            continue
        values = dataset.data.iloc[:, position].dropna()
        if values.empty or pt.infer_dtype(values, skipna=True) != "string":
            continue
        if values.str.fullmatch(report.CODE_PATTERN).mean() >= report.CODE_SHARE:
            findings.append(report.token_finding(name, count))
    return findings


def _guardrail_findings(guard: Guardrails, dataset: Dataset) -> list[Finding]:
    # The sampling finding says analyses ran on a sample. If pairwise work was skipped, nothing
    # here used the sample, so only the pairwise finding applies. Rows come first (see S13).
    sampled = guard.frame is not dataset.data
    if sampled and not guard.pairwise:
        return list(guard.findings[1:])
    return list(guard.findings)


def _column_metrics(
    names: list[str],
    counts: list[int],
    n_rows: int,
    labels: list[_Label],
    pairwise: _Pairwise,
) -> list[dict]:
    columns = []
    for position, (name, count, label) in enumerate(zip(names, counts, labels, strict=True)):
        predictors = []
        if label.label in ("detected", "not_detected") and position in pairwise.index_of:
            found = pairwise.dependence.columns[pairwise.index_of[position]].detected
            predictors = [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "effect": p.effect,
                    "p_adjusted": p.p_adjusted,
                }
                for p in found[: report.LISTED]
            ]
        columns.append(
            {
                "name": name,
                "n_missing": count,
                "rate": count / n_rows,
                "dependence": label.label,
                "dependence_reason": label.reason,
                "comparisons_run": label.comparisons,
                "predictors": predictors,
            }
        )
    return columns


def _pair_metrics(pairwise: _Pairwise, names: list[str]) -> dict:
    summary = pairwise.pairs
    return {
        "tested": summary.tested,
        "skipped_small_counts": summary.skipped,
        "above_threshold": summary.above_threshold,
        "listed": [
            {
                "a": names[pairwise.partial[pair.a]],
                "b": names[pairwise.partial[pair.b]],
                "phi": pair.phi,
                "n_both": pair.n_both,
                "p_adjusted": pair.p_adjusted,
            }
            for pair in summary.listed
        ],
    }
