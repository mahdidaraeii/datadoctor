"""Duplicate diagnostics: repeated rows, records hidden behind a unique id, and repeated ids."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import duplicates_findings as report
from datadoctor.quality.codes import MISSING, column_codes


class RowDuplicates(NamedTuple):
    """Rows that repeat an earlier row. ``extra`` counts the repeats, not the originals."""

    extra: int
    groups: int
    largest_group: int
    rate: float
    first_positions: list[int]


class IdentifierDuplicates(NamedTuple):
    name: str
    confidence: float
    duplicated_values: int
    extra_rows: int
    rate: float
    exact_copy_groups: int
    conflicting_groups: int
    first_positions: list[int]


def check_duplicates(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Find repeated rows, records hidden behind a unique id, and repeated identifiers.

    Every row is used: a sample could split a pair of duplicates, so the size guardrails of
    ``config`` do not apply. ``config`` is recorded in the result.

    Two values are the same when they are equal, and all missing values are equal to each other.
    Columns holding values that cannot be compared, such as lists or dicts, are left out of the
    comparison. They are named in ``metrics`` and in a finding, because rows that differ only in
    those columns then count as duplicates. Values are never printed, only counts and row
    positions (counted from 0), since identifiers can be personal data.

    ``metrics`` holds ``rows`` (the exact repeats), ``rows_ignoring_identifiers`` (``None`` unless
    the dataset has columns typed as identifiers by the schema profile), ``identifiers`` (the
    repeated-value counts of the identifiers that are checked) and ``identifiers_not_checked``.

    Findings:

    - repeated rows, graded by the share of rows that are extra copies;
    - repeated rows once identifier columns are ignored, when that finds more. A unique id makes
      every row differ, so a plain comparison of rows always reports clean. The confidence is the
      lowest identifier confidence among the ignored columns, since the claim is only as good as
      the typing that produced it;
    - per identifier column whose name matches an identifier pattern, values that repeat, split
      into groups whose rows are exact copies and groups whose rows differ (the same id with
      conflicting data). An identifier that is typed so only because it is nearly unique is not
      checked, since repeats in it contradict that typing and do not show a defect. It is named
      in ``identifiers_not_checked``, and is still ignored when looking for hidden repeats;
    - an INFO finding when columns were left out of the comparison.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for duplicates in a dataset with no rows")

    names = [str(label) for label in frame.columns]
    schema = profile_schema(dataset).metrics["columns"]
    all_codes = [column_codes(frame.iloc[:, i]) for i in range(n_columns)]
    comparable = [i for i, codes in enumerate(all_codes) if codes is not None]
    left_out = [names[i] for i, codes in enumerate(all_codes) if codes is None]
    identifiers = [i for i in comparable if schema[i]["semantic_type"] == "identifier"]

    matrix = np.column_stack([all_codes[i] for i in comparable]) if comparable else None
    rows = _row_duplicates(matrix, n_rows) if comparable else None

    hidden = None
    if identifiers and len(identifiers) < len(comparable):
        keep = [k for k, i in enumerate(comparable) if i not in identifiers]
        hidden = _row_duplicates(matrix[:, keep], n_rows)

    # Repeated values only count against a column that is typed as an identifier because of its
    # name. When it is typed so only because it is nearly unique, repeats contradict that typing
    # instead of showing a defect, so such columns are not checked.
    named = [i for i in identifiers if schema[i]["identifier_named"]]
    by_identifier = [
        _identifier_duplicates(
            names[i], schema[i]["identifier_confidence"], all_codes[i], matrix, n_rows
        )
        for i in named
    ]

    findings: list[Finding] = []
    if rows is not None and rows.extra:
        findings.append(report.rows_finding(rows, len(comparable), left_out))
    if hidden is not None and hidden.extra > (rows.extra if rows else 0):
        ignored = [names[i] for i in identifiers]
        confidence = min(schema[i]["identifier_confidence"] for i in identifiers)
        findings.append(report.hidden_finding(hidden, ignored, confidence, left_out))
    for duplicates in by_identifier:
        if duplicates.extra_rows:
            findings.append(report.identifier_finding(duplicates))
    if left_out:
        findings.append(report.left_out_finding(left_out, compared=len(comparable)))

    metrics = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "rows": _rows_metrics(rows, len(comparable), left_out),
        "rows_ignoring_identifiers": None
        if hidden is None
        else {"identifier_columns": [names[i] for i in identifiers], **_rows_metrics(hidden)},
        "identifiers": [duplicates._asdict() for duplicates in by_identifier],
        "identifiers_not_checked": [names[i] for i in identifiers if i not in named],
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _row_duplicates(matrix: np.ndarray, n_rows: int) -> RowDuplicates:
    codes = pd.DataFrame(matrix)
    in_group = codes.duplicated(keep=False).to_numpy()
    if not in_group.any():
        return RowDuplicates(0, 0, 0, 0.0, [])
    subset = codes[in_group]
    sizes = subset.groupby(list(subset.columns), sort=False).size()
    extra = int((sizes - 1).sum())
    positions = np.flatnonzero(in_group)[subset.duplicated().to_numpy()]
    return RowDuplicates(
        extra,
        len(sizes),
        int(sizes.max()),
        extra / n_rows,
        [int(p) for p in positions[: report.LISTED]],
    )


def _rows_metrics(
    duplicates: RowDuplicates | None,
    compared_columns: int | None = None,
    left_out: list[str] | None = None,
) -> dict:
    if duplicates is None:
        duplicates = RowDuplicates(0, 0, 0, 0.0, [])
    metrics = {
        "duplicate_rows": duplicates.extra,
        "duplicate_groups": duplicates.groups,
        "largest_group": duplicates.largest_group,
        "rate": duplicates.rate,
        "first_positions": duplicates.first_positions,
    }
    if compared_columns is not None:
        metrics["compared_columns"] = compared_columns
        metrics["left_out"] = left_out or []
    return metrics


def _identifier_duplicates(
    name: str, confidence: float, codes: np.ndarray, matrix: np.ndarray, n_rows: int
) -> IdentifierDuplicates:
    present = np.flatnonzero(codes != MISSING)
    values = pd.Series(codes[present])
    repeated = values.duplicated(keep=False).to_numpy()
    if not repeated.any():
        return IdentifierDuplicates(name, confidence, 0, 0, 0.0, 0, 0, [])

    extra_mask = values.duplicated().to_numpy()
    extra = int(extra_mask.sum())
    rows = present[repeated]
    content = pd.DataFrame(matrix[rows]).groupby(list(range(matrix.shape[1])), sort=False).ngroup()
    per_id = pd.DataFrame({"id": codes[rows], "content": content.to_numpy()})
    distinct_content = per_id.groupby("id")["content"].nunique()
    conflicting = int((distinct_content > 1).sum())
    return IdentifierDuplicates(
        name,
        confidence,
        len(distinct_content),
        extra,
        extra / n_rows,
        len(distinct_content) - conflicting,
        conflicting,
        [int(p) for p in present[extra_mask][: report.LISTED]],
    )
