"""Size guardrails: keep heavy analyses affordable on large datasets, and say so.

Two limits come from ``AnalysisConfig``. Above the row threshold, analyses work on a seeded
sample of the rows. Above the column threshold, pairwise computations are skipped. Each limit
that applies produces an INFO finding, so an analysis that used a sample never looks as if it
used every row.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.result import Finding, Severity


class Guardrails(NamedTuple):
    """What an analysis should use, and what was applied to get there.

    Attributes:
        frame: The rows to analyze. The dataset's own frame when no limit applies, otherwise a
            new frame holding the sample.
        findings: One INFO finding per limit that applied, rows first.
        pairwise: Whether pairwise computations, such as correlations between all pairs of
            columns, are allowed.
    """

    frame: pd.DataFrame
    findings: tuple[Finding, ...]
    pairwise: bool


def apply_guardrails(dataset: Dataset, config: AnalysisConfig) -> Guardrails:
    """Apply the size limits of ``config`` to a dataset.

    The limits come from ``config`` and never from the dataset: the same dataset can be analyzed
    under different settings, and whatever ``config`` says is what is applied.

    Rows: when the dataset has more rows than ``config.row_threshold``, exactly that many rows
    are drawn uniformly at random with ``config.random_seed``. The draw uses numpy's frozen
    ``RandomState`` stream, so the same data and seed select the same rows in every environment.
    Rows keep their original order and index labels, and the dataset's frame is not changed.

    Columns: when the dataset has more columns than ``config.column_threshold``, ``pairwise`` is
    false.

    Args:
        dataset: The dataset about to be analyzed.
        config: The settings to apply.

    Returns:
        The frame to analyze, the findings for the limits that applied, and whether pairwise
        work is allowed.
    """
    if not isinstance(config, AnalysisConfig):
        raise TypeError(f"config must be an AnalysisConfig, got {type(config).__name__}")

    frame = dataset.data
    findings = []

    n_rows = len(frame)
    if n_rows > config.row_threshold:
        positions = np.random.RandomState(config.random_seed).choice(
            n_rows, size=config.row_threshold, replace=False
        )
        frame = frame.iloc[np.sort(positions)]
        findings.append(_sampled_finding(n_rows, config))

    n_columns = dataset.data.shape[1]
    pairwise = n_columns <= config.column_threshold
    if not pairwise:
        findings.append(_pairwise_skipped_finding(n_columns, config))

    return Guardrails(frame, tuple(findings), pairwise)


def _sampled_finding(n_rows: int, config: AnalysisConfig) -> Finding:
    return Finding(
        category="guardrail",
        severity=Severity.INFO,
        confidence=1.0,
        title="Analysis ran on a sample of the rows",
        evidence=(
            f"The dataset has {n_rows:,} rows, above the row threshold of "
            f"{config.row_threshold:,}. Analyses that apply this guardrail used a uniform random "
            f"sample of {config.row_threshold:,} rows drawn with seed {config.random_seed}."
        ),
        interpretation=(
            "Numbers computed on the sample are estimates of the full-data values and can "
            "differ from them, most for rare categories and rare events."
        ),
        limitations=(
            "The sample is uniform over rows and ignores structure such as groups, time order "
            "and class balance, so it can misrepresent rare classes. Counts from the sample "
            "are not counts of the full data."
        ),
        recommendation=(
            "Raise the row threshold to analyze every row, or repeat the analysis with another "
            "seed to see whether conclusions hold."
        ),
    )


def _pairwise_skipped_finding(n_columns: int, config: AnalysisConfig) -> Finding:
    return Finding(
        category="guardrail",
        severity=Severity.INFO,
        confidence=1.0,
        title="Pairwise analyses were skipped",
        evidence=(
            f"The dataset has {n_columns:,} columns, above the column threshold of "
            f"{config.column_threshold:,}. Pairwise computations, such as correlations between "
            "every pair of columns, were skipped because their cost grows with the square of "
            "the column count."
        ),
        interpretation=(
            "No relationships between columns were computed, so their absence from the results "
            "says nothing about whether such relationships exist."
        ),
        limitations=(
            "Results for single columns are not affected. Skipping is decided by the column "
            "count alone, not by measuring the cost on this data."
        ),
        recommendation=(
            "Raise the column threshold, or select the columns of interest and analyze those."
        ),
    )
