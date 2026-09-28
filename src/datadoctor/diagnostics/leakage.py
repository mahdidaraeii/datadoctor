"""Leakage detection: features that look like they encode the target itself.

``check_leakage`` looks for three signals. A single feature whose association with the target
exceeds a documented threshold is a candidate for leakage, not a confirmed defect: a strong
legitimate predictor looks identical to leakage above the threshold, and association strength
alone cannot tell them apart. A feature whose association is close enough to a perfect one is a
stronger, more specific claim: this is very likely the target itself, duplicated under another
name or encoding. A column name suggesting a post-outcome event, such as a cancellation or a
settlement, is a weaker, name-based signal, independent of the other two.

The two statistical checks reuse ``eda.relationships_stats.column_association``, the same
Pearson/correlation-ratio/Cramer's V dispatch ``eda.relationships`` uses for its own top-predictors
finding, rather than re-deriving it. ``column_association`` always returns its effect on the same
0 to 1 scale as an absolute correlation (the correlation ratio is already the square root of
eta-squared, not eta-squared itself), so the same threshold is always compared against the same
kind of number.

The three statistics do not, however, share the same *reachable* range, and the thresholds are
not calibrated to mean the same strength of evidence across them. Pearson correlation and
Cramer's V can both reach 1.0 for a real, deterministic dependency. The correlation ratio between
a continuous column and a categorical target cannot, in general: when the continuous side has an
ordinary overlapping distribution (a Gaussian mixture, for instance, rather than one distinct
value per class), discretizing the target into classes throws away information that no strength
of relationship can recover, capping the reachable correlation ratio well below 1.0 regardless of
how deterministic the true relationship is. A continuous column that takes one distinct value per
class (a genuine step function, with no within-class variance) has no such ceiling and reaches
1.0 exactly, the same as the other two statistics. A near-perfect numeric predictor of a
classification target is therefore only caught by this check when the relationship is close to a
step function; a smooth, near-deterministic one is not distinguishable from a real, unusually
strong predictor by this measure. This is stated in the near-perfect finding's limitations,
not worked around by a lower or statistic-specific threshold.

A categorical candidate with fewer than ``diagnostics.splits.MIN_AVERAGE_GROUP_SIZE`` rows per
distinct value is excluded before testing. Sparse categories, such as an almost-unique column
with a handful of incidental repeats, can show a spuriously high correlation ratio or Cramer's V
purely from having only one or two rows each to compare, not from a real relationship: the same
failure mode S26 found and guarded against for group-structure detection, and just as much a risk
here, where "suspiciously high" is exactly what both checks are looking for.

The near-perfect and duplicate tiers are exclusive: a column that clears the duplicate threshold
is reported only as a duplicate, not also as a near-perfect predictor, since it is a stronger and
more specific claim about the same evidence.

All three checks need a usable target, the same numeric-is-regression,
categorical-or-boolean-is-classification rule ``eda.relationships`` uses: with no target, or a
target of a type that rule does not cover, every check, including the name-based one, is skipped.
"""

import re

import numpy as np
import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.guardrails import apply_guardrails
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.diagnostics import leakage_findings as report
from datadoctor.diagnostics.splits import MIN_AVERAGE_GROUP_SIZE
from datadoctor.eda.relationships import ALPHA
from datadoctor.eda.relationships_stats import MAX_CATEGORIES, column_association
from datadoctor.profiling.schema import profile_schema

_TASKS = {"numeric": "regression", "categorical": "classification", "boolean": "classification"}
_ASSOCIATION_TYPES = {"numeric", "categorical", "boolean"}

# A single feature's association with the target, on the 0 to 1 absolute-correlation scale
# column_association returns, above which it is reported as a leakage candidate.
NEAR_PERFECT_THRESHOLD = 0.90
# Above this, the feature is reported as very likely a duplicate of the target rather than
# merely a candidate. Both thresholds are stated conventions, not laws.
DUPLICATE_THRESHOLD = 0.999

# Name tokens suggesting a value recorded at or after the outcome, matched as whole words after
# splitting on non-letters and on camelCase, the same tokenizing approach quality.impossible uses.
# Deliberately few and English-only, to keep false alarms rare; see the finding's limitations.
POST_OUTCOME_TOKENS = frozenset(
    {"outcome", "outcomes", "discharge", "settlement", "settled", "closed", "closing", "followup"}
    | {"cancellation", "cancelled", "canceled", "terminated"}
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", _CAMEL.sub(" ", name).lower()))


def check_leakage(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Look for features that look like they encode the target itself.

    **Near-perfect predictor.** Every numeric, categorical or boolean column other than the
    target (on the row guardrail's sample) is compared against a usable target with
    ``column_association``, excluding a categorical candidate with too many distinct values
    (``eda.relationships_stats.MAX_CATEGORIES``) or too few rows per distinct value
    (``diagnostics.splits.MIN_AVERAGE_GROUP_SIZE``). Bonferroni-adjusted p-values use the same
    ``eda.relationships.ALPHA`` significance convention as S21. A column whose adjusted p-value
    is significant and whose effect reaches ``NEAR_PERFECT_THRESHOLD`` is reported, unless it
    also reaches ``DUPLICATE_THRESHOLD``, in which case only the duplicate finding names it.

    **Duplicated target.** The same comparison, at ``DUPLICATE_THRESHOLD``.

    **Suspicious naming.** Every column other than the target whose name contains a token in
    ``POST_OUTCOME_TOKENS`` is reported, independent of its actual association with the target.

    A target that is absent, or whose type ``eda.relationships`` also treats as unusable for
    association testing, skips all three checks, including the name-based one.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under, including the row guardrail. Recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows = len(frame)
    if n_rows == 0:
        raise DatasetError("cannot look for leakage in a dataset with no rows")

    schema = profile_schema(dataset).metrics["columns"]
    names = [column["name"] for column in schema]
    kinds = [column["semantic_type"] for column in schema]
    target_position = None if dataset.target is None else list(frame.columns).index(dataset.target)

    guard = apply_guardrails(dataset, config)
    sample = guard.frame
    findings: list[Finding] = list(guard.findings)

    task = None
    if target_position is not None and kinds[target_position] in _TASKS:
        task = _TASKS[kinds[target_position]]

    near_perfect_metrics, duplicate_metrics, naming_metrics = None, None, None
    if task is not None:
        target_values, target_codes = _target_arrays(sample.iloc[:, target_position], task)
        near_perfect_metrics, duplicate_metrics, association_findings = _check_associations(
            sample, schema, names, kinds, n_rows, target_position, task, target_values, target_codes
        )
        findings.extend(association_findings)

        naming_metrics, naming_finding = _check_naming(names, target_position)
        if naming_finding is not None:
            findings.append(naming_finding)

    metrics = {
        "n_rows": n_rows,
        "near_perfect": near_perfect_metrics,
        "duplicated": duplicate_metrics,
        "suspicious_names": naming_metrics,
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _target_arrays(
    target_series: pd.Series, task: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if task == "regression":
        return target_series.to_numpy(dtype="float64", na_value=np.nan), None
    codes, _ = pd.factorize(target_series)
    return None, codes


def _check_associations(
    sample: pd.DataFrame,
    schema: list[dict],
    names: list[str],
    kinds: list[str],
    n_rows: int,
    target_position: int,
    task: str,
    target_values: np.ndarray | None,
    target_codes: np.ndarray | None,
) -> tuple[dict, dict, list[Finding]]:
    candidates = [
        i
        for i, kind in enumerate(kinds)
        if i != target_position
        and kind in _ASSOCIATION_TYPES
        and _testable(schema[i], kind, n_rows)
    ]

    records: list[tuple[str, float, float]] = []
    effects: dict[str, float] = {}
    for i in candidates:
        result = column_association(
            sample.iloc[:, i], kinds[i] == "numeric", task, target_values, target_codes
        )
        if result is not None:
            records.append((names[i], result.effect, result.p))
            effects[names[i]] = result.effect

    tests_run = len(records)
    significant = {name: effect for name, effect, p in records if min(1.0, p * tests_run) < ALPHA}

    duplicates = [
        name for name, effect in significant.items() if abs(effect) >= DUPLICATE_THRESHOLD
    ]
    near_perfect = [
        name
        for name, effect in significant.items()
        if abs(effect) >= NEAR_PERFECT_THRESHOLD and name not in duplicates
    ]

    candidate_names = [names[i] for i in candidates]
    near_perfect_metrics = {
        "candidates": candidate_names,
        "effects": effects,
        "flagged": near_perfect,
    }
    duplicate_metrics = {"candidates": candidate_names, "effects": effects, "flagged": duplicates}

    findings = []
    if near_perfect:
        findings.append(report.near_perfect_predictor_finding(near_perfect, NEAR_PERFECT_THRESHOLD))
    if duplicates:
        findings.append(report.duplicated_target_finding(duplicates, DUPLICATE_THRESHOLD))
    return near_perfect_metrics, duplicate_metrics, findings


def _testable(column: dict, kind: str, n_rows: int) -> bool:
    if kind == "numeric":
        return True
    cardinality = column["cardinality"]
    if cardinality is None or cardinality > MAX_CATEGORIES:
        return False
    return n_rows / cardinality >= MIN_AVERAGE_GROUP_SIZE


def _check_naming(names: list[str], target_position: int) -> tuple[dict, Finding | None]:
    flagged = [
        name
        for i, name in enumerate(names)
        if i != target_position and _tokens(name) & POST_OUTCOME_TOKENS
    ]
    metrics = {"flagged": flagged}
    if not flagged:
        return metrics, None
    return metrics, report.suspicious_naming_finding(flagged)
