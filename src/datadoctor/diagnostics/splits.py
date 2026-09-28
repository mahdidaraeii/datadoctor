"""Split-strategy diagnostics: does a plain random train/test split risk leaking information?

``check_split_strategy`` looks for four things a plain random split can get wrong: a datetime
column whose value is associated with the target (a random split leaks the future into
training), a named and repeated entity column whose value is associated with the target (a
random split can put the same entity on both sides), too few rows for a stable split at
conventional ratios, and a classification target imbalanced enough to warrant stratification.
Temporal and group associations reuse the same Pearson/eta-squared/Cramer's V machinery and the
same effect-size and significance conventions as ``eda.relationships``, rather than inventing new
ones. Each check reports at most one ``Finding``, naming every column that qualifies.

The candidate group columns are found from ``profiling.schema``'s ``identifier_named`` flag,
computed for every column regardless of whether it is promoted to the ``identifier`` semantic
type. ``quality.duplicates.check_duplicates`` only acts on promoted identifiers, which requires
near-uniqueness; a genuine entity column such as a foreign key is deliberately repeated, so it is
never promoted and never reaches that check. ``identifier_named`` is the part of that signal
that does generalize here, and reusing it directly (rather than ``quality.duplicates``'s
promotion-gated output) is why a repeated entity column's confidence below is the identifier
heuristic's own confidence for a named, non-unique column, not 1.0: the association strength that
gates whether the finding fires at all is exact, but whether the candidate is really an entity
column, as opposed to a coincidentally named one, is only as good as the name-pattern heuristic.
A datetime column's typing does not carry this uncertainty, so the temporal finding stays at 1.0.
"""

import numpy as np
import pandas as pd
from scipy import stats

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.guardrails import apply_guardrails
from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.diagnostics import splits_findings as report
from datadoctor.eda.relationships import (
    ALPHA,
    DISCRETE_MAX_DISTINCT,
    IMBALANCE_LOW,
    IMBALANCE_MEDIUM,
    MIN_EFFECT,
)
from datadoctor.eda.relationships_stats import MAX_CATEGORIES, MIN_PAIRS, cramers_v, eta_squared
from datadoctor.profiling.schema import profile_schema

_TASKS = {"numeric": "regression", "categorical": "classification", "boolean": "classification"}

# The share of rows a conventional train/test split holds out, and the fewest test rows below
# which reported performance is treated as too noisy to trust. Both are stated conventions.
CONVENTIONAL_TEST_FRACTION = 0.2
MIN_TEST_ROWS = 30

# A named identifier-like column needs at least this many rows per distinct value on average to
# be treated as a repeated entity. Below it, the column is closer to a unique identifier with a
# handful of incidental repeats (two customers who happen to share an id) than to genuine group
# structure, the same spirit as relationships_stats.MAX_CATEGORIES capping a contingency table
# that would otherwise be too sparse to mean anything.
MIN_AVERAGE_GROUP_SIZE = 2.0


def check_split_strategy(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Look for structure that a plain random train/test split would mishandle.

    **Temporal.** Every datetime column is compared against a usable target (Pearson
    correlation for a regression target, the correlation ratio for a classification one, on the
    row guardrail's sample), the same association machinery and effect-size and significance
    conventions ``eda.relationships`` uses. A time-based split is recommended when at least one
    clears them.

    **Group.** Every column named like an identifier (``identifier_named`` from
    ``profiling.schema``, regardless of whether it was promoted to the ``identifier`` semantic
    type) with fewer distinct values than rows is compared the same way (the correlation ratio
    for a regression target, Cramer's V for a classification one). A grouped split is recommended
    when at least one clears the same thresholds; the finding's confidence is the identifier
    heuristic's own confidence for that column, since the association strength is exact but the
    candidate's identity as a real entity column is only as good as the name pattern.

    **Small n.** Independent of the target: below ``MIN_TEST_ROWS`` expected rows in a
    ``CONVENTIONAL_TEST_FRACTION`` test split, computed on every row regardless of the row
    guardrail.

    **Stratification.** For a usable classification target, the minority class's share of every
    row (not the guardrail's sample), graded the same ``IMBALANCE_MEDIUM``/``IMBALANCE_LOW``
    conventions as ``eda.relationships``'s class-imbalance finding.

    A target that is absent, or whose type ``eda.relationships`` also treats as unusable for
    association testing, skips temporal, group and stratification silently; small n still runs.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under, including the row guardrail. Recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows = len(frame)
    if n_rows == 0:
        raise DatasetError("cannot assess split strategy for a dataset with no rows")

    schema = profile_schema(dataset).metrics["columns"]
    names = [column["name"] for column in schema]
    kinds = [column["semantic_type"] for column in schema]
    target_position = None if dataset.target is None else list(frame.columns).index(dataset.target)

    guard = apply_guardrails(dataset, config)
    sample = guard.frame
    findings: list[Finding] = list(guard.findings)

    task = None
    target_values = None
    target_codes = None
    if target_position is not None and kinds[target_position] in _TASKS:
        task = _TASKS[kinds[target_position]]
        target_series = sample.iloc[:, target_position]
        if task == "regression":
            target_values = target_series.to_numpy(dtype="float64", na_value=np.nan)
        else:
            target_codes, _ = pd.factorize(target_series)

    small_n_metrics, small_n_finding = _check_small_n(n_rows)
    if small_n_finding is not None:
        findings.append(small_n_finding)

    temporal_metrics, temporal_finding = _check_temporal(
        sample, names, kinds, target_position, task, target_values, target_codes
    )
    if temporal_finding is not None:
        findings.append(temporal_finding)

    group_metrics, group_finding = _check_groups(
        sample, schema, names, n_rows, task, target_values, target_codes
    )
    if group_finding is not None:
        findings.append(group_finding)

    strat_metrics = None
    if target_position is not None:
        target_kind = kinds[target_position]
        target_full_series = frame.iloc[:, target_position]
        if target_kind in ("categorical", "boolean") or _is_discrete_numeric(target_full_series):
            strat_metrics, strat_finding = _check_stratification(target_full_series)
            if strat_finding is not None:
                findings.append(strat_finding)

    metrics = {
        "n_rows": n_rows,
        "small_n": small_n_metrics,
        "temporal": temporal_metrics,
        "groups": group_metrics,
        "stratification": strat_metrics,
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _check_small_n(n_rows: int) -> tuple[dict, Finding | None]:
    expected_test_rows = n_rows * CONVENTIONAL_TEST_FRACTION
    metrics = {"n_rows": n_rows, "expected_test_rows": expected_test_rows}
    if expected_test_rows >= MIN_TEST_ROWS:
        return {**metrics, "flagged": False}, None
    severity = Severity.HIGH if expected_test_rows < MIN_TEST_ROWS / 2 else Severity.MEDIUM
    finding = report.small_n_finding(n_rows, expected_test_rows, severity)
    return {**metrics, "flagged": True}, finding


def _check_temporal(
    sample: pd.DataFrame,
    names: list[str],
    kinds: list[str],
    target_position: int | None,
    task: str | None,
    target_values: np.ndarray | None,
    target_codes: np.ndarray | None,
) -> tuple[dict, Finding | None]:
    candidates = [i for i, kind in enumerate(kinds) if kind == "datetime" and i != target_position]
    metrics = {"candidates": [names[i] for i in candidates], "effects": {}, "flagged": []}
    if task is None or not candidates:
        return metrics, None

    records = []
    for i in candidates:
        ordinal = _datetime_ordinal(sample.iloc[:, i])
        effect_and_p = _association(ordinal, task, target_values, target_codes)
        if effect_and_p is not None:
            records.append((names[i], *effect_and_p))
            metrics["effects"][names[i]] = effect_and_p[0]

    flagged = _significant(records)
    metrics["flagged"] = [name for name, _ in flagged]
    if not flagged:
        return metrics, None
    return metrics, report.temporal_split_finding([name for name, _ in flagged])


def _check_groups(
    sample: pd.DataFrame,
    schema: list[dict],
    names: list[str],
    n_rows: int,
    task: str | None,
    target_values: np.ndarray | None,
    target_codes: np.ndarray | None,
) -> tuple[dict, Finding | None]:
    candidates = [
        i
        for i, column in enumerate(schema)
        if column["identifier_named"]
        and column["cardinality"] is not None
        and column["cardinality"] < n_rows
        and n_rows / column["cardinality"] >= MIN_AVERAGE_GROUP_SIZE
    ]
    metrics = {"candidates": [names[i] for i in candidates], "effects": {}, "flagged": []}
    if task is None or not candidates:
        return metrics, None

    records = []
    confidences: dict[str, float] = {}
    for i in candidates:
        if schema[i]["cardinality"] > MAX_CATEGORIES:
            continue
        codes, _ = pd.factorize(sample.iloc[:, i])
        effect_and_p = _association(codes, task, target_values, target_codes, group=True)
        if effect_and_p is not None:
            records.append((names[i], *effect_and_p))
            confidences[names[i]] = schema[i]["identifier_confidence"]
            metrics["effects"][names[i]] = effect_and_p[0]

    flagged = _significant(records)
    metrics["flagged"] = [name for name, _ in flagged]
    if not flagged:
        return metrics, None
    confidence = min(confidences[name] for name, _ in flagged)
    return metrics, report.grouped_split_finding([name for name, _ in flagged], confidence)


def _check_stratification(target_series: pd.Series) -> tuple[dict | None, Finding | None]:
    values = target_series.dropna()
    if values.empty:
        return None, None
    minority_share = float(values.value_counts(normalize=True).min())
    severity = _imbalance_severity(minority_share)
    if severity is None:
        return {"minority_share": minority_share, "flagged": False}, None
    return {"minority_share": minority_share, "flagged": True}, report.stratified_split_finding(
        target_series.name, minority_share, severity
    )


def _imbalance_severity(minority_share: float) -> Severity | None:
    if minority_share < IMBALANCE_MEDIUM:
        return Severity.MEDIUM
    if minority_share < IMBALANCE_LOW:
        return Severity.LOW
    return None


def _is_discrete_numeric(series: pd.Series) -> bool:
    """Whether a numeric target has few enough distinct whole numbers to check for imbalance.

    Mirrors ``eda.relationships``'s rule exactly: a 0/1- or 1-to-5-coded classification target is
    typed ``numeric`` by the schema profile (a whole-number column is never reinterpreted), so
    stratification is checked independently of ``task``, the same way that module's class-balance
    view is shown independently of it. ``False`` when there is no finite data to judge.
    """
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    whole = bool(np.all(finite % 1 == 0))
    return whole and len(np.unique(finite)) <= DISCRETE_MAX_DISTINCT


def _datetime_ordinal(series: pd.Series) -> np.ndarray:
    """``series`` as nanoseconds since the epoch, ``NaN`` where it does not parse as a date."""
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.notna().to_numpy()
    ordinal = np.full(len(parsed), np.nan)
    ordinal[valid] = parsed[valid].astype("int64").to_numpy(dtype="float64")
    return ordinal


def _association(
    values: np.ndarray,
    task: str,
    target_values: np.ndarray | None,
    target_codes: np.ndarray | None,
    *,
    group: bool = False,
) -> tuple[float, float] | None:
    """A candidate column's association with the target, and its raw p-value, or ``None``.

    ``values`` is a genuine numeric variable (a datetime ordinal) unless ``group`` is set, in
    which case it is a categorical column's codes from ``pandas.factorize``: arbitrary integer
    labels with no ordinal meaning, so a regression target still needs the correlation ratio
    (group as the grouping factor, target as the continuous variable) rather than a Pearson
    correlation directly against the codes, which would measure an accident of label order
    rather than the real group-to-target association.
    """
    if group:
        if task == "regression":
            mask = np.isfinite(target_values) & (values >= 0)
            result = eta_squared(target_values[mask], values[mask])
            return (float(np.sqrt(result.eta2)), result.p) if result is not None else None
        mask = (values >= 0) & (target_codes >= 0)
        result = cramers_v(values[mask], target_codes[mask])
        return (result.v, result.p) if result is not None else None
    if task == "regression":
        mask = np.isfinite(values) & np.isfinite(target_values)
        if mask.sum() < MIN_PAIRS:
            return None
        x, y = values[mask], target_values[mask]
        if x.std() == 0 or y.std() == 0:
            return None
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    mask = np.isfinite(values) & (target_codes >= 0)
    result = eta_squared(values[mask], target_codes[mask])
    return (float(np.sqrt(result.eta2)), result.p) if result is not None else None


def _significant(records: list[tuple[str, float, float]]) -> list[tuple[str, float]]:
    tests_run = len(records)
    return [
        (name, effect)
        for name, effect, p in records
        if abs(effect) >= MIN_EFFECT and min(1.0, p * tests_run) < ALPHA
    ]
