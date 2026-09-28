"""ML readiness assessment: composes prior signals into one blocking/attention summary.

``check_readiness`` computes no new statistics. It composes four kinds of input, each already
computed elsewhere in this project:

- **Target constancy and target missingness** (S14/S15): computed directly here as a single
  column's ``nunique``/``isna``, not by calling the full ``quality`` sweep. The full
  ``run_quality_checks`` scans every column with seven analyzers, several of which (duplicates,
  outliers) ignore the row guardrail entirely; measured at 1,000,000 rows, that costs roughly
  9.75 seconds to learn two facts about one column that cost about 6 milliseconds to compute
  directly. The threshold and severity grading are not re-derived: ``grade`` and the title
  constants below are imported from ``quality.constants`` and ``quality.missingness_findings``,
  so the numbers and wording stay identical to what those modules would themselves report.
- **Target usability**: the same numeric-is-regression, categorical-or-boolean-is-classification
  rule used throughout ``diagnostics`` and ``eda.relationships``, checked directly against the
  schema rather than by calling ``check_relationships``, which also computes a correlation
  matrix and writes plot files as a side effect -- work and output this step has no use for.
- **Split strategy** (S26) and **leakage** (S27): unlike the two above, there is no cheap
  shortcut here. What this step needs from them is their own computation, not a fact derivable
  from a single column scan. They are accepted as optional parameters, ``splits`` and
  ``leakage``: when the caller already has a result (the calling convention
  ``diagnostics.run.run_diagnostics``, S30, uses), it is used exactly as given and never
  recomputed. When absent, ``check_readiness`` computes it itself, so it remains a valid,
  self-sufficient two-argument analyzer, usable on its own.

None of this is re-emitted as a new finding. Doing so would list the same problem twice once
``diagnose`` (S30) or ``report`` (S31) merge readiness's findings together with quality's,
splits' and leakage's own. The blocking and attention lists live in ``metrics`` only, each entry
a plain ``{"source", "title", "columns"}`` mapping; the title is always one of the constants
imported from the analyzer that owns it, never retyped here. ``check_readiness``'s own findings
are exactly two possible ones: a headline summarizing the overall state, and a target-type
finding for the one blocking condition with no other analyzer to source it from.

Temporal and grouped split-strategy findings stay in the blocking bucket (an undetected split
leak invalidates evaluation, not just training), and the headline says so explicitly when either
fires. With no target set, the headline says readiness was not assessed, never that nothing
blocks: those are different claims, and reporting the second when the truth is the first would
overstate what was checked.
"""

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding, Severity, sort_findings
from datadoctor.diagnostics import readiness_findings as report
from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.leakage_findings import (
    DUPLICATE_TITLE,
    NAMING_TITLE,
    NEAR_PERFECT_TITLE,
)
from datadoctor.diagnostics.splits import check_split_strategy
from datadoctor.diagnostics.splits_findings import (
    GROUPED_TITLE,
    SMALL_N_TITLE,
    STRATIFIED_TITLE,
    TEMPORAL_TITLE,
)
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality.constants import CONSTANT_TARGET_TITLE
from datadoctor.quality.missingness_findings import TARGET_MISSING_TITLE, grade

_TASKS = {"numeric": "regression", "categorical": "classification", "boolean": "classification"}
_BLOCKING_SPLITS_TITLES = (TEMPORAL_TITLE, GROUPED_TITLE)
_ATTENTION_SPLITS_TITLES = (SMALL_N_TITLE, STRATIFIED_TITLE)
_ATTENTION_LEAKAGE_TITLES = (NEAR_PERFECT_TITLE, NAMING_TITLE)


def check_readiness(
    dataset: Dataset,
    config: AnalysisConfig,
    *,
    splits: AnalysisResult | None = None,
    leakage: AnalysisResult | None = None,
) -> AnalysisResult:
    """Compose target, split-strategy and leakage signals into a blocking/attention summary.

    See the module docstring for what is computed directly, what is reused from ``splits`` and
    ``leakage``, and why nothing here is re-emitted as a new finding.

    Args:
        dataset: The dataset to assess. It must have at least one row.
        config: The settings to run under. Only used when ``splits`` or ``leakage`` must be
            computed here; recorded in the result regardless.
        splits: The result of ``check_split_strategy(dataset, config)``, if already computed.
            Computed here when ``None``.
        leakage: The result of ``check_leakage(dataset, config)``, if already computed.
            Computed here when ``None``.

    Returns:
        ``metrics`` with ``n_rows``, ``target_assessed`` and the ``blocking``/``attention``
        lists described above. ``findings`` holds the headline and, when applicable, the
        target-type finding, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows = len(frame)
    if n_rows == 0:
        raise DatasetError("cannot assess readiness for a dataset with no rows")

    if dataset.target is None:
        metrics = {"n_rows": n_rows, "target_assessed": False, "blocking": [], "attention": []}
        return AnalysisResult(
            findings=(report.not_assessed_finding(),), metrics=metrics, config=config
        )

    schema = profile_schema(dataset).metrics["columns"]
    target_position = list(frame.columns).index(dataset.target)
    target_name = dataset.target
    target_kind = schema[target_position]["semantic_type"]
    target_series = frame.iloc[:, target_position]

    blocking: list[dict] = []
    attention: list[dict] = []
    findings: list[Finding] = []

    non_null = target_series.dropna()
    if len(non_null) > 0 and non_null.nunique() <= 1:
        blocking.append(
            {"source": "quality", "title": CONSTANT_TARGET_TITLE, "columns": [target_name]}
        )

    missing_count = int(target_series.isna().sum())
    if missing_count > 0:
        entry = {"source": "quality", "title": TARGET_MISSING_TITLE, "columns": [target_name]}
        if grade(missing_count / n_rows) is Severity.CRITICAL:
            blocking.append(entry)
        else:
            attention.append(entry)

    if target_kind not in _TASKS:
        unusable = report.target_unusable_finding(target_name, target_kind)
        findings.append(unusable)
        blocking.append({"source": "readiness", "title": unusable.title, "columns": [target_name]})

    if splits is None:
        splits = check_split_strategy(dataset, config)
    for finding in splits.findings:
        entry = {
            "source": "splits",
            "title": finding.title,
            "columns": list(finding.affected_columns),
        }
        if finding.title in _BLOCKING_SPLITS_TITLES:
            blocking.append(entry)
        elif finding.title in _ATTENTION_SPLITS_TITLES:
            attention.append(entry)

    if leakage is None:
        leakage = check_leakage(dataset, config)
    for finding in leakage.findings:
        entry = {
            "source": "leakage",
            "title": finding.title,
            "columns": list(finding.affected_columns),
        }
        if finding.title == DUPLICATE_TITLE:
            blocking.append(entry)
        elif finding.title in _ATTENTION_LEAKAGE_TITLES:
            attention.append(entry)

    findings.append(report.headline_finding(blocking, attention))

    metrics = {
        "n_rows": n_rows,
        "target_assessed": True,
        "blocking": blocking,
        "attention": attention,
    }
    return AnalysisResult(findings=tuple(sort_findings(findings)), metrics=metrics, config=config)
