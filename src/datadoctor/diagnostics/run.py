"""Run every analyzer diagnose needs, and rank the result into recommendations.

``run_diagnostics`` computes quality, eda, split strategy, leakage and readiness in full.
Readiness reuses the already-computed ``splits`` and ``leakage`` results rather than
recomputing them -- the calling convention ``readiness.py``'s own docstring documents as what
this module uses. Quality is run in full even though, under the current merge registry (see
``recommendations.py``), no quality finding can ever survive the filter below on its own:
``build_recommendations`` needs every analyzer's complete finding set to detect a shared root
cause, and skipping quality on the assumption that will stay true would turn into a silent
correctness bug the day the registry grows to include a quality category -- nothing in the test
suite would catch it, since the whole premise for skipping would be that quality's output is
always discarded.

``findings`` and ``metrics["recommendations"]`` are scoped to this package's own domain: split
strategy, leakage and readiness. A ``Recommendation`` is kept only when at least one of its
source findings belongs to one of those three categories. A merged entry (the two groups
``recommendations.py`` registers) is kept whole, both source findings included, when either side
qualifies -- the eda side of a merge is never dropped just because eda itself is out of scope. A
singleton quality-only or eda-only recommendation is dropped. The full quality and eda finding
sets are not repeated here: they have their own commands, ``quality`` and ``explore``.

Findings are the flattened form (``Recommendation.to_finding()``), sorted most severe first;
``metrics["recommendations"]`` keeps the unflattened ``Recommendation`` objects, with both source
findings intact, for anyone who wants that detail rather than the joined summary.
"""

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Severity, sort_findings
from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.recommendations import Recommendation, build_recommendations
from datadoctor.diagnostics.splits import check_split_strategy
from datadoctor.eda.run import run_eda
from datadoctor.quality.run import run_quality_checks

_KEPT_CATEGORIES = {"split_strategy", "leakage", "readiness"}


def run_diagnostics(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Run quality, eda, split strategy, leakage and readiness, and rank the result.

    See the module docstring for what is run, what is reused, and which recommendations make it
    into ``findings``.

    Args:
        dataset: The dataset to diagnose. It must have at least one row.
        config: The settings to run under. Recorded in the result.

    Returns:
        ``findings`` holds the kept recommendations, flattened to one ``Finding`` each, most
        severe first. ``metrics`` holds ``checks_run`` (every analyzer actually run, not only
        the ones whose findings are kept), ``findings_by_severity``, each of ``split_strategy``,
        ``leakage`` and ``readiness``'s own metrics, and ``recommendations``, the kept
        ``Recommendation`` objects in full.
    """
    if len(dataset.data) == 0:
        raise DatasetError("cannot run diagnostics on a dataset with no rows")

    quality = run_quality_checks(dataset, config)
    eda = run_eda(dataset, config)
    splits = check_split_strategy(dataset, config)
    leakage = check_leakage(dataset, config)
    readiness = check_readiness(dataset, config, splits=splits, leakage=leakage)

    recommendations = build_recommendations(
        quality=quality, eda=eda, splits=splits, leakage=leakage, readiness=readiness
    )
    kept = _kept(recommendations)

    findings = tuple(sort_findings(rec.to_finding() for rec in kept))
    by_severity = sorted(Severity, key=lambda severity: severity.rank, reverse=True)
    metrics = {
        "checks_run": ["quality", "eda", "split_strategy", "leakage", "readiness"],
        "findings_by_severity": {
            severity.value: sum(finding.severity is severity for finding in findings)
            for severity in by_severity
        },
        "split_strategy": splits.metrics,
        "leakage": leakage.metrics,
        "readiness": readiness.metrics,
        "recommendations": [rec.to_dict() for rec in kept],
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _kept(recommendations: tuple[Recommendation, ...]) -> tuple[Recommendation, ...]:
    return tuple(
        rec
        for rec in recommendations
        if any(finding.category in _KEPT_CATEGORIES for finding in rec.findings)
    )
