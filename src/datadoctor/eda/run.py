"""Run every exploratory analyzer on a dataset and merge what they found."""

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.result import AnalysisResult, Severity, sort_findings
from datadoctor.eda.relationships import check_relationships
from datadoctor.eda.univariate import check_univariate


def run_eda(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Run every exploratory analyzer and merge their results.

    The analyzers are univariate (every column's own distribution) and relationships (the
    correlation matrix and, with a target set, feature-to-target associations and class balance),
    run in that order. Each one is documented in its own module. Univariate runs first and raises
    if the dataset has no rows, so that check is not repeated here.

    ``findings`` holds every finding, ordered from most to least severe. Findings of equal
    severity keep the order of the analyzers above, so the same input always gives the same list.
    A finding that both analyzers report identically, such as the notice that a row sample was
    used, appears once.
    ``metrics`` holds ``checks_run`` (the analyzer names), ``findings_by_severity`` (a count for
    every severity, most severe first) and one entry per analyzer, under its name, with that
    analyzer's own metrics.
    ``artifacts`` holds every saved figure from both analyzers, keyed by the names each one uses;
    the two never collide.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded, and ``output_dir`` says where the
            figures go.

    Returns:
        The merged findings, metrics and artifacts, with ``config`` recorded.
    """
    results = {
        "univariate": check_univariate(dataset, config),
        "relationships": check_relationships(dataset, config),
    }
    # dict.fromkeys drops repeats and keeps the first of each, so the order stays deterministic.
    every = dict.fromkeys(f for result in results.values() for f in result.findings)
    findings = sort_findings(every)
    by_severity = sorted(Severity, key=lambda severity: severity.rank, reverse=True)
    metrics = {
        "checks_run": list(results),
        "findings_by_severity": {
            severity.value: sum(finding.severity is severity for finding in findings)
            for severity in by_severity
        },
        **{name: result.metrics for name, result in results.items()},
    }
    artifacts = {key: path for result in results.values() for key, path in result.artifacts.items()}
    return AnalysisResult(findings=findings, metrics=metrics, artifacts=artifacts, config=config)
