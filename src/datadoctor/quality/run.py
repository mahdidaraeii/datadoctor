"""Run every data quality and privacy analyzer on a dataset and merge what they found."""

import datetime

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Severity, sort_findings
from datadoctor.privacy.pii import check_pii
from datadoctor.quality.constants import check_constants
from datadoctor.quality.dtypes import check_dtypes
from datadoctor.quality.duplicates import check_duplicates
from datadoctor.quality.impossible import check_impossible_values
from datadoctor.quality.missingness import check_missingness
from datadoctor.quality.outliers import check_outliers


def run_quality_checks(
    dataset: Dataset, config: AnalysisConfig, *, as_of: datetime.date | None = None
) -> AnalysisResult:
    """Run every data quality and privacy analyzer and merge their results.

    The analyzers are missingness, duplicates, constants, outliers, dtypes, impossible values and
    privacy (personal data), run in that order. Each one is documented in its own module.

    ``findings`` holds every finding, ordered from most to least severe. Findings of equal
    severity keep the order of the analyzers above, so the same input always gives the same list.
    A finding that two analyzers report identically, such as the notice that a row sample was
    used, appears once.
    ``metrics`` holds ``checks_run`` (the analyzer names), ``findings_by_severity`` (a count for
    every severity, most severe first) and one entry per analyzer, under its name, with that
    analyzer's own metrics. The per-analyzer metrics carry column detail that the findings leave
    out on purpose, such as the lowest and highest values of numeric columns.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.
        as_of: The reference date for "the future" in the impossible values check. Defaults to
            the dataset's load date, which is recorded in ``metrics``.

    Returns:
        The merged findings and metrics, with ``config`` recorded.
    """
    if len(dataset.data) == 0:
        raise DatasetError("cannot check the quality of a dataset with no rows")

    results = {
        "missingness": check_missingness(dataset, config),
        "duplicates": check_duplicates(dataset, config),
        "constants": check_constants(dataset, config),
        "outliers": check_outliers(dataset, config),
        "dtypes": check_dtypes(dataset, config),
        "impossible_values": check_impossible_values(dataset, config, as_of=as_of),
        "privacy": check_pii(dataset, config),
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
    return AnalysisResult(findings=findings, metrics=metrics, config=config)
