"""The minimal diagnostic report: profile, quality, eda and diagnose combined into one document.

``build_report`` computes all four in full. ``diagnose`` reuses the ``quality`` and ``eda``
results already computed here rather than a second time -- see ``diagnostics.run``'s own
docstring for the cost that avoids: on the order of 13 and 9 to 12 seconds respectively at
1,000,000 rows, plus writing eda's plot files a second time.

Unlike ``diagnose``, which deliberately scopes itself to split strategy, leakage and readiness
and says so in its own header, ``report`` is the opposite: profile's, quality's and eda's own
findings are included in full, unfiltered, alongside diagnose's own already-filtered
recommendation findings, so the document is everything in one place.

Combining a raw eda finding with diagnose's already-merged recommendation for that same fact
would otherwise show it twice, once incompletely -- see
``diagnostics.recommendations.drop_findings_superseded_by_recommendations`` for how that is
resolved, applied here after the ordinary ``dict.fromkeys`` exact-duplicate collapse and before
the findings are ranked.

The document is written to ``config.output_dir / "report.md"``, always: this is ``report``'s
primary output, not an opt-in export the way ``--json`` is for ``quality`` and ``explore``. Its
path is recorded in ``artifacts["report"]``, alongside eda's own plot files (never re-saved,
since eda only runs once here); its full text is also in ``metrics["markdown"]``.
"""

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Severity, sort_findings
from datadoctor.diagnostics.recommendations import drop_findings_superseded_by_recommendations
from datadoctor.diagnostics.run import run_diagnostics
from datadoctor.eda.run import run_eda
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality.run import run_quality_checks
from datadoctor.render import render_report_markdown

_REPORT_FILENAME = "report.md"


def build_report(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Run profile, quality, eda and diagnose, and combine them into one document.

    See the module docstring for what runs once versus is reused, which findings are kept, and
    where the document is written.

    Args:
        dataset: The dataset to report on. It must have at least one row.
        config: The settings to run under. ``output_dir`` is where the document, and any plot
            files eda writes, are saved. Recorded in the result.

    Returns:
        ``findings`` holds every source's findings, merged and ranked (see the module
        docstring). ``metrics`` holds ``checks_run``, ``findings_by_severity``, each of
        ``profile``, ``quality``, ``eda`` and ``diagnose``'s own metrics, and ``markdown``, the
        document's full text. ``artifacts`` holds the written document under ``"report"`` and
        every plot file eda wrote.
    """
    if len(dataset.data) == 0:
        raise DatasetError("cannot build a report for a dataset with no rows")

    profile = profile_schema(dataset)
    quality = run_quality_checks(dataset, config)
    eda = run_eda(dataset, config)
    diagnose = run_diagnostics(dataset, config, quality=quality, eda=eda)

    ordered_findings = (*profile.findings, *quality.findings, *eda.findings, *diagnose.findings)
    unique_findings = dict.fromkeys(ordered_findings)
    resolved_findings = drop_findings_superseded_by_recommendations(unique_findings)
    findings = tuple(sort_findings(resolved_findings))

    by_severity = sorted(Severity, key=lambda severity: severity.rank, reverse=True)
    findings_by_severity = {
        severity.value: sum(finding.severity is severity for finding in findings)
        for severity in by_severity
    }

    markdown = render_report_markdown(
        dataset, config, findings, findings_by_severity, eda.artifacts
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / _REPORT_FILENAME
    report_path.write_text(markdown, encoding="utf-8")

    metrics = {
        "checks_run": ["profile", "quality", "eda", "diagnose"],
        "findings_by_severity": findings_by_severity,
        "profile": profile.metrics,
        "quality": quality.metrics,
        "eda": eda.metrics,
        "diagnose": diagnose.metrics,
        "markdown": markdown,
    }
    artifacts = {"report": report_path, **eda.artifacts}
    return AnalysisResult(findings=findings, metrics=metrics, artifacts=artifacts, config=config)
