"""Terminal rendering of results, and the markdown and HTML reports.

Formatting only. Every number shown here was produced by the engine; nothing is computed.
"""

import html as html_lib
from pathlib import Path

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.provenance import Provenance
from datadoctor.core.result import AnalysisResult, Finding, Severity

_SHA_SHOWN = 16
_SEVERITY_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "bold yellow",
    Severity.LOW: "bold cyan",
    Severity.INFO: "bold blue",
}
# Same hue family as _SEVERITY_STYLE above, translated to CSS for the HTML report.
_REPORT_CSS = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.5; color: #1a1a1a; }
h1, h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 0.25rem;
         font-weight: bold; font-size: 0.8rem; color: white; }
.badge-critical { background: #7a0010; }
.badge-high { background: #c62828; }
.badge-medium { background: #b8860b; }
.badge-low { background: #00838f; }
.badge-info { background: #1565c0; }
.finding { border-left: 4px solid #ccc; padding: 0.25rem 1rem; margin: 1rem 0; }
.finding.severity-critical { border-left-color: #7a0010; }
.finding.severity-high { border-left-color: #c62828; }
.finding.severity-medium { border-left-color: #b8860b; }
.finding.severity-low { border-left-color: #00838f; }
.finding.severity-info { border-left-color: #1565c0; }
.confidence { color: #666; font-weight: normal; font-size: 0.85rem; }
dl { margin: 0.5rem 0; }
dt { font-weight: bold; margin-top: 0.4rem; }
dd { margin-left: 0; }
figure { margin: 1rem 0; }
figcaption { color: #666; font-size: 0.85rem; }
img { max-width: 100%; }
""".strip()


def render_profile(console: Console, dataset: Dataset, result: AnalysisResult) -> None:
    """Print the provenance and the schema of a dataset as two tables."""
    console.print(_provenance_table(dataset, dataset.provenance))
    console.print()
    console.print(_schema_table(result))


def render_quality(console: Console, dataset: Dataset, result: AnalysisResult) -> None:
    """Print the findings of a quality run, most severe first, with the settings they ran under.

    Text is printed as text, never as markup, because column names and messages can contain
    square brackets.
    """
    rows, columns = dataset.data.shape
    metrics = result.metrics
    counts = [
        f"{count} {severity}"
        for severity, count in metrics["findings_by_severity"].items()
        if count
    ]
    console.print(Text(f"Quality: {dataset.name} ({rows} rows, {columns} columns)", style="bold"))
    console.print(Text(f"Settings: {_settings(result)}"))
    console.print()
    if not result.findings:
        checks = ", ".join(metrics["checks_run"])
        console.print(Text(f"No findings from: {checks}."))
        console.print(Text("This means these checks found nothing at their thresholds."))
        console.print(Text("It does not show that the data is clean."))
        return
    console.print(Text(f"Findings: {len(result.findings)} ({', '.join(counts)})"))
    for finding in result.findings:
        console.print()
        console.print(_finding_block(finding))


def render_explore(console: Console, dataset: Dataset, result: AnalysisResult) -> None:
    """Print the files an EDA run generated, then its findings, most severe first.

    Text is printed as text, never as markup, because column names and messages can contain
    square brackets.
    """
    rows, columns = dataset.data.shape
    metrics = result.metrics
    counts = [
        f"{count} {severity}"
        for severity, count in metrics["findings_by_severity"].items()
        if count
    ]
    console.print(Text(f"Explore: {dataset.name} ({rows} rows, {columns} columns)", style="bold"))
    console.print(Text(f"Settings: {_settings(result)}"))
    console.print()
    console.print(Text(f"Files: {len(result.artifacts)}", style="bold"))
    for key, path in sorted(result.artifacts.items()):
        console.print(Text(f"  {key}: {path}"))
    console.print()
    if not result.findings:
        checks = ", ".join(metrics["checks_run"])
        console.print(Text(f"No findings from: {checks}."))
        return
    console.print(Text(f"Findings: {len(result.findings)} ({', '.join(counts)})"))
    for finding in result.findings:
        console.print()
        console.print(_finding_block(finding))


def render_diagnose(console: Console, dataset: Dataset, result: AnalysisResult) -> None:
    """Print the diagnose findings, most severe first, with the settings they ran under.

    Text is printed as text, never as markup, because column names and messages can contain
    square brackets.
    """
    rows, columns = dataset.data.shape
    metrics = result.metrics
    counts = [
        f"{count} {severity}"
        for severity, count in metrics["findings_by_severity"].items()
        if count
    ]
    console.print(Text(f"Diagnose: {dataset.name} ({rows} rows, {columns} columns)", style="bold"))
    console.print(Text(f"Settings: {_settings(result)}"))
    console.print(
        Text(
            "Covers: split strategy, leakage and readiness only. Quality and eda findings are "
            "not repeated here -- run `quality` and `explore` for those."
        )
    )
    console.print()
    if not result.findings:
        console.print(Text("No findings from: split strategy, leakage, readiness."))
        console.print(Text("This means these checks found nothing at their thresholds."))
        console.print(Text("It does not show that the data is clean or ready to model."))
        return
    console.print(Text(f"Findings: {len(result.findings)} ({', '.join(counts)})"))
    for finding in result.findings:
        console.print()
        console.print(_finding_block(finding))


def render_report_markdown(
    dataset: Dataset,
    config: AnalysisConfig,
    findings: tuple[Finding, ...],
    findings_by_severity: dict[str, int],
    figures: dict[str, Path],
) -> str:
    """Build the combined report as a markdown document: an executive summary, every finding in
    the same evidence/interpretation/limitations-separated block style used everywhere else in
    this project, and a limitations section for the report as a whole.

    Formatting only, like every other function here: ``findings``, ``findings_by_severity`` and
    ``figures`` are already computed. Nothing here depends on wall-clock time, so the same
    findings and config always produce the same text.
    """
    rows, columns = dataset.data.shape
    lines = [
        f"# Report: {dataset.name}",
        "",
        f"{rows} rows, {columns} columns. Settings: seed {config.random_seed}, row threshold "
        f"{config.row_threshold}, column threshold {config.column_threshold}.",
        "",
        "## Executive summary",
        "",
    ]
    counts = [f"{count} {severity}" for severity, count in findings_by_severity.items() if count]
    if not findings:
        lines += [
            "No findings from: profile, quality, eda, diagnose.",
            "This means these checks found nothing at their thresholds.",
            "It does not show that the data is clean or ready to model.",
            "",
        ]
    else:
        lines += [f"{len(findings)} findings ({', '.join(counts)}).", ""]

    if figures:
        lines += ["## Figures", ""]
        lines += [f"- **{key}**: `{path}`" for key, path in sorted(figures.items())]
        lines.append("")

    if findings:
        lines += ["## Findings", ""]
        for finding in findings:
            lines += _markdown_finding_block(finding)
            lines.append("")

    lines += [
        "## Limitations",
        "",
        "This report combines schema profiling, data quality checks, exploratory analysis and "
        "ML-readiness diagnostics. It does not train or evaluate a model, and these checks do "
        "not cover every way a dataset can fail. Each finding's own Limitations field states "
        "what that specific check did not test; this section is about the report as a whole.",
        "",
    ]
    return "\n".join(lines)


def _markdown_finding_block(finding: Finding) -> list[str]:
    lines = [
        f"### {finding.severity.value.upper()} -- {finding.title} "
        f"(confidence {finding.confidence})",
        "",
    ]
    if finding.affected_columns:
        lines.append(f"- **Columns**: {', '.join(finding.affected_columns)}")
    lines.append(f"- **Evidence**: {finding.evidence}")
    lines.append(f"- **Interpretation**: {finding.interpretation}")
    lines.append(f"- **Limitations**: {finding.limitations}")
    if finding.recommendation:
        lines.append(f"- **Recommendation**: {finding.recommendation}")
    return lines


def render_report_html(result: AnalysisResult, dataset_name: str | None = None) -> str:
    """Build the combined report as a self-contained HTML document.

    Same content and section order as ``render_report_markdown`` -- executive summary, figures,
    findings, limitations -- styled inline (a ``<style>`` block in ``<head>``, no separate CSS
    file), with each finding's severity shown as a colored badge and a matching left border.

    Formatting only, like every other function here: takes the already-built ``result`` and
    nothing else is computed. ``dataset_name`` is the one thing ``AnalysisResult`` carries no
    field for; it falls back to a generic title when not given.

    Figures are referenced by their bare filename (``<img src="name.png">``), not embedded and
    not given a full path: this document is always written next to the same PNGs ``eda`` wrote,
    so the bare filename is already the correct relative reference. Every piece of text that
    came from the data -- titles, column names, evidence and the rest, and each figure's alt
    text -- is HTML-escaped before insertion, so a value containing ``<``, ``>`` or ``&`` cannot
    break the markup or be misrendered.
    """
    escape = html_lib.escape
    title = escape(dataset_name) if dataset_name else "Diagnostic Report"
    profile_metrics = result.metrics.get("profile", {})
    rows = profile_metrics.get("n_rows")
    columns = profile_metrics.get("n_columns")
    config = result.config

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]
    if rows is not None and columns is not None:
        parts.append(f"<p>{rows} rows, {columns} columns.</p>")
    if config is not None:
        parts.append(
            f"<p>Settings: seed {config.random_seed}, row threshold {config.row_threshold}, "
            f"column threshold {config.column_threshold}.</p>"
        )

    parts.append("<h2>Executive summary</h2>")
    by_severity = result.metrics.get("findings_by_severity", {})
    counts = [f"{count} {severity}" for severity, count in by_severity.items() if count]
    if not result.findings:
        parts += [
            "<p>No findings from: profile, quality, eda, diagnose.</p>",
            "<p>This means these checks found nothing at their thresholds.</p>",
            "<p>It does not show that the data is clean or ready to model.</p>",
        ]
    else:
        parts.append(f"<p>{len(result.findings)} findings ({escape(', '.join(counts))}).</p>")

    figures = {key: path for key, path in result.artifacts.items() if key != "report"}
    if figures:
        parts.append("<h2>Figures</h2>")
        for key, path in sorted(figures.items()):
            alt = escape(key)
            parts.append(
                f'<figure><img src="{escape(path.name)}" alt="{alt}">'
                f"<figcaption>{alt}</figcaption></figure>"
            )

    if result.findings:
        parts.append("<h2>Findings</h2>")
        parts += [_html_finding_block(finding) for finding in result.findings]

    parts += [
        "<h2>Limitations</h2>",
        "<p>This report combines schema profiling, data quality checks, exploratory analysis "
        "and ML-readiness diagnostics. It does not train or evaluate a model, and these checks "
        "do not cover every way a dataset can fail. Each finding's own Limitations field states "
        "what that specific check did not test; this section is about the report as a whole.</p>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def _html_finding_block(finding: Finding) -> str:
    escape = html_lib.escape
    severity = finding.severity.value
    lines = [
        f'<div class="finding severity-{severity}">',
        f'<h3><span class="badge badge-{severity}">{severity.upper()}</span> '
        f"{escape(finding.title)} "
        f'<span class="confidence">confidence {finding.confidence}</span></h3>',
        "<dl>",
    ]
    if finding.affected_columns:
        lines.append(f"<dt>Columns</dt><dd>{escape(', '.join(finding.affected_columns))}</dd>")
    lines.append(f"<dt>Evidence</dt><dd>{escape(finding.evidence)}</dd>")
    lines.append(f"<dt>Interpretation</dt><dd>{escape(finding.interpretation)}</dd>")
    lines.append(f"<dt>Limitations</dt><dd>{escape(finding.limitations)}</dd>")
    if finding.recommendation:
        lines.append(f"<dt>Recommendation</dt><dd>{escape(finding.recommendation)}</dd>")
    lines += ["</dl>", "</div>"]
    return "\n".join(lines)


def _settings(result: AnalysisResult) -> str:
    config = result.config
    text = (
        f"seed {config.random_seed}, row threshold {config.row_threshold}, "
        f"column threshold {config.column_threshold}"
    )
    if "impossible_values" in result.metrics:
        text += f", dates after {result.metrics['impossible_values']['as_of']} count as future"
    return text


def _finding_block(finding: Finding) -> Group:
    head = Text.assemble(
        (f" {finding.severity.value.upper()} ", _SEVERITY_STYLE[finding.severity]),
        " ",
        (finding.title, "bold"),
        f"  confidence {finding.confidence}",
    )
    # A grid gives every field a hanging indent, so long text wraps under its own label.
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    fields = [
        ("Columns", ", ".join(finding.affected_columns)),
        ("Evidence", finding.evidence),
        ("Interpretation", finding.interpretation),
        ("Limitations", finding.limitations),
        ("Recommendation", finding.recommendation),
    ]
    for label, value in fields:
        if value:
            grid.add_row(Text(f"  {label}:", style="dim"), Text(value))
    return Group(head, grid)


def _provenance_table(dataset: Dataset, provenance: Provenance) -> Table:
    table = Table(title="Provenance", title_justify="left", show_header=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")

    rows_and_columns = f"{provenance.shape[0]} rows, {provenance.shape[1]} columns"
    table.add_row("File", dataset.source or "(not from a file)")
    if provenance.file_sha256 is not None:
        # The full digest is 64 characters and does not fit a terminal table. It is in --json.
        table.add_row(f"SHA-256 (first {_SHA_SHOWN})", provenance.file_sha256[:_SHA_SHOWN])
        table.add_row("Size", f"{provenance.file_size_bytes} bytes")
    table.add_row("Shape", rows_and_columns)
    table.add_row("Loaded at", provenance.loaded_at)
    table.add_row("Versions", _versions(provenance))
    table.add_row("Config", _config(provenance))
    table.add_row("Read as missing", _converted_tokens(provenance))
    table.add_row("Leading zeros dropped", _leading_zeros(provenance))
    table.add_row("Kept as text", ", ".join(provenance.text_columns) or "none")
    table.add_row("Unnamed columns", ", ".join(provenance.unnamed_columns) or "none")
    table.add_row("Promoted index", ", ".join(provenance.promoted_index) or "none")
    return table


def _versions(provenance: Provenance) -> str:
    return (
        f"datadoctor {provenance.package_version}, python {provenance.python_version}, "
        f"pandas {provenance.pandas_version}"
    )


def _config(provenance: Provenance) -> str:
    config = provenance.config
    return (
        f"seed {config.random_seed}, row threshold {config.row_threshold}, "
        f"column threshold {config.column_threshold}"
    )


def _converted_tokens(provenance: Provenance) -> str:
    if not provenance.converted_tokens:
        return "none"
    lines = []
    for column, tokens in provenance.converted_tokens.items():
        listed = ", ".join(f"{token} x{count}" for token, count in tokens.items())
        lines.append(f"{column}: {listed}")
    return "\n".join(lines)


def _leading_zeros(provenance: Provenance) -> str:
    if not provenance.leading_zeros:
        return "none"
    lines = []
    for column, entry in provenance.leading_zeros.items():
        line = f"{column}: {entry['values']} of {entry['checked']} rows"
        if "width" in entry:
            line += f", {entry['width']} characters wide"
        lines.append(line)
    return "\n".join(lines)


def _schema_table(result: AnalysisResult) -> Table:
    metrics = result.metrics
    title = f"Schema ({metrics['n_rows']} rows, {metrics['n_columns']} columns)"
    table = Table(title=title, title_justify="left")
    # Text folds instead of being truncated, so nothing is hidden on a narrow terminal, and the
    # column names always keep some room.
    table.add_column("Column", style="bold", overflow="fold", min_width=8)
    table.add_column("Dtype", overflow="fold")
    table.add_column("Type", overflow="fold")
    table.add_column("Distinct", justify="right", overflow="fold")
    table.add_column("Nulls", justify="right", overflow="fold")
    table.add_column("Identifier", justify="right", overflow="fold")

    for column in metrics["columns"]:
        table.add_row(
            column["name"],
            column["dtype"],
            column["semantic_type"],
            _shown(column["cardinality"]),
            str(column["null_count"]),
            _shown(column["identifier_confidence"]),
        )
    return table


def _shown(value: int | float | None) -> str:
    return "-" if value is None else str(value)
