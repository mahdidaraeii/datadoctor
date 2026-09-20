"""Terminal rendering of results.

Formatting only. Every number shown here was produced by the engine; nothing is computed.
"""

from rich.console import Console
from rich.table import Table

from datadoctor.core.dataset import Dataset
from datadoctor.core.provenance import Provenance
from datadoctor.core.result import AnalysisResult

_SHA_SHOWN = 16


def render_profile(console: Console, dataset: Dataset, result: AnalysisResult) -> None:
    """Print the provenance and the schema of a dataset as two tables."""
    console.print(_provenance_table(dataset, dataset.provenance))
    console.print()
    console.print(_schema_table(result))


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
