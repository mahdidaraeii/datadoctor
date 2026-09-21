"""Command line interface. A thin layer over the analysis engine."""

import datetime
import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from datadoctor import AnalysisConfig, __version__, load_dataset
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DataDoctorError
from datadoctor.core.result import AnalysisResult
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import run_quality_checks
from datadoctor.render import render_profile, render_quality

app = typer.Typer(
    help="A diagnostic workbench for tabular datasets.",
    no_args_is_help=True,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"datadoctor {__version__}")
        raise typer.Exit()


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """A diagnostic workbench for tabular datasets."""


Target = Annotated[
    str | None, typer.Option(help="Name of the column to be predicted, if there is one.")
]
FileFormat = Annotated[
    str | None,
    typer.Option(
        "--format",
        help="File format: csv, tsv, parquet, excel or json. Default: from the extension.",
    ),
]
Separator = Annotated[
    str | None, typer.Option(help="csv only: the field separator. Default: a comma.")
]
Encoding = Annotated[str | None, typer.Option(help="csv, tsv and json only: the text encoding.")]
Sheet = Annotated[str | None, typer.Option(help="Excel only: the sheet name.")]
TextColumn = Annotated[
    list[str] | None,
    typer.Option(
        "--text-column",
        help=(
            "csv, tsv and Excel only: keep this column as text, so codes such as 01234 keep "
            "their leading zeros. Repeat the option for more columns."
        ),
    ),
]


@app.command()
def profile(
    path: Annotated[Path, typer.Argument(metavar="PATH", help="The data file to profile.")],
    target: Target = None,
    file_format: FileFormat = None,
    separator: Separator = None,
    encoding: Encoding = None,
    sheet: Sheet = None,
    text_column: TextColumn = None,
    json_path: Annotated[
        Path | None,
        typer.Option(
            "--json", help="Write the result to this file as JSON instead of printing tables."
        ),
    ] = None,
) -> None:
    """Show the schema and provenance of a data file."""
    try:
        dataset = load_dataset(
            path,
            file_format=file_format,
            target=target,
            separator=separator,
            encoding=encoding,
            sheet=sheet,
            text_columns=text_column,
        )
        result = profile_schema(dataset)
    except DataDoctorError as exc:
        _fail(str(exc))

    if json_path is None:
        render_profile(Console(), dataset, result)
        return
    _write_json(json_path, dataset, "schema", result)
    typer.echo(f"Wrote profile to {json_path}")


@app.command()
def quality(
    path: Annotated[Path, typer.Argument(metavar="PATH", help="The data file to check.")],
    target: Target = None,
    file_format: FileFormat = None,
    separator: Separator = None,
    encoding: Encoding = None,
    sheet: Sheet = None,
    text_column: TextColumn = None,
    as_of: Annotated[
        datetime.datetime | None,
        typer.Option(
            "--as-of",
            formats=["%Y-%m-%d"],
            metavar="YYYY-MM-DD",
            help="Dates after this day count as the future. Default: the day the file is loaded.",
        ),
    ] = None,
    json_path: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help=(
                "Write the result to this file as JSON instead of printing it. The file holds "
                "per-column detail that the findings leave out, such as minimum and maximum "
                "values, so review it before sharing."
            ),
        ),
    ] = None,
) -> None:
    """List the data quality problems in a data file, most severe first."""
    config = AnalysisConfig()
    try:
        dataset = load_dataset(
            path,
            file_format=file_format,
            target=target,
            separator=separator,
            encoding=encoding,
            sheet=sheet,
            text_columns=text_column,
            config=config,
        )
        result = run_quality_checks(dataset, config, as_of=None if as_of is None else as_of.date())
    except DataDoctorError as exc:
        _fail(str(exc))

    if json_path is None:
        render_quality(Console(), dataset, result)
        return
    _write_json(json_path, dataset, "quality", result)
    typer.echo(f"Wrote quality report to {json_path}")


def _write_json(path: Path, dataset: Dataset, key: str, result: AnalysisResult) -> None:
    """Write the provenance and the result side by side, never one inside the other."""
    document = {
        "dataset": {"name": dataset.name, "source": dataset.source, "target": dataset.target},
        "provenance": dataset.provenance.to_dict(),
        key: result.to_dict(),
    }
    text = json.dumps(document, indent=2, allow_nan=False)
    try:
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot write {path}: {exc.strerror}")
