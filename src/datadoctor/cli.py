"""Command line interface. A thin layer over the analysis engine."""

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from datadoctor import __version__, load_dataset
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DataDoctorError
from datadoctor.core.result import AnalysisResult
from datadoctor.profiling.schema import profile_schema
from datadoctor.render import render_profile

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


@app.command()
def profile(
    path: Annotated[Path, typer.Argument(metavar="PATH", help="The data file to profile.")],
    target: Annotated[
        str | None, typer.Option(help="Name of the column to be predicted, if there is one.")
    ] = None,
    file_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help="File format: csv, tsv, parquet, excel or json. Default: from the extension.",
        ),
    ] = None,
    separator: Annotated[
        str | None, typer.Option(help="csv only: the field separator. Default: a comma.")
    ] = None,
    encoding: Annotated[
        str | None, typer.Option(help="csv, tsv and json only: the text encoding.")
    ] = None,
    sheet: Annotated[str | None, typer.Option(help="Excel only: the sheet name.")] = None,
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
        )
        result = profile_schema(dataset)
    except DataDoctorError as exc:
        _fail(str(exc))

    if json_path is None:
        render_profile(Console(), dataset, result)
        return
    _write_json(json_path, dataset, result)
    typer.echo(f"Wrote profile to {json_path}")


def _write_json(path: Path, dataset: Dataset, result: AnalysisResult) -> None:
    """Write the provenance and the result side by side, never one inside the other."""
    document = {
        "dataset": {"name": dataset.name, "source": dataset.source, "target": dataset.target},
        "provenance": dataset.provenance.to_dict(),
        "schema": result.to_dict(),
    }
    text = json.dumps(document, indent=2, allow_nan=False)
    try:
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot write {path}: {exc.strerror}")
