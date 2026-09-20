"""Command line interface. A thin layer over the analysis engine."""

from typing import Annotated

import typer

from datadoctor import __version__

app = typer.Typer(
    help="A diagnostic workbench for tabular datasets.",
    no_args_is_help=True,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"datadoctor {__version__}")
        raise typer.Exit()


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
