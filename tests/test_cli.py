from typer.testing import CliRunner

from datadoctor import __version__
from datadoctor.cli import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits_zero():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"datadoctor {__version__}"
