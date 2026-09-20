import json
import re
from pathlib import Path

from typer.testing import CliRunner

from datadoctor import AnalysisResult, Provenance, __version__, load_dataset
from datadoctor.cli import app
from datadoctor.profiling.schema import profile_schema

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.csv"

runner = CliRunner()


def run(*args):
    # A wide terminal keeps rich from folding cells, so rows can be read back.
    return runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})


def table_row(output, first_cell):
    """The cells of the table row whose first cell is ``first_cell``, box characters removed."""
    for line in output.splitlines():
        cells = [cell.strip() for cell in re.split(r"[|│┃]", line)][1:-1]
        if cells and cells[0] == first_cell:
            return cells
    raise AssertionError(f"no table row starting with {first_cell!r} in:\n{output}")


def test_version_flag_prints_version_and_exits_zero():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"datadoctor {__version__}"


def test_bare_invocation_shows_help_and_exits_two():
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "profile" in result.output


class TestProfileTables:
    def test_schema_table_has_one_row_per_column(self):
        result = run("profile", SAMPLE)

        assert result.exit_code == 0
        assert table_row(result.output, "id")[2:] == ["numeric", "4", "0", "0.4"]
        assert table_row(result.output, "score")[2:] == ["numeric", "3", "1", "-"]
        assert table_row(result.output, "city")[2:] == ["categorical", "4", "0", "-"]
        assert table_row(result.output, "active")[2:] == ["boolean", "2", "0", "-"]

    def test_provenance_table_shows_the_file_facts(self, tmp_path):
        path = tmp_path / "known.csv"
        path.write_bytes(b"a\n1\n")

        output = run("profile", path).output

        assert table_row(output, "SHA-256 (first 16)")[1] == "309b0e45a73d3fc5"
        assert table_row(output, "Size")[1] == "4 bytes"
        assert table_row(output, "Shape")[1] == "1 rows, 1 columns"
        for note in ("Read as missing", "Unnamed columns", "Promoted index"):
            assert table_row(output, note)[1] == "none"

    def test_loader_notes_are_shown(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_text("city,,score\nNA,x,1\nnull,y,2\nNA,z,3\n", encoding="utf-8", newline="")

        output = run("profile", path).output

        assert table_row(output, "Read as missing")[1] == "city: NA x2, null x1"
        assert table_row(output, "Unnamed columns")[1] == "Unnamed: 1"


class TestJsonFile:
    def test_json_flag_writes_a_file_instead_of_printing_tables(self, tmp_path):
        out = tmp_path / "profile.json"

        result = run("profile", SAMPLE, "--json", out)

        assert result.exit_code == 0
        assert "Wrote profile to" in result.output
        assert "Schema (" not in result.output
        assert out.exists()

    def test_file_holds_dataset_provenance_and_schema_side_by_side(self, tmp_path):
        out = tmp_path / "profile.json"
        run("profile", SAMPLE, "--target", "id", "--json", out)

        document = json.loads(out.read_text(encoding="utf-8"))

        assert set(document) == {"dataset", "provenance", "schema"}
        assert document["dataset"] == {
            "name": "sample",
            "source": SAMPLE.as_posix(),
            "target": "id",
        }
        assert Provenance.from_dict(document["provenance"]).shape == (4, 4)
        expected = profile_schema(load_dataset(SAMPLE, target="id"))
        assert AnalysisResult.from_dict(document["schema"]) == expected

    def test_unwritable_path_is_a_clean_error(self, tmp_path):
        result = run("profile", SAMPLE, "--json", tmp_path / "missing_dir" / "out.json")

        assert result.exit_code == 1
        assert "error: cannot write" in result.output
        assert "Traceback" not in result.output


class TestFlagsReachTheLoader:
    def test_target_removes_the_identifier_flag(self):
        assert table_row(run("profile", SAMPLE).output, "id")[-1] == "0.4"
        assert table_row(run("profile", SAMPLE, "--target", "id").output, "id")[-1] == "-"

    def test_separator(self, tmp_path):
        path = tmp_path / "eu.csv"
        path.write_text("a;b\n1;2\n", encoding="utf-8", newline="")

        assert run("profile", path, "--separator", ";").exit_code == 0

    def test_format(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n", encoding="utf-8", newline="")

        assert run("profile", path).exit_code == 1
        assert run("profile", path, "--format", "csv").exit_code == 0

    def test_encoding(self, tmp_path):
        path = tmp_path / "latin.csv"
        path.write_bytes("city\nZürich\n".encode("latin-1"))

        assert run("profile", path).exit_code == 1
        assert run("profile", path, "--encoding", "latin-1").exit_code == 0

    def test_sheet(self):
        workbook = FIXTURES / "multisheet.xlsx"

        assert run("profile", workbook).exit_code == 1
        assert table_row(run("profile", workbook, "--sheet", "second").output, "z")


class TestFailures:
    def test_a_load_error_is_one_line_and_exit_one(self, tmp_path):
        result = run("profile", tmp_path / "nope.csv")

        assert result.exit_code == 1
        assert "error: file not found" in result.output
        assert "Traceback" not in result.output
