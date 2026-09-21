import datetime
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from datadoctor import AnalysisConfig, AnalysisResult, Provenance, __version__, load_dataset
from datadoctor.cli import app
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import run_quality_checks

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
    assert "quality" in result.output


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


def defects_csv(tmp_path):
    """100 rows with planted defects: missing scores, a stray token among numbers, a negative
    age, future dates and a column that never changes."""
    lines = ["age,units,flag,visit,score,city"]
    for k in range(100):
        age = -3 if k in (1, 2) else 20 + k % 40
        units = "unknown" if k in (4, 9) else 100 + k * 7 % 90
        visit = "2999-01-01" if k in (3, 6) else "2024-03-01"
        score = "" if k % 3 == 0 else k * 1.5
        city = ["Oslo", "Paris", "Rome", "Lima"][k % 4]
        lines.append(f"{age},{units},same,{visit},{score},{city}")
    path = tmp_path / "defects.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


def clean_csv(tmp_path):
    lines = ["x,y,city"]
    for k in range(100):
        lines.append(
            f"{k * 37 % 101},{k * 53 % 97 * 1.5},{['Oslo', 'Paris', 'Rome', 'Lima'][k % 4]}"
        )
    path = tmp_path / "clean.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


class TestQualityOutput:
    def test_findings_are_printed_most_severe_first(self, tmp_path):
        result = run("quality", defects_csv(tmp_path), "--target", "score")

        titles = [
            "Target column has missing values",  # high
            "Numbers stored as text",  # medium
            "Constant columns",  # low
        ]
        assert result.exit_code == 0
        positions = [result.output.index(title) for title in titles]
        assert positions == sorted(positions)
        first_heading = next(line for line in result.output.splitlines() if "confidence" in line)
        assert first_heading.strip().startswith("HIGH")

    def test_the_planted_defects_are_named_by_column(self, tmp_path):
        output = run("quality", defects_csv(tmp_path)).output

        for title in [
            "Numbers stored as text",
            "Negative values in age column age",
            "Dates in the future in visit",
            "Constant columns",
        ]:
            assert title in output
        assert "Columns:        units" in output

    def test_each_finding_keeps_evidence_interpretation_and_limitations_apart(self, tmp_path):
        output = run("quality", defects_csv(tmp_path)).output

        for label in ["Evidence:", "Interpretation:", "Limitations:", "Recommendation:"]:
            assert label in output

    def test_fields_a_finding_does_not_have_are_left_out_rather_than_printed_empty(self, tmp_path):
        lines = [line.strip() for line in run("quality", defects_csv(tmp_path)).output.splitlines()]

        assert "Columns:" not in lines
        assert "Recommendation:" not in lines

    def test_the_summary_counts_only_the_severities_that_occur(self, tmp_path):
        output = run("quality", defects_csv(tmp_path), "--target", "score").output

        summary = next(line for line in output.splitlines() if line.startswith("Findings:"))
        assert "1 high" in summary
        assert "critical" not in summary
        assert " 0 " not in summary

    def test_the_settings_and_reference_date_are_shown(self, tmp_path):
        output = run("quality", defects_csv(tmp_path), "--as-of", "2030-06-15").output

        assert "seed 42, row threshold 100000, column threshold 100" in output
        assert "dates after 2030-06-15 count as future" in output

    def test_target_reaches_the_analyzers(self, tmp_path):
        path = defects_csv(tmp_path)

        assert "Target column has missing values" not in run("quality", path).output
        assert (
            "Target column has missing values" in run("quality", path, "--target", "score").output
        )

    def test_no_findings_is_reported_as_the_checks_that_found_nothing_not_as_clean_data(
        self, tmp_path
    ):
        result = run("quality", clean_csv(tmp_path))

        assert result.exit_code == 0
        assert (
            "No findings from: missingness, duplicates, constants, outliers, dtypes, "
            "impossible_values."
        ) in result.output
        assert "It does not show that the data is clean." in result.output
        assert "Findings:" not in result.output

    def test_square_brackets_in_a_column_name_are_printed_as_they_are(self, tmp_path):
        path = tmp_path / "brackets.csv"
        lines = ["[bold]flag[/bold]"] + ["same"] * 30
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

        assert "[bold]flag[/bold]" in run("quality", path).output


class TestQualityFlagsReachTheLoader:
    def test_separator(self, tmp_path):
        path = tmp_path / "eu.csv"
        path.write_text("flag;x\n" + "same;1\n" * 30, encoding="utf-8", newline="")

        assert "Columns:        flag" in run("quality", path, "--separator", ";").output
        assert "Columns:        flag" not in run("quality", path).output

    def test_format(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n", encoding="utf-8", newline="")

        assert run("quality", path).exit_code == 1
        assert run("quality", path, "--format", "csv").exit_code == 0

    def test_encoding(self, tmp_path):
        path = tmp_path / "latin.csv"
        path.write_bytes("city\nZürich\n".encode("latin-1"))

        assert run("quality", path).exit_code == 1
        assert run("quality", path, "--encoding", "latin-1").exit_code == 0

    def test_sheet(self):
        workbook = FIXTURES / "multisheet.xlsx"

        assert run("quality", workbook).exit_code == 1
        assert run("quality", workbook, "--sheet", "second").exit_code == 0


class TestAsOf:
    def test_the_reference_date_decides_which_dates_are_in_the_future(self, tmp_path):
        path = defects_csv(tmp_path)

        assert (
            "Dates in the future in visit" in run("quality", path, "--as-of", "2000-01-01").output
        )
        assert (
            "Dates in the future in visit"
            not in run("quality", path, "--as-of", "3000-01-01").output
        )


class TestQualityJson:
    def test_json_flag_writes_a_file_instead_of_printing_findings(self, tmp_path):
        out = tmp_path / "quality.json"

        result = run("quality", defects_csv(tmp_path), "--json", out)

        assert result.exit_code == 0
        assert "Wrote quality report to" in result.output
        assert "Findings:" not in result.output
        assert out.exists()

    def test_file_holds_dataset_provenance_and_quality_side_by_side(self, tmp_path):
        path = defects_csv(tmp_path)
        out = tmp_path / "quality.json"
        run("quality", path, "--target", "score", "--as-of", "2025-01-01", "--json", out)

        document = json.loads(out.read_text(encoding="utf-8"))

        assert set(document) == {"dataset", "provenance", "quality"}
        assert document["dataset"]["target"] == "score"
        assert Provenance.from_dict(document["provenance"]).shape == (100, 6)
        dataset = load_dataset(path, target="score")
        expected = run_quality_checks(dataset, AnalysisConfig(), as_of=datetime.date(2025, 1, 1))
        assert AnalysisResult.from_dict(document["quality"]) == expected

    def test_file_carries_column_detail_that_the_findings_leave_out(self, tmp_path):
        out = tmp_path / "quality.json"
        run("quality", defects_csv(tmp_path), "--json", out)

        quality = json.loads(out.read_text(encoding="utf-8"))["quality"]

        age = next(
            c for c in quality["metrics"]["impossible_values"]["columns"] if c["name"] == "age"
        )
        assert age["min"] == -3.0
        assert all("-3.0" not in f["evidence"] for f in quality["findings"])

    def test_unwritable_path_is_a_clean_error(self, tmp_path):
        result = run("quality", defects_csv(tmp_path), "--json", tmp_path / "gone" / "out.json")

        assert result.exit_code == 1
        assert "error: cannot write" in result.output
        assert "Traceback" not in result.output


class TestQualityFailures:
    def test_a_load_error_is_one_line_and_exit_one(self, tmp_path):
        result = run("quality", tmp_path / "nope.csv")

        assert result.exit_code == 1
        assert "error: file not found" in result.output
        assert "Traceback" not in result.output
