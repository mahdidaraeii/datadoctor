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
        for note in (
            "Read as missing",
            "Leading zeros dropped",
            "Kept as text",
            "Unnamed columns",
            "Promoted index",
        ):
            assert table_row(output, note)[1] == "none"

    def test_loader_notes_are_shown(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_text("city,,score\nNA,x,1\nnull,y,2\nNA,z,3\n", encoding="utf-8", newline="")

        output = run("profile", path).output

        assert table_row(output, "Read as missing")[1] == "city: NA x2, null x1"
        assert table_row(output, "Unnamed columns")[1] == "Unnamed: 1"


def zips_csv(tmp_path):
    """40 rows of a zero-padded code, read as numbers, and a plain number that repeats, so that
    it is neither an identifier nor a cause of repeated rows."""
    lines = ["zip,amount"]
    lines += [
        f"{['01234', '56789', '00042', '90210'][k % 4]},{100 + k * 7 % 13}" for k in range(40)
    ]
    path = tmp_path / "zips.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


class TestLeadingZerosLostAtLoad:
    def test_profile_shows_what_the_file_had_and_the_loaded_column_as_numbers(self, tmp_path):
        output = run("profile", zips_csv(tmp_path)).output

        assert (
            table_row(output, "Leading zeros dropped")[1] == "zip: 20 of 40 rows, 5 characters wide"
        )
        assert table_row(output, "zip")[2] == "numeric"

    def test_a_column_whose_values_had_different_widths_is_shown_without_a_width(self, tmp_path):
        path = tmp_path / "mixed.csv"
        path.write_text("code\n007\n12\n0042\n8\n", encoding="utf-8", newline="")

        output = run("profile", path).output

        assert table_row(output, "Leading zeros dropped")[1] == "code: 2 of 4 rows"

    def test_quality_reports_the_finding_for_the_column(self, tmp_path):
        output = run("quality", zips_csv(tmp_path)).output

        assert "Leading zeros were lost when the file was read" in output
        assert "Columns:        zip" in output

    def test_the_record_is_in_the_json_provenance(self, tmp_path):
        out = tmp_path / "quality.json"
        run("quality", zips_csv(tmp_path), "--json", out)

        provenance = Provenance.from_dict(json.loads(out.read_text(encoding="utf-8"))["provenance"])

        assert provenance.leading_zeros == {"zip": {"values": 20, "checked": 40, "width": 5}}


class TestTextColumnFlag:
    def test_profile_reads_the_column_as_text_and_says_so(self, tmp_path):
        output = run("profile", zips_csv(tmp_path), "--text-column", "zip").output

        assert table_row(output, "Kept as text")[1] == "zip"
        assert table_row(output, "Leading zeros dropped")[1] == "none"
        assert table_row(output, "zip")[2] == "categorical"

    def test_the_flag_can_be_repeated(self, tmp_path):
        path = zips_csv(tmp_path)

        output = run("profile", path, "--text-column", "zip", "--text-column", "amount").output

        assert table_row(output, "Kept as text")[1] == "zip, amount"
        assert table_row(output, "amount")[2] == "categorical"

    def test_quality_has_nothing_to_report_once_the_column_is_text(self, tmp_path):
        path = zips_csv(tmp_path)

        with_flag = run("quality", path, "--text-column", "zip")

        assert "No findings from:" in with_flag.output
        assert "Leading zeros were lost" in run("quality", path).output

    def test_the_finding_points_to_the_flag(self, tmp_path):
        output = run("quality", zips_csv(tmp_path)).output

        assert "--text-column zip" in output

    def test_the_request_is_in_the_json_provenance(self, tmp_path):
        out = tmp_path / "quality.json"
        run("quality", zips_csv(tmp_path), "--text-column", "zip", "--json", out)

        provenance = Provenance.from_dict(json.loads(out.read_text(encoding="utf-8"))["provenance"])

        assert provenance.text_columns == ("zip",)

    def test_a_column_the_file_does_not_have_is_a_clean_error(self, tmp_path):
        for command in ("profile", "quality"):
            result = run(command, zips_csv(tmp_path), "--text-column", "nope")

            assert result.exit_code == 1
            assert "error: " in result.output
            assert "no column named 'nope'" in result.output
            assert "Traceback" not in result.output

    def test_a_format_that_already_keeps_text_rejects_it(self):
        result = run("profile", FIXTURES / "sample.json", "--text-column", "id")

        assert result.exit_code == 1
        assert "text_columns does not apply to json files" in result.output


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
            "impossible_values, privacy."
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


class TestPrivacyInQuality:
    def emails_csv(self, tmp_path):
        lines = ["id,contact,score"]
        lines += [f"{k},user{k}@example.com,{k % 7}" for k in range(60)]
        path = tmp_path / "people.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        return path

    def test_a_privacy_finding_is_printed_with_its_evidence_and_never_the_values(self, tmp_path):
        output = run("quality", self.emails_csv(tmp_path)).output

        assert "Email addresses found in columns" in output
        assert "Columns:        contact" in output
        assert "To confirm" in output
        assert "user5@example.com" not in output
        assert "example.com" not in output

    def test_the_json_file_carries_the_privacy_metrics_without_the_values(self, tmp_path):
        out = tmp_path / "quality.json"
        run("quality", self.emails_csv(tmp_path), "--json", out)

        text = out.read_text(encoding="utf-8")
        quality = json.loads(text)["quality"]

        assert quality["metrics"]["privacy"]["columns"][0]["name"] == "contact"
        assert "user5@example.com" not in text


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


def explore_csv(tmp_path):
    """60 rows: ``ident`` is not plotted or compared, ``x`` is strongly associated with the
    numeric target ``y``."""
    lines = ["ident,x,y"]
    for k in range(60):
        x = k % 5
        y = x * 20 + k % 7
        lines.append(f"ID-{k:04d},{x},{y}")
    path = tmp_path / "explore.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


class TestExploreOutput:
    def test_files_are_listed_and_exist_under_the_configured_directory(self, tmp_path):
        result = run("explore", explore_csv(tmp_path), "--target", "y")

        assert result.exit_code == 0
        assert "Files: 2" in result.output
        for key in ("target_distribution", "univariate_numeric_1"):
            match = re.search(rf"{key}: (\S+)", result.output)
            assert match is not None
            assert Path(match.group(1)).exists()
        # sorted alphabetically, not in the order the analyzers happened to produce them: the
        # relationships analyzer runs second but "target_distribution" sorts before "univariate_*".
        assert result.output.index("target_distribution") < result.output.index(
            "univariate_numeric_1"
        )

    def test_findings_are_printed_most_severe_first(self, tmp_path):
        output = run("explore", explore_csv(tmp_path), "--target", "y").output

        titles = [
            "Some columns were not plotted",  # univariate, an INFO finding
            "Columns most associated with the target",  # relationships, also INFO
            "Some columns were not compared",  # relationships, also INFO
        ]
        positions = [output.index(title) for title in titles]
        assert positions == sorted(positions)
        summary = next(line for line in output.splitlines() if line.startswith("Findings:"))
        assert "3 info" in summary

    def test_target_reaches_the_analyzers(self, tmp_path):
        path = explore_csv(tmp_path)

        assert "Columns most associated with the target" not in run("explore", path).output
        assert (
            "Columns most associated with the target"
            in run("explore", path, "--target", "y").output
        )

    def test_no_findings_is_reported_as_the_checks_that_found_nothing_not_as_clean_data(
        self, tmp_path
    ):
        result = run("explore", clean_csv(tmp_path))

        assert result.exit_code == 0
        assert "No findings from: univariate, relationships." in result.output
        assert "Findings:" not in result.output


class TestExploreFlagsReachTheLoader:
    def test_format(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n", encoding="utf-8", newline="")

        assert run("explore", path).exit_code == 1
        assert run("explore", path, "--format", "csv").exit_code == 0


class TestExploreFailures:
    def test_a_load_error_is_one_line_and_exit_one(self, tmp_path):
        result = run("explore", tmp_path / "nope.csv")

        assert result.exit_code == 1
        assert "error: file not found" in result.output
        assert "Traceback" not in result.output

    def test_a_dataset_with_no_rows_is_a_clean_error(self, tmp_path):
        # The loader itself refuses an empty file, so run_eda's own guard is never reached here;
        # that guard is exercised directly in tests/eda/test_run.py.
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n", encoding="utf-8", newline="")

        result = run("explore", path)

        assert result.exit_code == 1
        assert "error: " in result.output
        assert "no data rows" in result.output
        assert "Traceback" not in result.output
