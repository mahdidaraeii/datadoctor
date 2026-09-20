import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from datadoctor import load_dataset
from datadoctor.core.exceptions import DataLoadError

FIXTURES = Path(__file__).parent.parent / "fixtures"

EXPECTED = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "score": [91.5, None, 78.0, 88.25],
        "city": ["Oslo", "Zürich", "Paris", "Rome"],
        "active": [True, False, True, True],
    }
)


def copy_fixture(name, tmp_path, as_name=None):
    destination = tmp_path / (as_name or name)
    shutil.copy(FIXTURES / name, destination)
    return destination


class TestEveryFormat:
    @pytest.mark.parametrize(
        "name",
        ["sample.csv", "sample.tsv", "sample.json", "sample.parquet", "sample.xlsx"],
    )
    def test_loads_to_the_same_table(self, name):
        dataset = load_dataset(FIXTURES / name)

        pd.testing.assert_frame_equal(dataset.data, EXPECTED)
        assert dataset.name == "sample"


class TestFormatResolution:
    def test_extension_is_matched_case_insensitively(self, tmp_path):
        path = copy_fixture("sample.csv", tmp_path, as_name="SAMPLE.CSV")

        pd.testing.assert_frame_equal(load_dataset(path).data, EXPECTED)

    def test_file_format_overrides_the_extension(self, tmp_path):
        path = copy_fixture("sample.csv", tmp_path, as_name="sample.txt")

        pd.testing.assert_frame_equal(load_dataset(path, file_format="csv").data, EXPECTED)

    def test_unknown_extension_is_not_guessed(self, tmp_path):
        path = copy_fixture("sample.csv", tmp_path, as_name="sample.txt")

        with pytest.raises(DataLoadError, match="file_format"):
            load_dataset(path)

    def test_unknown_file_format_lists_the_valid_ones(self):
        with pytest.raises(DataLoadError, match="expected one of: csv, excel, json, parquet, tsv"):
            load_dataset(FIXTURES / "sample.csv", file_format="xml")

    def test_option_that_does_not_apply_to_the_format_is_rejected(self):
        with pytest.raises(DataLoadError, match="separator does not apply to tsv"):
            load_dataset(FIXTURES / "sample.tsv", separator=";")


class TestExcel:
    def test_several_sheets_require_a_choice_and_are_listed(self):
        with pytest.raises(DataLoadError, match=r"2 sheets \(first, second\).*sheet="):
            load_dataset(FIXTURES / "multisheet.xlsx")

    def test_sheet_can_be_chosen_by_name_or_position(self):
        by_name = load_dataset(FIXTURES / "multisheet.xlsx", sheet="second")
        by_position = load_dataset(FIXTURES / "multisheet.xlsx", sheet=0)

        assert by_name.data["z"].tolist() == [10, 20]
        pd.testing.assert_frame_equal(by_position.data, EXPECTED)

    def test_unknown_sheet_name_lists_the_sheets(self):
        with pytest.raises(DataLoadError, match=r"no sheet named 'third'; sheets: first, second"):
            load_dataset(FIXTURES / "multisheet.xlsx", sheet="third")

    def test_sheet_position_out_of_range(self):
        with pytest.raises(DataLoadError, match="no sheet at position 5"):
            load_dataset(FIXTURES / "multisheet.xlsx", sheet=5)

    def test_duplicate_header_is_reported_not_renamed(self):
        with pytest.raises(DataLoadError, match="duplicate column names: a"):
            load_dataset(FIXTURES / "duplicate_header.xlsx")

    def test_empty_sheet(self):
        with pytest.raises(DataLoadError, match="sheet 'blank' is empty"):
            load_dataset(FIXTURES / "empty_sheet.xlsx")


class TestJson:
    def write(self, tmp_path, text):
        path = tmp_path / "data.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_values_keep_their_json_types(self, tmp_path):
        path = self.write(
            tmp_path,
            '[{"zip": "01234", "created_at": "2024-01-05", "n": "7"},'
            ' {"zip": "99999", "created_at": "2024-02-06", "n": "8"}]',
        )

        data = load_dataset(path).data

        assert data["zip"].tolist() == ["01234", "99999"]
        assert data["created_at"].tolist() == ["2024-01-05", "2024-02-06"]
        assert data["n"].tolist() == ["7", "8"]

    def test_top_level_must_be_an_array(self, tmp_path):
        with pytest.raises(DataLoadError, match="array of objects, got dict"):
            load_dataset(self.write(tmp_path, '{"a": 1}'))

    def test_every_element_must_be_an_object(self, tmp_path):
        with pytest.raises(DataLoadError, match="element 1 of the array is not an object"):
            load_dataset(self.write(tmp_path, '[{"a": 1}, 2]'))

    def test_duplicate_keys_are_reported_not_dropped(self, tmp_path):
        with pytest.raises(DataLoadError, match="duplicate keys in an object: a"):
            load_dataset(self.write(tmp_path, '[{"a": 1, "a": 2}]'))

    def test_empty_array_has_no_data_rows(self, tmp_path):
        with pytest.raises(DataLoadError, match="no data rows"):
            load_dataset(self.write(tmp_path, "[]"))


class TestParquet:
    def test_stored_named_index_becomes_a_column(self, tmp_path):
        path = tmp_path / "indexed.parquet"
        pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="key")).to_parquet(path)

        data = load_dataset(path).data

        assert list(data.columns) == ["key", "a"]
        assert data["key"].tolist() == ["x", "y"]
        assert isinstance(data.index, pd.RangeIndex)

    def test_stored_index_that_collides_with_a_column_is_reported(self, tmp_path):
        path = tmp_path / "clash.parquet"
        pd.DataFrame({"a": [1, 2]}, index=pd.Index([5, 6], name="a")).to_parquet(path)

        with pytest.raises(DataLoadError, match="stored index cannot be turned into a column"):
            load_dataset(path)


class TestUnreadableFiles:
    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("bad.parquet", "could not read the file as parquet"),
            ("bad.xlsx", "could not read the file as an Excel workbook"),
        ],
    )
    def test_corrupt_file_is_reported(self, tmp_path, name, message):
        path = tmp_path / name
        path.write_bytes(b"this is not a real file")

        with pytest.raises(DataLoadError, match=message):
            load_dataset(path)


class TestMissingEngine:
    @pytest.mark.parametrize(
        ("name", "package", "extra"),
        [("sample.parquet", "pyarrow", "parquet"), ("sample.xlsx", "openpyxl", "excel")],
    )
    def test_install_hint_names_the_extra(self, monkeypatch, name, package, extra):
        monkeypatch.setitem(sys.modules, package, None)

        with pytest.raises(DataLoadError, match=rf"datadoctor\[{extra}\]"):
            load_dataset(FIXTURES / name)
