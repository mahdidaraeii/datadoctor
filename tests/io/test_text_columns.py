import json

import openpyxl
import pandas as pd
import pytest

from datadoctor import load_dataset
from datadoctor.core.exceptions import DataLoadError

ZIPS = ["01234", "56789", "00042", "90210"]


def zips_csv(tmp_path, name="zips.csv", separator=","):
    lines = [separator.join(["zip", "amount", "note"])]
    lines += [separator.join([ZIPS[k % 4], str(100 + k), "x"]) for k in range(8)]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


class TestKeepingColumnsAsText:
    def test_a_csv_column_keeps_its_zeros_and_the_others_are_read_as_before(self, tmp_path):
        dataset = load_dataset(zips_csv(tmp_path), text_columns=["zip"])

        assert dataset.data["zip"].tolist() == [ZIPS[k % 4] for k in range(8)]
        assert pd.api.types.is_string_dtype(dataset.data["zip"])
        assert pd.api.types.is_integer_dtype(dataset.data["amount"])

    def test_a_tsv_column_keeps_its_zeros(self, tmp_path):
        path = zips_csv(tmp_path, "zips.tsv", separator="\t")

        dataset = load_dataset(path, text_columns=["zip"])

        assert dataset.data["zip"].tolist()[:2] == ["01234", "56789"]

    def test_an_excel_column_keeps_its_zeros_and_numbers_become_their_text(self, tmp_path):
        book = openpyxl.Workbook()
        book.active.append(["zip", "amount"])
        book.active.append(["01234", 100])
        book.active.append([56789, 101])
        path = tmp_path / "zips.xlsx"
        book.save(path)

        dataset = load_dataset(path, text_columns=["zip"])

        assert dataset.data["zip"].tolist() == ["01234", "56789"]
        assert pd.api.types.is_integer_dtype(dataset.data["amount"])

    def test_the_request_is_recorded_once_per_column_in_the_order_given(self, tmp_path):
        dataset = load_dataset(zips_csv(tmp_path), text_columns=["note", "zip", "note"])

        assert dataset.provenance.text_columns == ("note", "zip")

    def test_no_request_records_nothing(self, tmp_path):
        assert load_dataset(zips_csv(tmp_path)).provenance.text_columns == ()

    def test_a_column_kept_as_text_is_not_reported_as_having_lost_its_zeros(self, tmp_path):
        without = load_dataset(zips_csv(tmp_path))
        kept = load_dataset(zips_csv(tmp_path), text_columns=["zip"])

        assert "zip" in without.provenance.leading_zeros
        assert kept.provenance.leading_zeros == {}

    def test_missing_value_tokens_in_a_text_column_are_still_read_as_missing_and_recorded(
        self, tmp_path
    ):
        path = tmp_path / "codes.csv"
        path.write_text("code,n\n007,1\nNA,2\n012,3\n", encoding="utf-8", newline="")

        dataset = load_dataset(path, text_columns=["code"])

        assert dataset.data["code"].isna().tolist() == [False, True, False]
        assert dataset.provenance.converted_tokens == {"code": {"NA": 1}}

    def test_an_unnamed_column_is_selected_by_the_name_it_is_given(self, tmp_path):
        path = tmp_path / "unnamed.csv"
        path.write_text("a,,c\n1,007,3\n2,008,4\n", encoding="utf-8", newline="")
        book = openpyxl.Workbook()
        book.active.append(["a", None, "c"])
        book.active.append([1, "007", 3])
        book.active.append([2, "008", 4])
        workbook = tmp_path / "unnamed.xlsx"
        book.save(workbook)

        from_csv = load_dataset(path, text_columns=["Unnamed: 1"])
        from_excel = load_dataset(workbook, text_columns=["Unnamed: 1"])

        assert from_csv.data["Unnamed: 1"].tolist() == ["007", "008"]
        assert from_excel.data["Unnamed: 1"].tolist() == ["007", "008"]


class TestWhatIsRejected:
    def test_a_name_the_file_does_not_have_is_an_error_that_lists_the_columns(self, tmp_path):
        with pytest.raises(
            DataLoadError, match=r"no column named 'nope'; columns: zip, amount, note$"
        ):
            load_dataset(zips_csv(tmp_path), text_columns=["zip", "nope"])

    def test_every_missing_name_is_reported(self, tmp_path):
        with pytest.raises(DataLoadError, match="no column named 'x', 'y';"):
            load_dataset(zips_csv(tmp_path), text_columns=["x", "zip", "y"])

    def test_a_long_column_list_is_cut_short(self, tmp_path):
        path = tmp_path / "wide.csv"
        path.write_text(",".join(f"c{i}" for i in range(25)) + "\n" + ",".join("1" * 25) + "\n")

        with pytest.raises(DataLoadError, match=r"c19 and 5 more$"):
            load_dataset(path, text_columns=["nope"])

    def test_exactly_twenty_columns_are_all_listed_without_a_remainder(self, tmp_path):
        path = tmp_path / "twenty.csv"
        path.write_text(",".join(f"c{i}" for i in range(20)) + "\n" + ",".join("1" * 20) + "\n")

        with pytest.raises(DataLoadError, match=r"c18, c19$"):
            load_dataset(path, text_columns=["nope"])

    def test_json_and_parquet_reject_the_option_because_they_already_keep_text(self, tmp_path):
        as_json = tmp_path / "zips.json"
        as_json.write_text(json.dumps([{"zip": "01234"}]), encoding="utf-8")
        as_parquet = tmp_path / "zips.parquet"
        pd.DataFrame({"zip": ["01234"]}).to_parquet(as_parquet)

        with pytest.raises(DataLoadError, match="text_columns does not apply to json files"):
            load_dataset(as_json, text_columns=["zip"])
        with pytest.raises(DataLoadError, match="text_columns does not apply to parquet files"):
            load_dataset(as_parquet, text_columns=["zip"])

    def test_an_empty_request_is_not_an_option_that_was_given(self, tmp_path):
        as_json = tmp_path / "zips.json"
        as_json.write_text(json.dumps([{"zip": "01234"}]), encoding="utf-8")

        assert load_dataset(as_json, text_columns=[]).data["zip"].tolist() == ["01234"]

    @pytest.mark.parametrize("bad", ["zip", ["zip", 3]])
    def test_a_single_string_or_a_non_string_name_is_a_type_error(self, tmp_path, bad):
        with pytest.raises(TypeError, match="sequence of column names"):
            load_dataset(zips_csv(tmp_path), text_columns=bad)
