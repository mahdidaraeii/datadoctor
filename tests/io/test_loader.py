import pandas as pd
import pytest

from datadoctor import load_dataset
from datadoctor.core.exceptions import DataLoadError, DatasetError


def write(directory, name, content):
    path = directory / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")
    return path


class TestLoading:
    def test_file_is_loaded_exactly_as_written(self, tmp_path):
        path = write(
            tmp_path, "customers.csv", "age,income,city\n21,30.5,Oslo\n35,,Paris\n58,72.0,Rome\n"
        )

        dataset = load_dataset(path)

        expected = pd.DataFrame(
            {
                "age": [21, 35, 58],
                "income": [30.5, None, 72.0],
                "city": ["Oslo", "Paris", "Rome"],
            }
        )
        pd.testing.assert_frame_equal(dataset.data, expected)

    def test_metadata_comes_from_the_file_and_the_arguments(self, tmp_path):
        path = write(tmp_path, "customers.csv", "age,churn\n21,0\n35,1\n")

        dataset = load_dataset(path, target="churn")

        assert dataset.name == "customers"
        assert dataset.source == path.as_posix()
        assert dataset.target == "churn"

    def test_source_is_the_path_as_given_not_an_absolute_one(self, tmp_path, monkeypatch):
        write(tmp_path, "customers.csv", "a\n1\n")
        monkeypatch.chdir(tmp_path)

        assert load_dataset("customers.csv").source == "customers.csv"

    def test_target_must_be_a_column_of_the_file(self, tmp_path):
        path = write(tmp_path, "customers.csv", "age,churn\n21,0\n")

        with pytest.raises(DatasetError, match="'salary'"):
            load_dataset(path, target="salary")


class TestSeparator:
    def test_explicit_separator_is_used(self, tmp_path):
        path = write(tmp_path, "eu.csv", "a;b\n1;2\n3;4\n")

        assert list(load_dataset(path, separator=";").data.columns) == ["a", "b"]

    def test_wrong_separator_is_reported_with_the_fix(self, tmp_path):
        path = write(tmp_path, "eu.csv", "a;b\n1;2\n3;4\n")

        with pytest.raises(DataLoadError, match=r"pass separator=';'"):
            load_dataset(path)

    def test_separator_must_be_one_character(self, tmp_path):
        path = write(tmp_path, "data.csv", "a,b\n1,2\n")

        with pytest.raises(DataLoadError, match="single character"):
            load_dataset(path, separator=",,")


class TestEncoding:
    def test_byte_order_mark_does_not_hide_a_duplicated_first_column_name(self, tmp_path):
        path = write(tmp_path, "bom.csv", b"\xef\xbb\xbfa,b,a\n1,2,3\n")

        with pytest.raises(DataLoadError, match="duplicate column names: a"):
            load_dataset(path)

    @pytest.mark.parametrize(
        "content",
        [b"st\xe4dt\nOslo\n", b"city\nZ\xfcrich\n"],
        ids=["undecodable-header", "undecodable-body"],
    )
    def test_undecodable_file_names_the_encoding_and_suggests_a_fix(self, tmp_path, content):
        path = write(tmp_path, "latin.csv", content)

        with pytest.raises(DataLoadError, match="encoding='latin-1'"):
            load_dataset(path)

    def test_explicit_encoding_reads_the_file_correctly(self, tmp_path):
        path = write(tmp_path, "latin.csv", b"city\nZ\xfcrich\n")

        assert load_dataset(path, encoding="latin-1").data["city"].tolist() == ["Zürich"]


class TestFailures:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DataLoadError, match="file not found"):
            load_dataset(tmp_path / "nope.csv")

    def test_directory(self, tmp_path):
        with pytest.raises(DataLoadError, match="not a file"):
            load_dataset(tmp_path)

    def test_empty_file(self, tmp_path):
        path = write(tmp_path, "empty.csv", b"")

        with pytest.raises(DataLoadError, match="empty"):
            load_dataset(path)

    def test_header_without_rows(self, tmp_path):
        path = write(tmp_path, "header.csv", "a,b\n")

        with pytest.raises(DataLoadError, match="no data rows"):
            load_dataset(path)

    def test_duplicate_column_names_are_named_and_never_renamed(self, tmp_path):
        path = write(tmp_path, "dup.csv", "a,b,a\n1,2,3\n")

        with pytest.raises(DataLoadError, match="duplicate column names: a"):
            load_dataset(path)

    def test_row_longer_than_the_header_in_the_middle_of_the_file(self, tmp_path):
        path = write(tmp_path, "ragged.csv", "a,b\n1,2\n3,4,5\n")

        with pytest.raises(DataLoadError, match="could not parse"):
            load_dataset(path)

    def test_every_row_longer_than_the_header_is_not_silently_shifted_or_truncated(self, tmp_path):
        path = write(tmp_path, "wide.csv", "a,b\n1,2,3\n4,5,6\n")

        with pytest.raises(DataLoadError, match="more fields than the header"):
            load_dataset(path)
