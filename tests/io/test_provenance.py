import hashlib
import shutil
from pathlib import Path

import pandas as pd

from datadoctor import AnalysisConfig, load_dataset

FIXTURES = Path(__file__).parent.parent / "fixtures"

# sha256 of the four bytes b"a\n1\n", computed independently of the code under test.
KNOWN_BYTES = b"a\n1\n"
KNOWN_SHA256 = "309b0e45a73d3fc5325e2b6ed0a01ef8b9cde6b05a5633c1f893f970d52bfddc"


class TestFileHash:
    def test_hash_and_size_are_those_of_the_file_bytes(self, tmp_path):
        path = tmp_path / "known.csv"
        path.write_bytes(KNOWN_BYTES)

        provenance = load_dataset(path).provenance

        assert provenance.file_sha256 == KNOWN_SHA256
        assert provenance.file_size_bytes == 4

    def test_hash_does_not_depend_on_the_path(self, tmp_path):
        first = tmp_path / "first.csv"
        second = tmp_path / "elsewhere.csv"
        first.write_bytes(KNOWN_BYTES)
        second.write_bytes(KNOWN_BYTES)

        assert (
            load_dataset(first).provenance.file_sha256
            == load_dataset(second).provenance.file_sha256
        )

    def test_a_file_larger_than_one_chunk_hashes_like_a_single_pass(self, tmp_path):
        content = b"a\n" + b"1\n" * 700_000
        path = tmp_path / "big.csv"
        path.write_bytes(content)

        provenance = load_dataset(path).provenance

        assert len(content) > 1 << 20
        assert provenance.file_sha256 == hashlib.sha256(content).hexdigest()
        assert provenance.file_size_bytes == len(content)


class TestRecordedContext:
    def test_shape_is_that_of_the_loaded_frame(self):
        dataset = load_dataset(FIXTURES / "sample.csv")

        assert dataset.provenance.shape == dataset.data.shape == (4, 4)

    def test_config_and_seed_are_recorded(self):
        config = AnalysisConfig(random_seed=7, row_threshold=500, column_threshold=12)

        assert load_dataset(FIXTURES / "sample.csv", config=config).provenance.config == config

    def test_config_defaults_when_omitted(self):
        assert load_dataset(FIXTURES / "sample.csv").provenance.config == AnalysisConfig()


class TestConvertedTokens:
    def test_literal_tokens_read_as_missing_are_counted_per_column(self, tmp_path):
        path = tmp_path / "tokens.csv"
        path.write_text(
            'city,score\nOslo,1\nNA,2\n"NA",3\nnull,\nRome,None\n', encoding="utf-8", newline=""
        )

        provenance = load_dataset(path).provenance

        # The empty score cell is missing but is not a literal token, so it is not counted.
        assert provenance.converted_tokens == {"city": {"NA": 2, "null": 1}, "score": {"None": 1}}

    def test_empty_cells_alone_are_not_tokens(self):
        provenance = load_dataset(FIXTURES / "sample.csv").provenance

        assert provenance.converted_tokens == {}

    def test_excel_tokens_are_found_when_column_labels_are_numbers(self, tmp_path):
        path = tmp_path / "years.xlsx"
        frame = pd.DataFrame({2020: ["NA", "b"], 2021: [1, None], 2022: ["null", "q"]})
        frame.to_excel(path, index=False)

        provenance = load_dataset(path).provenance

        assert provenance.converted_tokens == {"2020": {"NA": 1}, "2022": {"null": 1}}


class TestUnnamedColumns:
    def test_empty_csv_header_cell_is_recorded(self, tmp_path):
        path = tmp_path / "gap.csv"
        path.write_text("a,,c\n1,2,3\n", encoding="utf-8", newline="")

        assert load_dataset(path).provenance.unnamed_columns == ("Unnamed: 1",)

    def test_empty_excel_header_cell_is_recorded(self, tmp_path):
        path = tmp_path / "gap.xlsx"
        pd.DataFrame({"a": [1], "": [2], "c": [3]}).to_excel(path, index=False)

        assert load_dataset(path).provenance.unnamed_columns == ("Unnamed: 1",)

    def test_a_file_with_every_header_named_records_none(self):
        assert load_dataset(FIXTURES / "sample.csv").provenance.unnamed_columns == ()


class TestPromotedIndex:
    def test_stored_index_columns_are_recorded_by_name(self, tmp_path):
        named = tmp_path / "named.parquet"
        unnamed = tmp_path / "unnamed.parquet"
        pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="key")).to_parquet(named)
        pd.DataFrame({"a": [1, 2]}, index=[10, 20]).to_parquet(unnamed)

        assert load_dataset(named).provenance.promoted_index == ("key",)
        assert load_dataset(unnamed).provenance.promoted_index == ("index",)

    def test_a_default_index_promotes_nothing(self):
        assert load_dataset(FIXTURES / "sample.parquet").provenance.promoted_index == ()


def test_provenance_of_the_same_file_differs_only_in_the_timestamp(tmp_path):
    path = tmp_path / "copy.csv"
    shutil.copy(FIXTURES / "sample.csv", path)

    first = load_dataset(path).provenance.to_dict()
    second = load_dataset(path).provenance.to_dict()
    first.pop("loaded_at")
    second.pop("loaded_at")

    assert first == second
