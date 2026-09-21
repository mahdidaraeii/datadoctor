import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Provenance, Severity, load_dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality.dtypes import check_dtypes

CONFIG = AnalysisConfig()


def run(frame):
    return check_dtypes(Dataset(data=frame, name="t"), CONFIG)


def column(result, name):
    return next(c for c in result.metrics["columns"] if c["name"] == name)


def kinds(result):
    return {c["name"]: c["kind"] for c in result.metrics["columns"]}


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


def numbers(n):
    return [str(k * 3) for k in range(n)]


class TestNumbersStoredAsText:
    def test_the_column_is_reported_with_the_tokens_that_keep_it_from_being_numeric(self):
        # 96 numbers (plain, with a thousands comma, negative, exponent) and four placeholders.
        values = ["12.5", "1,234.5", "-3", "7e2"] * 24 + ["?", "?", "-", "unknown"]

        result = run(pd.DataFrame({"price": values}))

        c = column(result, "price")
        assert kinds(result) == {"price": "numbers_as_text"}
        assert (c["numeric_share"], c["non_numeric"]) == (0.96, 4)
        assert c["tokens"] == [
            {"token": "?", "count": 2},
            {"token": "-", "count": 1},
            {"token": "unknown", "count": 1},
        ]
        found = finding(result, "Numbers stored as text")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 0.8)
        assert found.affected_columns == ("price",)
        assert "96.0% parse as numbers" in found.evidence
        assert "'?' x2, '-' x1, 'unknown' x1" in found.evidence

    def test_confidence_is_higher_when_every_value_parses(self):
        result = run(pd.DataFrame({"clean": numbers(100)}))

        found = finding(result, "Numbers stored as text")

        assert found.confidence == 0.9
        assert "every value parses as a number" in found.evidence

    def test_the_cutoff_is_95_percent_and_inclusive(self):
        def share(numeric):
            return run(pd.DataFrame({"c": numbers(numeric) + ["x"] * (100 - numeric)}))

        assert kinds(share(95)) == {"c": "numbers_as_text"}
        assert kinds(share(94)) == {}

    def test_a_short_column_is_not_judged(self):
        assert kinds(run(pd.DataFrame({"c": numbers(15)}))) == {}

    def test_long_tokens_are_not_listed_and_at_most_five_are(self):
        # The long value is the most frequent non-numeric one, so it would rank first if it were
        # listed. It is 41 characters, over the limit of 20.
        long_value = "this is a real value that failed to parse"
        values = numbers(191) + [long_value] * 3 + list("abcdefg")
        result = run(pd.DataFrame({"c": values}))

        c = column(result, "c")

        assert [t["token"] for t in c["tokens"]] == ["a", "b", "c", "d", "e"]
        assert c["tokens_not_listed"] == 5  # the three long values, f and g
        assert long_value not in finding(result, "Numbers stored as text").evidence

    def test_empty_strings_are_shown_as_empty(self):
        result = run(pd.DataFrame({"c": numbers(96) + [""] * 4}))

        assert "(empty) x4" in finding(result, "Numbers stored as text").evidence

    def test_formats_that_are_ambiguous_or_carry_symbols_are_not_guessed(self):
        frame = pd.DataFrame(
            {"european": ["1.234,5"] * 100, "money": ["$12"] * 100, "percent": ["45%"] * 100}
        )

        assert kinds(run(frame)) == {}

    def test_many_columns_make_one_finding(self):
        frame = pd.DataFrame({f"c{i}": numbers(100) for i in range(12)})

        result = run(frame)

        found = finding(result, "Numbers stored as text")
        assert len(result.findings) == 1
        assert "12 text columns hold" in found.evidence
        assert "and 7 more" in found.evidence


class TestLeadingZeros:
    def test_zero_padded_values_are_reported_as_codes_and_not_as_numbers(self):
        frame = pd.DataFrame({"zip": [f"{k:05d}" for k in range(100)], "clean": numbers(100)})

        result = run(frame)

        assert kinds(result) == {"zip": "numeric_codes", "clean": "numbers_as_text"}
        codes = finding(result, "Numeric-looking columns with leading zeros")
        assert (codes.severity, codes.confidence) == (Severity.INFO, 0.6)
        assert codes.affected_columns == ("zip",)
        assert "100 values with a leading zero" in codes.evidence
        assert "cannot be undone" in codes.interpretation
        assert finding(result, "Numbers stored as text").affected_columns == ("clean",)

    def test_a_decimal_below_one_is_not_a_leading_zero(self):
        result = run(pd.DataFrame({"c": [f"0.{k}" for k in range(100)]}))

        assert kinds(result) == {"c": "numbers_as_text"}


def zero_padded_csv(tmp_path):
    """60 rows. ``zip`` is read as numbers and loses its zeros. ``sku`` has a stray ``?``, so it
    is read as text and keeps them."""
    lines = ["zip,amount,sku"]
    for k in range(60):
        zip_code = ["01234", "56789", "00042", "90210"][k % 4]
        lines.append(f"{zip_code},{100 + k},{'?' if k % 30 == 0 else f'{k:03d}'}")
    path = tmp_path / "codes.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


class TestLeadingZerosLostAtLoad:
    def test_a_column_that_lost_its_zeros_is_reported_from_the_loaders_record(self, tmp_path):
        result = check_dtypes(load_dataset(zero_padded_csv(tmp_path)), CONFIG)

        lost = finding(result, "Leading zeros were lost when the file was read")
        assert (lost.severity, lost.confidence) == (Severity.MEDIUM, 0.7)
        assert lost.affected_columns == ("zip",)
        assert "zip (30 of the 60 rows; every value was 5 characters wide)" in lost.evidence
        assert "cannot be recovered" in lost.interpretation
        assert 'text_columns=["zip"]' in lost.recommendation
        assert "--text-column zip on the command line" in lost.recommendation
        assert column(result, "zip") == {
            "name": "zip",
            "kind": "codes_lost",
            "leading_zero": 30,
            "checked": 60,
            "width": 5,
        }

    def test_a_column_is_reported_by_one_path_only(self, tmp_path):
        result = check_dtypes(load_dataset(zero_padded_csv(tmp_path)), CONFIG)

        # zip lost its zeros at load. sku is text that still has them.
        assert kinds(result) == {"zip": "codes_lost", "sku": "numeric_codes"}
        by_title = {f.title: f.affected_columns for f in result.findings}
        assert by_title == {
            "Leading zeros were lost when the file was read": ("zip",),
            "Numeric-looking columns with leading zeros": ("sku",),
        }

    def test_a_record_for_a_column_that_is_text_in_the_frame_is_ignored(self):
        frame = pd.DataFrame({"zip": [f"{k:05d}" for k in range(100)]})
        record = {"zip": {"values": 100, "checked": 100, "width": 5}}
        dataset = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, leading_zeros=record)
        )

        result = check_dtypes(dataset, CONFIG)

        assert kinds(result) == {"zip": "numeric_codes"}

    def test_the_column_that_lost_the_most_zeros_is_listed_first(self):
        frame = pd.DataFrame({"few": [1, 2, 3], "many": [4, 5, 6], "tie": [7, 8, 9]})
        record = {
            "few": {"values": 2, "checked": 3},
            "many": {"values": 9, "checked": 3},
            "tie": {"values": 2, "checked": 3},
        }
        dataset = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, leading_zeros=record)
        )

        lost = finding(
            check_dtypes(dataset, CONFIG), "Leading zeros were lost when the file was read"
        )

        text = lost.evidence
        assert text.index("many (") < text.index("few (") < text.index("tie (")

    def test_without_a_record_nothing_is_reported(self):
        assert run(pd.DataFrame({"zip": [1234, 42, 56789]})).findings == ()

    def test_a_partial_check_is_said_and_a_missing_width_is_left_out(self):
        frame = pd.DataFrame({"zip": [1234, 42, 56789]})
        record = {"zip": {"values": 3, "checked": 100_000}}
        dataset = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, leading_zeros=record)
        )

        lost = finding(
            check_dtypes(dataset, CONFIG), "Leading zeros were lost when the file was read"
        )

        assert "zip (3 of the first 100,000 rows)" in lost.evidence
        assert "characters wide" not in lost.evidence

    def test_the_recommendation_names_at_most_five_columns_and_quotes_awkward_names(self):
        names = ["zip code", "b", "c", "d", "e", "f", "g"]
        frame = pd.DataFrame({name: [1, 2, 3] for name in names})
        record = {name: {"values": 9 - i, "checked": 3} for i, name in enumerate(names)}
        dataset = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, leading_zeros=record)
        )

        lost = finding(
            check_dtypes(dataset, CONFIG), "Leading zeros were lost when the file was read"
        )

        assert "--text-column 'zip code' --text-column b --text-column c" in lost.recommendation
        assert lost.recommendation.count("--text-column") == 5
        assert "and 2 more" in lost.recommendation

    def test_a_single_row_is_not_pluralized(self):
        frame = pd.DataFrame({"zip": [1234]})
        record = {"zip": {"values": 1, "checked": 1}}
        dataset = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, leading_zeros=record)
        )

        lost = finding(
            check_dtypes(dataset, CONFIG), "Leading zeros were lost when the file was read"
        )

        assert "zip (1 of the 1 row)" in lost.evidence


class TestColumnsKeptAsTextOnRequest:
    def frames(self):
        return pd.DataFrame({"zip": [f"{k:05d}" for k in range(100)], "amount": numbers(100)})

    def test_they_are_not_reported_as_numbers_stored_as_text_or_as_codes(self):
        frame = self.frames()
        kept = Dataset(
            data=frame,
            name="t",
            provenance=Provenance.capture(frame, text_columns=("zip", "amount")),
        )

        assert check_dtypes(kept, CONFIG).findings == ()
        # The same frame without the request is reported, so the request is what silences them.
        assert kinds(run(frame)) == {"zip": "numeric_codes", "amount": "numbers_as_text"}

    def test_only_the_requested_columns_are_left_out(self):
        frame = self.frames()
        kept = Dataset(
            data=frame, name="t", provenance=Provenance.capture(frame, text_columns=("zip",))
        )

        assert kinds(check_dtypes(kept, CONFIG)) == {"amount": "numbers_as_text"}


class TestMixedTypes:
    def test_an_excel_column_of_numbers_and_text_gets_exact_type_counts(self, tmp_path):
        path = tmp_path / "mixed.xlsx"
        # "n/a" would be read as missing by the loader, so the placeholder is "unknown".
        pd.DataFrame({"v": pd.Series([1, 2, "unknown", 4.5, "x"] * 20, dtype=object)}).to_excel(
            path, index=False
        )

        result = check_dtypes(load_dataset(path), CONFIG)

        assert column(result, "v")["types"] == {"int": 40, "str": 40, "float": 20}
        found = finding(result, "Columns mix value types")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 1.0)
        assert "int 40, str 40, float 20" in found.evidence

    def test_a_json_field_that_changes_type_is_found_including_containers(self, tmp_path):
        path = tmp_path / "mixed.json"
        rows = ['{"v": 1}', '{"v": "a"}', '{"v": true}', '{"v": [1]}'] * 25
        path.write_text("[" + ",".join(rows) + "]", encoding="utf-8")

        result = check_dtypes(load_dataset(path), CONFIG)

        assert column(result, "v")["types"] == {"bool": 25, "int": 25, "list": 25, "str": 25}

    def test_numpy_scalars_are_named_like_python_ones(self):
        values = [np.bool_(True), np.int64(3), np.float64(2.5), "x"] * 25

        result = run(pd.DataFrame({"v": pd.Series(values, dtype=object)}))

        assert column(result, "v")["types"] == {"bool": 25, "float": 25, "int": 25, "str": 25}

    def test_a_column_of_one_type_is_not_mixed(self):
        frame = pd.DataFrame(
            {
                "ints": pd.Series(range(100), dtype=object),
                "dicts": pd.Series([{"k": 1}] * 100, dtype=object),
            }
        )

        assert kinds(run(frame)) == {}


class TestNothingIsConverted:
    def test_the_frame_is_left_exactly_as_it_was(self):
        frame = pd.DataFrame(
            {"price": numbers(96) + ["?"] * 4, "zip": [f"{k:05d}" for k in range(100)]}
        )
        snapshot = frame.copy(deep=True)

        result = run(frame)

        assert result.findings
        pd.testing.assert_frame_equal(frame, snapshot)


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=7, row_threshold=10)
        dataset = Dataset(data=pd.DataFrame({"c": numbers(100)}), name="t")

        first = check_dtypes(dataset, config)
        second = check_dtypes(dataset, config)

        assert first.config == config
        assert first == second
        assert AnalysisResult.from_json(first.to_json()) == first

    def test_the_size_guardrails_do_not_sample_the_rows(self):
        dataset = Dataset(data=pd.DataFrame({"c": numbers(100)}), name="t")

        result = check_dtypes(dataset, AnalysisConfig(row_threshold=10))

        assert column(result, "c")["non_null"] == 100

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            run(pd.DataFrame({"a": []}))
