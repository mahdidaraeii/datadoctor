import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality.duplicates import check_duplicates

CONFIG = AnalysisConfig()


def unique_rows(n):
    """n rows that all differ: a is unique but not sequential, so it is not an identifier."""
    return pd.DataFrame(
        {
            "a": (np.arange(n) * 37) % 5000,
            "b": np.arange(n) % 13,
            "c": (["x", "y", "z"] * (n // 3 + 1))[:n],
        }
    )


def with_copies(frame, positions):
    return pd.concat([frame, frame.iloc[positions]], ignore_index=True)


def run(frame, target=None):
    return check_duplicates(Dataset(data=frame, name="t", target=target), CONFIG)


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


class TestRows:
    def test_repeats_are_counted_with_their_groups_and_positions(self):
        # Rows 0 to 5 are each copied 5 more times: 30 extra rows in 6 groups of 6.
        frame = with_copies(unique_rows(976), [k for k in range(6) for _ in range(5)])

        result = run(frame)

        rows = result.metrics["rows"]
        assert (rows["duplicate_rows"], rows["duplicate_groups"], rows["largest_group"]) == (
            30,
            6,
            6,
        )
        assert rows["rate"] == 30 / 1006
        assert rows["first_positions"] == [976, 977, 978, 979, 980]
        assert rows["compared_columns"] == 3
        assert (
            finding(result, "Duplicate rows").severity,
            finding(result, "Duplicate rows").confidence,
        ) == (
            Severity.MEDIUM,
            1.0,
        )

    def test_severity_grades_by_the_share_of_extra_copies_with_inclusive_bounds(self):
        expected = {
            9: Severity.LOW,
            10: Severity.MEDIUM,
            99: Severity.MEDIUM,
            100: Severity.HIGH,
            499: Severity.HIGH,
            500: Severity.CRITICAL,
        }
        found = {}
        for extra in expected:
            frame = with_copies(unique_rows(1000 - extra), [0] * extra)
            found[extra] = finding(run(frame), "Duplicate rows").severity

        assert found == expected

    def test_missing_values_in_the_same_place_make_rows_identical(self):
        same = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [np.nan, np.nan, np.nan]})
        different = pd.DataFrame({"a": [1.0, 1.0], "b": [np.nan, 5.0]})

        assert run(same).metrics["rows"]["duplicate_rows"] == 1
        assert run(different).metrics["rows"]["duplicate_rows"] == 0

    def test_none_and_nan_are_the_same_missing_value(self):
        # Compared as raw values, pandas treats None and NaN in an object column as different.
        frame = pd.DataFrame({"a": [1, 2, 2], "o": pd.Series(["x", None, np.nan], dtype=object)})

        assert run(frame).metrics["rows"]["duplicate_rows"] == 1

    def test_negative_zero_equals_zero(self):
        assert run(pd.DataFrame({"a": [0.0, -0.0, 1.0]})).metrics["rows"]["duplicate_rows"] == 1

    def test_no_repeats_gives_no_findings(self):
        result = run(unique_rows(200))

        assert result.findings == ()
        assert result.metrics["rows"]["duplicate_rows"] == 0
        assert result.metrics["rows_ignoring_identifiers"] is None


class TestHiddenByIdentifiers:
    def with_id(self, frame, column="record_id", start=1):
        frame = frame.copy()
        frame.insert(0, column, np.arange(len(frame)) + start)
        return frame

    def test_a_unique_id_hides_repeats_and_ignoring_it_reveals_them(self):
        content = with_copies(unique_rows(976), [k for k in range(6) for _ in range(5)])
        frame = self.with_id(content)

        result = run(frame)

        hidden = finding(result, "Duplicate rows hidden by identifier columns")
        schema = profile_schema(Dataset(data=frame, name="t")).metrics["columns"]
        assert result.metrics["rows"]["duplicate_rows"] == 0  # every row differs by its id
        assert result.metrics["rows_ignoring_identifiers"]["duplicate_rows"] == 30
        assert result.metrics["rows_ignoring_identifiers"]["identifier_columns"] == ["record_id"]
        assert hidden.confidence == schema[0]["identifier_confidence"] == 0.95
        assert hidden.severity is Severity.MEDIUM
        assert "record_id" in hidden.evidence
        assert not [f for f in result.findings if f.title == "Duplicate rows"]

    def test_it_says_nothing_when_ignoring_ids_reveals_nothing(self):
        result = run(self.with_id(unique_rows(200)))

        assert result.metrics["rows_ignoring_identifiers"]["duplicate_rows"] == 0
        assert result.findings == ()

    def test_confidence_is_the_lowest_among_the_ignored_identifier_columns(self):
        content = with_copies(unique_rows(300), [0] * 10)
        frame = self.with_id(self.with_id(content, "record_id", 1000), "row_number", 1)

        result = run(frame)

        # record_id is named and unique (0.95). row_number is only sequential (0.6).
        hidden = finding(result, "Duplicate rows hidden by identifier columns")
        assert hidden.confidence == 0.6


class TestDuplicateIdentifiers:
    def orders(self, ids, amounts):
        base = pd.DataFrame({"order_id": np.arange(994), "amount": np.arange(994) * 1.5 + 0.25})
        extras = pd.DataFrame({"order_id": ids, "amount": amounts})
        return pd.concat([base, extras], ignore_index=True)

    def test_repeated_ids_split_into_exact_copies_and_conflicts(self):
        # id 5 three times and id 6 twice, all exact copies. id 7 twice and id 8 three times, with
        # rows that differ. 6 extra rows in 4 groups.
        frame = self.orders([5, 5, 6, 7, 8, 8], [7.75, 7.75, 9.25, 1_000_000.5, 12.25, 2_000_000.5])

        result = run(frame)

        (ids,) = result.metrics["identifiers"]
        assert (ids["duplicated_values"], ids["extra_rows"]) == (4, 6)
        assert (ids["exact_copy_groups"], ids["conflicting_groups"]) == (2, 2)
        assert ids["rate"] == 0.006
        assert result.metrics["rows"]["duplicate_rows"] == 4  # the exact copies only
        found = finding(result, "Repeated values in identifier column order_id")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 0.95)
        assert found.affected_columns == ("order_id",)
        assert "integrity problem" in found.interpretation

    def test_severity_follows_the_share_of_rows_and_whether_rows_conflict(self):
        def severity(extra, conflicting):
            ids = list(range(extra))
            amounts = [k * 1.5 + 0.25 + (1e6 if conflicting else 0) for k in ids]
            base = pd.DataFrame(
                {
                    "order_id": np.arange(1000 - extra),
                    "amount": np.arange(1000 - extra) * 1.5 + 0.25,
                }
            )
            frame = pd.concat([base, pd.DataFrame({"order_id": ids, "amount": amounts})])
            title = "Repeated values in identifier column order_id"
            return finding(run(frame.reset_index(drop=True)), title).severity

        assert severity(5, conflicting=True) is Severity.MEDIUM  # 0.5% of rows
        assert severity(20, conflicting=False) is Severity.HIGH  # 2%, exact copies
        assert severity(20, conflicting=True) is Severity.CRITICAL  # 2%, conflicting data

    def test_missing_ids_are_not_repeats(self):
        ids = np.concatenate([np.arange(997, dtype=float), [np.nan] * 3])
        frame = pd.DataFrame({"order_id": ids, "amount": np.arange(1000) * 1.5})

        result = run(frame)

        assert result.metrics["identifiers"][0]["duplicated_values"] == 0
        assert result.findings == ()

    def test_a_name_only_candidate_is_not_checked_because_it_repeats_legitimately(self):
        frame = pd.DataFrame({"parent_id": np.arange(1000) % 5, "amount": np.arange(1000) * 1.5})

        assert run(frame).metrics["identifiers"] == []

    def test_values_are_never_printed(self):
        ids = np.arange(994) + 900000
        base = pd.DataFrame({"order_id": ids, "amount": np.arange(994) * 1.5 + 0.25})
        repeat = pd.DataFrame({"order_id": [900005, 900005], "amount": [9.0, 9.5]})

        result = run(pd.concat([base, repeat], ignore_index=True))

        assert result.findings
        text = " ".join(f.title + f.evidence for f in result.findings)
        assert "900005" not in text


class TestColumnsLeftOut:
    def test_nested_columns_are_left_out_and_named(self):
        frame = pd.DataFrame(
            {"a": [1, 1, 2], "b": ["x", "x", "y"], "nested": [{"k": 1}, {"k": 2}, {"k": 3}]}
        )

        result = run(frame)

        rows = result.metrics["rows"]
        assert (rows["left_out"], rows["compared_columns"]) == (["nested"], 2)
        assert rows["duplicate_rows"] == 1  # rows 0 and 1 differ only in the nested column
        note = finding(result, "Some columns could not be compared")
        assert note.severity is Severity.INFO
        assert note.affected_columns == ("nested",)
        assert "left out because their values cannot be compared" in (
            finding(result, "Duplicate rows").limitations
        )

    def test_a_frame_with_nothing_comparable_is_reported_not_treated_as_clean(self):
        result = run(pd.DataFrame({"nested": [{"k": 1}, {"k": 1}]}))

        (note,) = result.findings
        assert "No column could be compared" in note.evidence
        assert result.metrics["rows"]["duplicate_rows"] == 0


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        frame = with_copies(unique_rows(300), [0, 0, 1])
        config = AnalysisConfig(random_seed=7, row_threshold=10)

        first = check_duplicates(Dataset(data=frame, name="t"), config)
        second = check_duplicates(Dataset(data=frame, name="t"), config)

        assert first.config == config
        assert first == second
        assert AnalysisResult.from_json(first.to_json()) == first

    def test_the_size_guardrails_do_not_sample_the_rows(self):
        frame = with_copies(unique_rows(300), [0] * 5)

        result = check_duplicates(Dataset(data=frame, name="t"), AnalysisConfig(row_threshold=10))

        assert result.metrics["rows"]["duplicate_rows"] == 5

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            run(pd.DataFrame({"a": []}))
