import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Provenance, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality.impossible import check_impossible_values

CONFIG = AnalysisConfig()
AS_OF = dt.date(2025, 6, 1)
N = 200


def run(frame, as_of=AS_OF):
    return check_impossible_values(Dataset(data=frame, name="t"), CONFIG, as_of=as_of)


def check(result, name):
    return next(c for c in result.metrics["columns"] if c["name"] == name)


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


def varied(negatives=0, base=None):
    """N values that vary, with the first ones replaced by -1."""
    values = list(base if base is not None else np.arange(N) % 90 + 1)
    for k in range(negatives):
        values[k] = -1
    return values


class TestWhichColumnsAreChecked:
    def test_names_match_as_whole_words_not_as_substrings(self):
        columns = ["customer_age", "AgeYears", "AGE", "page", "stage", "average", "message"]
        frame = pd.DataFrame({name: varied(negatives=1) for name in columns})

        result = run(frame)

        assert {c["name"] for c in result.metrics["columns"]} == {"customer_age", "AgeYears", "AGE"}

    def test_names_where_a_negative_value_can_be_legitimate_are_left_alone(self):
        columns = ["balance", "amount", "price", "change", "quantity", "temperature"]
        frame = pd.DataFrame({name: varied(negatives=5) for name in columns})

        assert run(frame).metrics["columns"] == []
        assert run(frame).findings == ()

    def test_other_non_negative_quantities_are_checked(self):
        frame = pd.DataFrame({"height": varied(negatives=1), "word_count": varied(negatives=1)})

        result = run(frame)

        assert {c["name"] for c in result.metrics["columns"]} == {"height", "word_count"}
        found = finding(result, "Negative values in height")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 0.7)

    def test_text_columns_are_not_checked_for_negative_numbers(self):
        frame = pd.DataFrame({"age": [str(k) for k in varied(negatives=3)]})

        assert run(frame).metrics["columns"] == []


class TestAges:
    def test_negative_is_impossible_and_above_120_is_implausible(self):
        # -5, -1 and 0 (zero is fine), then 120 (the bound is inclusive), 121 and 130.
        base = list(np.arange(N - 6) % 90 + 1) + [0, 120, 121, 130, 100, 100]
        base[0], base[1] = -5, -1
        result = run(pd.DataFrame({"age": base}))

        c = check(result, "age")
        assert (c["negative"], c["above_maximum"], c["checked"]) == (2, 2, N)
        negative = finding(result, "Negative values in age column age")
        high = finding(result, "Implausibly high values in age column age")
        assert (negative.severity, negative.confidence) == (Severity.MEDIUM, 0.9)
        assert (high.severity, high.confidence) == (Severity.LOW, 0.6)
        assert c["min"] == -5.0 and c["max"] == 130.0

    def test_severity_rises_to_high_from_five_percent_of_the_values(self):
        def severity(count):
            result = run(pd.DataFrame({"age": varied(negatives=count)}))
            return finding(result, "Negative values in age column age").severity

        assert severity(9) is Severity.MEDIUM  # 4.5% of 200
        assert severity(10) is Severity.HIGH  # 5.0%

    def test_the_grade_also_applies_to_other_quantities(self):
        result = run(pd.DataFrame({"height": varied(negatives=10)}))

        assert finding(result, "Negative values in height").severity is Severity.HIGH

    def test_a_clean_age_column_gives_no_finding(self):
        assert run(pd.DataFrame({"age": varied()})).findings == ()

    def test_values_are_never_printed(self):
        values = varied()
        values[0] = -777
        result = run(pd.DataFrame({"age": values}))

        assert result.findings
        assert "777" not in " ".join(f.title + f.evidence for f in result.findings)
        assert check(result, "age")["min"] == -777.0


class TestFutureDates:
    def frame(self, values):
        return pd.DataFrame({"seen": values})

    def test_a_date_equal_to_the_reference_is_not_the_future_but_the_next_day_is(self):
        dates = ["2025-06-01"] * 99 + ["2025-06-02"]

        c = check(run(self.frame(dates)), "seen")

        assert (c["future"], c["checked"], c["rate"]) == (1, 100, 0.01)
        assert c["as_of"] == "2025-06-01"

    def test_iso_text_naive_datetimes_and_date_objects_all_work(self):
        text = ["2024-01-01"] * 98 + ["2031-05-01", "2027-02-02"]
        naive = list(pd.to_datetime(["2024-01-01"] * 98 + ["2031-05-01", "2027-02-02"]))
        objects = [dt.date(2024, 1, 1)] * 98 + [dt.date(2031, 5, 1), dt.date(2027, 2, 2)]
        frame = pd.DataFrame(
            {
                "text": text,
                "naive": pd.Series(naive),
                "objects": pd.Series(objects, dtype=object),
            }
        )

        result = run(frame)

        assert {c["name"]: c["future"] for c in result.metrics["columns"]} == {
            "text": 2,
            "naive": 2,
            "objects": 2,
        }

    def test_far_future_dates_behave_the_same_on_every_pandas_version(self):
        # pandas 2.3 cannot represent 9999-12-31 or 3000-01-01 as datetime64[ns] and turns them
        # into NaT, so the check must not depend on pandas date parsing.
        text = ["2024-01-01"] * 96 + ["9999-12-31", "3000-01-01", "2262-04-12", "2262-04-11"]
        objects = [dt.datetime(2024, 1, 1)] * 98 + [
            dt.datetime(9999, 12, 31),
            dt.datetime(3000, 1, 1),
        ]
        frame = pd.DataFrame({"text": text, "objects": pd.Series(objects, dtype=object)})

        result = run(frame)

        assert check(result, "text")["future"] == 4
        assert check(result, "text")["latest"] == "9999-12-31"
        assert check(result, "objects")["future"] == 2

    def test_the_time_of_day_is_ignored_for_text_dates(self):
        dates = ["2025-06-01T23:59:59"] * 99 + ["2025-06-01 00:00:00"]

        assert check(run(self.frame(dates)), "seen")["future"] == 0

    def test_timezone_aware_values_are_converted_to_utc_first(self):
        # 22:00 in New York on June 1 is already June 2 in UTC, so it is later than June 1.
        aware = pd.to_datetime(["2025-05-01T12:00:00"] * 99 + ["2025-06-01T22:00:00"])
        aware = pd.Series(aware).dt.tz_localize("America/New_York")

        assert check(run(self.frame(aware)), "seen")["future"] == 1

    @pytest.mark.parametrize("name", ["due_date", "expiry", "next_visit", "renewalDate", "eta"])
    def test_columns_that_are_expected_to_hold_future_dates_are_not_checked(self, name):
        frame = pd.DataFrame({name: ["2099-01-01"] * 100})

        assert run(frame).metrics["columns"] == []

    def test_a_text_column_with_a_placeholder_is_not_a_date_column(self):
        values = ["2031-05-01"] * 99 + ["unknown"]

        assert run(self.frame(values)).metrics["columns"] == []

    def test_one_placeholder_far_down_a_long_column_still_rules_it_out(self):
        # The first values all look like dates. Only the full scan sees the last one.
        values = ["2031-05-01"] * 1500 + ["unknown"]

        assert run(self.frame(values)).metrics["columns"] == []

    def test_a_finding_gives_the_reference_date_and_counts_but_not_the_dates(self):
        result = run(self.frame(["2024-01-01"] * 99 + ["2031-05-07"]))

        found = finding(result, "Dates in the future in seen")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 0.5)
        assert "later than 2025-06-01" in found.evidence
        assert "2031-05-07" not in found.evidence + found.interpretation + found.limitations
        assert check(result, "seen")["latest"] == "2031-05-07"  # in the metrics

    def test_severity_rises_to_high_from_five_percent(self):
        def severity(future):
            dates = ["2024-01-01"] * (100 - future) + ["2031-01-01"] * future
            return finding(run(self.frame(dates)), "Dates in the future in seen").severity

        assert severity(4) is Severity.MEDIUM
        assert severity(5) is Severity.HIGH


class TestDatesThatDoNotExist:
    def test_impossible_calendar_dates_are_counted_and_the_rest_still_checked(self):
        dates = ["2024-01-01"] * 95 + [
            "2024-02-30",
            "2024-13-01",
            "2031-05-01",
            "2027-02-02",
            "2029-01-01",
        ]

        result = run(pd.DataFrame({"d": dates}))

        c = check(result, "d")
        assert (c["invalid"], c["future"], c["checked"]) == (2, 3, 100)
        invalid = finding(result, "Dates that do not exist in d")
        assert (invalid.severity, invalid.confidence) == (Severity.MEDIUM, 0.9)
        assert "2 values (2.0% of 100)" in invalid.evidence
        assert finding(result, "Dates in the future in d").severity is Severity.MEDIUM

    def test_five_percent_of_them_is_high(self):
        dates = ["2024-01-01"] * 95 + ["2024-02-30"] * 5

        result = run(pd.DataFrame({"d": dates}))

        assert finding(result, "Dates that do not exist in d").severity is Severity.HIGH


class TestTheReferenceDate:
    def dataset(self, loaded_at):
        frame = pd.DataFrame({"seen": ["2025-06-01"] * 99 + ["2025-06-02"]})
        provenance = dataclasses.replace(Provenance.capture(frame), loaded_at=loaded_at)
        return Dataset(data=frame, name="t", provenance=provenance)

    def test_it_defaults_to_the_load_date_and_never_the_wall_clock(self):
        result = check_impossible_values(self.dataset("2025-06-01T10:00:00Z"), CONFIG)

        assert result.metrics["as_of"] == "2025-06-01"
        assert check(result, "seen")["future"] == 1

    def test_the_same_data_loaded_later_is_judged_against_the_later_date(self):
        result = check_impossible_values(self.dataset("2025-06-02T10:00:00Z"), CONFIG)

        assert check(result, "seen")["future"] == 0

    def test_an_explicit_date_wins_and_a_datetime_is_reduced_to_its_date(self):
        dataset = self.dataset("2030-01-01T00:00:00Z")

        by_date = check_impossible_values(dataset, CONFIG, as_of=dt.date(2025, 6, 1))
        by_datetime = check_impossible_values(dataset, CONFIG, as_of=dt.datetime(2025, 6, 1, 23, 0))

        assert by_date == by_datetime
        assert by_date.metrics["as_of"] == "2025-06-01"

    def test_it_must_be_a_date(self):
        with pytest.raises(TypeError, match="as_of must be a date"):
            check_impossible_values(
                self.dataset("2025-06-01T10:00:00Z"), CONFIG, as_of="2025-06-01"
            )


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=7, row_threshold=10)
        dataset = Dataset(data=pd.DataFrame({"age": varied(negatives=3)}), name="t")

        first = check_impossible_values(dataset, config, as_of=AS_OF)
        second = check_impossible_values(dataset, config, as_of=AS_OF)

        assert first.config == config
        assert first == second
        assert AnalysisResult.from_json(first.to_json()) == first

    def test_the_size_guardrails_do_not_sample_the_rows(self):
        dataset = Dataset(data=pd.DataFrame({"age": varied(negatives=3)}), name="t")

        result = check_impossible_values(dataset, AnalysisConfig(row_threshold=10), as_of=AS_OF)

        assert check(result, "age")["checked"] == N

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            run(pd.DataFrame({"a": []}))
