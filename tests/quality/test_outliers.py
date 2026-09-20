import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality.outliers import check_outliers

CONFIG = AnalysisConfig()

# -1000, 1..100 and 1000. Quartiles by linear interpolation, derived by hand: Q1 = 25.25 (between
# the 26th and 27th sorted values), Q3 = 75.75, so the IQR is 50.5 and the fences are 1.5 and
# 3 IQRs beyond the quartiles. Only the two planted values are outside any fence.
PLANTED = np.concatenate([[-1000.0], np.arange(1, 101, dtype=float), [1000.0]])


def run(frame, target=None):
    return check_outliers(Dataset(data=frame, name="t", target=target), CONFIG)


def column(result, name):
    return next(c for c in result.metrics["columns"] if c["name"] == name)


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


class TestTheRules:
    def test_quartiles_fences_and_flags_match_the_hand_derived_values(self):
        result = run(pd.DataFrame({"planted": PLANTED}))

        c = column(result, "planted")
        assert (c["q1"], c["q3"], c["iqr"]) == (25.25, 75.75, 50.5)
        assert (c["mild_low"], c["mild_high"]) == (-50.5, 151.5)
        assert (c["extreme_low"], c["extreme_high"]) == (-126.25, 227.25)
        assert (c["iqr_flagged"], c["iqr_extreme"], c["z_flagged"], c["both_flagged"]) == (
            2,
            2,
            2,
            2,
        )
        assert (c["flagged"], c["extreme"], c["n"]) == (2, 2, 102)
        assert c["first_positions"] == [0, 101]
        found = finding(result, "Extreme values in numeric columns")
        assert (found.severity, found.confidence) == (Severity.MEDIUM, 0.7)
        assert found.affected_columns == ("planted",)
        assert "quartiles 25.25 and 75.75" in found.evidence
        assert "-126.25 to 227.25" in found.evidence

    def test_a_right_skewed_tail_is_flagged_by_the_iqr_rule_only_and_stays_a_candidate(self):
        # 1..100 plus eight 200s: Q1 27.75, Q3 81.25, IQR 53.5, fence 161.5, wide fence 241.75.
        # The mean is 61.6 and the standard deviation 48.2, so 200 is only 2.9 deviations out.
        result = run(pd.DataFrame({"skewed": list(range(1, 101)) + [200] * 8}))

        c = column(result, "skewed")
        assert (c["iqr_flagged"], c["z_flagged"], c["both_flagged"]) == (8, 0, 0)
        assert (c["flagged"], c["extreme"]) == (8, 0)
        found = finding(result, "Outlier candidates in numeric columns")
        assert (found.severity, found.confidence) == (Severity.LOW, 0.4)
        assert not [f for f in result.findings if f.title.startswith("Extreme")]
        assert "8 by the IQR rule, 0 by z-score, 0 by both" in found.evidence

    def test_a_value_between_the_fence_and_three_deviations_is_flagged_by_z_score_only(self):
        # Two tight clusters make the IQR about 100, so the fence is far out at 252.5. The value
        # 215 is 3.2 standard deviations from the mean.
        clusters = list(np.arange(50) * 0.1) + list(100 + np.arange(50) * 0.1) + [215.0]

        c = column(run(pd.DataFrame({"x": clusters})), "x")

        assert (c["iqr_flagged"], c["z_flagged"], c["both_flagged"], c["flagged"]) == (0, 1, 0, 1)
        assert c["extreme"] == 0  # extreme needs both the wide fence and the z-score

    def test_the_z_score_uses_the_sample_standard_deviation(self):
        # 29 evenly spaced values from -1 to 1 and one at 2.24. Its z-score is 2.990 with the
        # sample standard deviation (n - 1) and 3.041 with the population one (n).
        values = list(np.linspace(-1.0, 1.0, 29)) + [2.24]

        c = column(run(pd.DataFrame({"x": values})), "x")

        assert c["z_flagged"] == 0
        assert c["iqr_flagged"] == 1  # 2.24 is beyond the fence at 2.107, so it is a candidate

    def test_no_flags_means_no_findings(self):
        frame = pd.DataFrame({"u": np.random.RandomState(0).uniform(size=300)})

        assert run(frame).findings == ()

    def test_a_numeric_target_is_examined(self):
        result = run(pd.DataFrame({"y": PLANTED}), target="y")

        assert column(result, "y")["extreme"] == 2

    def test_nullable_integers_are_handled(self):
        frame = pd.DataFrame({"n": pd.array(list(range(1, 101)) + [1000, None], dtype="Int64")})

        c = column(run(frame), "n")

        assert (c["n"], c["extreme"]) == (101, 1)


class TestValuesAreNotShown:
    def test_findings_carry_counts_and_fences_but_never_the_extreme_values(self):
        values = list(np.arange(1, 101, dtype=float)) + [123456.789, -98765.4321]

        result = run(pd.DataFrame({"salary": values}))

        text = " ".join(
            f.title + f.evidence + f.interpretation + f.limitations + f.recommendation
            for f in result.findings
        )
        assert result.findings
        assert "123456" not in text
        assert "98765" not in text
        # The lowest and highest values are in the metrics, for anyone who needs them.
        c = column(result, "salary")
        assert (c["min"], c["max"]) == (-98765.4321, 123456.789)


class TestColumnsThatCannotBeChecked:
    def test_each_rule_that_would_give_nonsense_skips_the_column(self):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(
            {
                "flag": [0] * 96 + [1] * 6,
                "tiny": list(rs.normal(size=20)) + [np.nan] * 82,
                "wall": list(range(-10, 0)) + [0] * 82 + list(range(1, 11)),
            }
        )

        result = run(frame)

        reasons = {c["name"]: c["reason"] for c in result.metrics["columns"]}
        assert reasons == {"flag": "too_few_distinct", "tiny": "too_few_values", "wall": "zero_iqr"}
        assert all(
            c["status"] == "skipped" and c["flagged"] == 0 for c in result.metrics["columns"]
        )

    def test_the_note_names_the_untestable_columns_and_stays_silent_about_flags(self):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(
            {
                "planted": PLANTED,
                "flag": [0] * 96 + [1] * 6,
                "tiny": list(rs.normal(size=20)) + [np.nan] * 82,
                "wall": list(range(-10, 0)) + [0] * 82 + list(range(1, 11)),
            }
        )

        result = run(frame)

        note = finding(result, "Some numeric columns could not be checked for outliers")
        assert note.severity is Severity.INFO
        assert note.affected_columns == ("tiny", "wall")
        assert "2 of 3 numeric columns" in note.evidence  # the flag column is not counted
        assert "fewer than 30 finite values" in note.evidence
        assert "interquartile range of zero" in note.evidence
        assert "flag" not in note.affected_columns

    def test_identifier_and_boolean_columns_are_not_examined(self):
        frame = pd.DataFrame(
            {"record_id": np.arange(1000) + 1, "ok": [True, False] * 500, "v": np.arange(1000) % 97}
        )

        names = [c["name"] for c in run(frame).metrics["columns"]]

        assert names == ["v"]


class TestInfiniteValues:
    def test_they_are_left_out_counted_and_reported(self):
        frame = pd.DataFrame(
            {"x": np.concatenate([np.random.RandomState(0).normal(size=100), [np.inf, -np.inf]])}
        )

        result = run(frame)

        c = column(result, "x")
        assert (c["n"], c["n_infinite"], c["status"]) == (100, 2, "assessed")
        note = finding(result, "Infinite values were left out of the outlier statistics")
        assert note.severity is Severity.INFO
        assert "x (2)" in note.evidence

    def test_the_limitation_is_repeated_where_the_statistics_are_reported(self):
        values = np.concatenate([PLANTED, [np.inf]])

        result = run(pd.DataFrame({"x": values}))

        extreme = finding(result, "Extreme values in numeric columns")
        assert "Infinite values in 1 column were left out" in extreme.limitations


class TestAggregation:
    def test_many_columns_with_extreme_values_make_one_finding(self):
        frame = pd.DataFrame({f"c{i}": PLANTED for i in range(12)})

        result = run(frame)

        found = finding(result, "Extreme values in numeric columns")
        assert len(result.findings) == 1
        assert len(found.affected_columns) == 12
        assert "12 columns have" in found.evidence
        assert "and 7 more" in found.evidence


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=7, row_threshold=10)
        dataset = Dataset(data=pd.DataFrame({"x": PLANTED}), name="t")

        first = check_outliers(dataset, config)
        second = check_outliers(dataset, config)

        assert first.config == config
        assert first == second
        assert AnalysisResult.from_json(first.to_json()) == first

    def test_the_size_guardrails_do_not_sample_the_rows(self):
        dataset = Dataset(data=pd.DataFrame({"x": PLANTED}), name="t")

        result = check_outliers(dataset, AnalysisConfig(row_threshold=10))

        assert column(result, "x")["n"] == 102

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            run(pd.DataFrame({"a": []}))
