import numpy as np
import pandas as pd
import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import Severity
from datadoctor.diagnostics.splits import check_split_strategy
from datadoctor.testing.synthetic import (
    SyntheticConfig,
    _exact_correlation,
    _standardize,
    make_synthetic_dataset,
)

CONFIG = AnalysisConfig()
SC = SyntheticConfig(n_rows=2000, n_features=3, random_seed=42)


def titles(result):
    return [f.title for f in result.findings]


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


def imbalanced_frame(minority_rows, majority_rows):
    return pd.DataFrame(
        {
            "feature": range(minority_rows + majority_rows),
            "target": pd.Categorical(["rare"] * minority_rows + ["common"] * majority_rows),
        }
    )


class TestValidation:
    def test_zero_rows_is_rejected(self):
        dataset = Dataset(data=pd.DataFrame({"a": []}), name="empty")

        with pytest.raises(DatasetError, match="no rows"):
            check_split_strategy(dataset, CONFIG)

    def test_config_must_be_an_analysis_config(self):
        dataset = make_synthetic_dataset(SC).dataset

        with pytest.raises(TypeError, match="AnalysisConfig"):
            check_split_strategy(dataset, {"row_threshold": 5})


class TestCleanDataset:
    def test_no_findings_on_a_dataset_with_no_planted_structure(self):
        result = check_split_strategy(make_synthetic_dataset(SC).dataset, CONFIG)

        assert result.findings == ()


class TestNoTarget:
    def test_a_datetime_candidate_with_no_target_does_not_crash(self):
        frame = pd.DataFrame(
            {"signup_date": pd.date_range("2020-01-01", periods=200), "value": range(200)}
        )
        dataset = Dataset(data=frame, name="no_target")

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["temporal"] == {
            "candidates": ["signup_date"],
            "effects": {},
            "flagged": [],
        }
        assert titles(result) == []

    def test_a_group_candidate_with_no_target_does_not_crash(self):
        frame = pd.DataFrame({"group_id": [f"g{i % 20}" for i in range(200)], "value": range(200)})
        dataset = Dataset(data=frame, name="no_target")

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["groups"] == {
            "candidates": ["group_id"],
            "effects": {},
            "flagged": [],
        }
        assert titles(result) == []


class TestTemporal:
    def test_fires_for_a_regression_target_drifting_over_time(self):
        dataset = make_synthetic_dataset(
            SC, temporal=True, task="regression", effect_sizes={"temporal": 0.9}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "A random split would leak future information")
        assert found.severity == Severity.HIGH
        assert found.confidence == 1.0
        assert found.affected_columns == ("event_time",)

    def test_fires_for_a_classification_target_drifting_over_time(self):
        dataset = make_synthetic_dataset(
            SC, temporal=True, task="classification", effect_sizes={"temporal": 0.9}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert "A random split would leak future information" in titles(result)

    def test_regression_effect_matches_the_requested_correlation_exactly(self):
        dataset = make_synthetic_dataset(
            SC, temporal=True, task="regression", effect_sizes={"temporal": 0.7}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["temporal"]["effects"]["event_time"] == pytest.approx(0.7, abs=1e-9)

    def test_a_weak_drift_does_not_fire(self):
        dataset = make_synthetic_dataset(
            SC, temporal=True, task="regression", effect_sizes={"temporal": 0.02}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert "A random split would leak future information" not in titles(result)
        assert result.metrics["temporal"]["candidates"] == ["event_time"]
        assert result.metrics["temporal"]["flagged"] == []

    def test_an_effect_below_min_effect_does_not_fire_even_if_significant(self):
        # At n=2000, r=0.05 is small (below MIN_EFFECT=0.1) but has enough power to be
        # statistically significant on its own, so this isolates the effect-size gate from the
        # significance gate: a test using a effect small enough to also be non-significant would
        # pass even with the effect-size gate deleted entirely.
        dataset = make_synthetic_dataset(
            SC, temporal=True, task="regression", effect_sizes={"temporal": 0.05}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["temporal"]["flagged"] == []

    def test_a_borderline_candidate_needs_bonferroni_correction_to_survive(self):
        # date_a's raw p-value (0.03) alone clears ALPHA=0.05, but with a second candidate
        # column tested in the same run, the Bonferroni-adjusted p (0.03 * 2 = 0.06) does not.
        # A weak-effect or non-significant fixture would pass this test even with the
        # multiple-comparison correction deleted entirely, so this isolates it specifically.
        n = 300
        rng = np.random.RandomState(3)
        target = _standardize(rng.normal(size=n))
        # r = 0.1253... is the Pearson correlation that gives p = 0.03 at n = 300, found by
        # inverting scipy's own pearsonr significance formula.
        date_a_rank = _exact_correlation(target, 0.12531906602756793, rng)
        date_b_rank = rng.normal(size=n)
        epoch = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {
                "date_a": epoch + pd.to_timedelta((date_a_rank * 100).astype("int64"), unit="D"),
                "date_b": epoch + pd.to_timedelta((date_b_rank * 100).astype("int64"), unit="D"),
                "target": target,
            }
        )
        dataset = Dataset(data=frame, name="bonferroni_check", target="target")

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["temporal"]["effects"]["date_a"] == pytest.approx(0.1253, abs=1e-3)
        assert result.metrics["temporal"]["flagged"] == []


class TestGroups:
    def test_fires_for_a_regression_target_correlated_within_group(self):
        dataset = make_synthetic_dataset(
            SC, groups=40, task="regression", effect_sizes={"group": 0.9}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "A random split could leak an entity across train and test")
        assert found.severity == Severity.HIGH
        assert found.affected_columns == ("group_id",)

    def test_fires_for_a_classification_target_correlated_within_group(self):
        dataset = make_synthetic_dataset(
            SC, groups=40, task="classification", effect_sizes={"group": 0.9}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert "A random split could leak an entity across train and test" in titles(result)

    def test_regression_effect_is_the_correlation_ratio_not_pearson_on_the_codes(self):
        # A regression target against a categorical group column must use the correlation ratio
        # (target as the continuous variable, group as the grouping factor). Pearson correlation
        # computed directly against pandas.factorize's arbitrary integer codes is a different,
        # meaningless number that happens to also clear the effect-size gate on some data, which
        # is what made this bug hard to notice by "does it fire" alone.
        dataset = make_synthetic_dataset(
            SC, groups=40, task="regression", effect_sizes={"group": 0.7}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["groups"]["effects"]["group_id"] == pytest.approx(0.7**0.5, abs=1e-9)

    def test_an_effect_below_min_effect_does_not_fire_even_if_significant(self):
        # Fewer groups means fewer degrees of freedom consumed by the F-test, so a small
        # correlation ratio can still be significant here (unlike the 40-group cases above),
        # isolating the effect-size gate from the significance gate the same way the analogous
        # temporal test does.
        dataset = make_synthetic_dataset(
            SC, groups=5, task="regression", effect_sizes={"group": 0.008}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["groups"]["effects"]["group_id"] < 0.1
        assert result.metrics["groups"]["flagged"] == []

    def test_confidence_is_the_identifier_heuristics_own_confidence_not_full(self):
        dataset = make_synthetic_dataset(
            SC, groups=40, task="regression", effect_sizes={"group": 0.9}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "A random split could leak an entity across train and test")
        assert found.confidence < 1.0

    def test_a_weak_group_effect_does_not_fire(self):
        dataset = make_synthetic_dataset(
            SC, groups=40, task="regression", effect_sizes={"group": 0.02}
        ).dataset

        result = check_split_strategy(dataset, CONFIG)

        assert "A random split could leak an entity across train and test" not in titles(result)
        assert result.metrics["groups"]["candidates"] == ["group_id"]
        assert result.metrics["groups"]["flagged"] == []

    def test_a_near_unique_identifier_is_not_a_group_candidate(self):
        # Mirrors data/telecom_customers.csv's customer_id: 629 distinct values in 639 rows,
        # essentially one customer per row with a handful of incidental repeats. Without a
        # minimum average group size, the correlation ratio (and Cramer's V) on almost-singleton
        # groups is trivially near 1.0 -- a group of size 1 perfectly "explains" its own single
        # value -- which would make any near-unique identifier a guaranteed false positive.
        n_rows = 639
        customer_id = [f"cust_{i}" for i in range(629)] + [f"cust_{i}" for i in range(10)]
        frame = pd.DataFrame({"customer_id": customer_id, "target": np.arange(n_rows) % 2})
        dataset = Dataset(data=frame, name="near_unique", target="target")

        result = check_split_strategy(dataset, CONFIG)

        assert "A random split could leak an entity across train and test" not in titles(result)
        assert result.metrics["groups"]["candidates"] == []


class TestSmallN:
    def test_just_below_the_floor_still_fires(self):
        # 140 rows gives an expected test size of 28, just under MIN_TEST_ROWS=30. A test using
        # a value far from the floor would pass even if the floor itself were weakened.
        dataset = Dataset(data=pd.DataFrame({"a": range(140)}), name="boundary")

        result = check_split_strategy(dataset, CONFIG)

        assert "Too few rows for a stable train/test split" in titles(result)

    def test_far_below_the_floor_is_high_severity(self):
        dataset = Dataset(data=pd.DataFrame({"a": range(50)}), name="tiny")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Too few rows for a stable train/test split")
        assert found.severity == Severity.HIGH

    def test_marginally_below_the_floor_is_medium_severity(self):
        dataset = Dataset(data=pd.DataFrame({"a": range(100)}), name="small")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Too few rows for a stable train/test split")
        assert found.severity == Severity.MEDIUM

    def test_enough_rows_does_not_fire(self):
        dataset = Dataset(data=pd.DataFrame({"a": range(200)}), name="plenty")

        result = check_split_strategy(dataset, CONFIG)

        assert "Too few rows for a stable train/test split" not in titles(result)

    def test_runs_without_a_target(self):
        dataset = Dataset(data=pd.DataFrame({"a": range(10)}), name="no_target")

        result = check_split_strategy(dataset, CONFIG)

        assert "Too few rows for a stable train/test split" in titles(result)


class TestStratification:
    def test_a_small_minority_class_fires(self):
        dataset = Dataset(data=imbalanced_frame(6, 194), name="rare", target="target")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Stratified splitting is recommended for this target")
        assert found.severity == Severity.MEDIUM
        assert "3.0%" in found.evidence

    def test_the_finding_names_the_target_column(self):
        # diagnostics.recommendations (S29) matches this finding against eda's class-imbalance
        # finding by affected_columns; an empty tuple here would make that match vacuous.
        dataset = Dataset(data=imbalanced_frame(6, 194), name="rare", target="target")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Stratified splitting is recommended for this target")
        assert found.affected_columns == ("target",)

    def test_a_moderately_small_minority_class_is_low_severity(self):
        dataset = Dataset(data=imbalanced_frame(16, 184), name="mild", target="target")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Stratified splitting is recommended for this target")
        assert found.severity == Severity.LOW

    def test_a_balanced_target_does_not_fire(self):
        dataset = Dataset(data=imbalanced_frame(100, 100), name="balanced", target="target")

        result = check_split_strategy(dataset, CONFIG)

        assert "Stratified splitting is recommended for this target" not in titles(result)

    def test_a_continuous_regression_target_is_not_checked_for_stratification(self):
        dataset = make_synthetic_dataset(SC, task="regression").dataset

        result = check_split_strategy(dataset, CONFIG)

        assert result.metrics["stratification"] is None

    def test_a_discrete_numeric_target_is_checked_regardless_of_task(self):
        # A whole-number rating column, such as wine quality, is typed numeric by the schema
        # profile and so is a regression target for association purposes, but it still gets the
        # classification-style imbalance check, the same way eda.relationships shows its
        # class-balance view independently of task.
        frame = pd.DataFrame(
            {
                "feature": range(1000),
                "rating": [3] + [6] * 999,
            }
        )
        dataset = Dataset(data=frame, name="ratings", target="rating")

        result = check_split_strategy(dataset, CONFIG)

        found = finding(result, "Stratified splitting is recommended for this target")
        assert found.severity == Severity.MEDIUM
