import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality.constants import check_constants

CONFIG = AnalysisConfig()
M = 1000


def planted_frame():
    return pd.DataFrame(
        {
            "const_str": ["x"] * M,
            "const_num": [5.0] * M,
            "const_missing": [np.nan] * 300 + [1.0] * 700,
            "near_995": [0] * 995 + [1] * 5,
            "edge_990": [0] * 990 + [1] * 10,
            "edge_989": [0] * 989 + [1] * 11,
            "all_null": [np.nan] * M,
            "varied": np.random.RandomState(0).normal(size=M),
            "nested": [{"k": 1}] * M,
            "neg_zero": [0.0, -0.0] * (M // 2),
        }
    )


def run(frame, target=None):
    return check_constants(Dataset(data=frame, name="t", target=target), CONFIG)


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


@pytest.fixture(scope="module")
def result():
    return run(planted_frame())


class TestClassification:
    def test_every_planted_column_gets_its_kind(self, result):
        kinds = {c["name"]: c["kind"] for c in result.metrics["columns"]}

        assert kinds == {
            "const_str": "constant",
            "const_num": "constant",
            "const_missing": "constant",  # constant among the values it has
            "near_995": "near_constant",
            "edge_990": "near_constant",  # exactly 99.0%: the cutoff is inclusive
            "edge_989": "varying",  # 98.9%: just below it
            "all_null": "all_null",
            "varied": "varying",
            "nested": "not_checked",
            "neg_zero": "constant",  # 0.0 and -0.0 are one value
        }

    def test_counts_and_shares_are_exact(self, result):
        columns = {c["name"]: c for c in result.metrics["columns"]}

        assert columns["edge_990"]["top_share"] == 0.99
        assert columns["near_995"]["top_share"] == 0.995
        assert (columns["const_missing"]["non_null"], columns["const_missing"]["missing"]) == (
            700,
            300,
        )
        assert columns["const_missing"]["distinct"] == 1
        assert columns["nested"]["distinct"] is None


class TestFindings:
    def test_constants_are_one_finding_that_names_every_column_and_notes_missing_values(
        self, result
    ):
        found = finding(result, "Constant columns")

        assert found.severity is Severity.LOW
        assert found.affected_columns == ("const_str", "const_num", "const_missing", "neg_zero")
        assert "4 columns" in found.evidence
        assert "1 of them also has missing values" in found.evidence

    def test_near_constants_are_reported_with_the_caveat_that_skew_can_be_legitimate(self, result):
        found = finding(result, "Near-constant columns")

        assert found.severity is Severity.INFO
        assert found.affected_columns == ("near_995", "edge_990")
        assert "near_995 (99.5%)" in found.evidence
        assert "not a defect by itself" in found.interpretation
        assert "Zero-inflated counts" in found.interpretation
        assert "rare-event flags" in found.interpretation
        assert "convention" in found.limitations

    def test_a_column_with_no_values_is_left_to_the_missingness_diagnostics(self, result):
        assert not [f for f in result.findings if "all_null" in f.affected_columns]

    def test_columns_that_cannot_be_compared_are_reported_not_skipped_silently(self, result):
        found = finding(result, "Some columns could not be checked")

        assert found.severity is Severity.INFO
        assert found.affected_columns == ("nested",)
        assert "1 column holds values" in found.evidence

    def test_many_constant_columns_make_one_finding(self):
        frame = pd.DataFrame({f"k{i}": [1] * 50 for i in range(12)} | {"v": range(50)})

        result = run(frame)

        found = finding(result, "Constant columns")
        assert len(result.findings) == 1
        assert len(found.affected_columns) == 12
        assert "12 columns" in found.evidence
        assert "and 7 more" in found.evidence

    def test_nothing_constant_gives_no_findings(self):
        frame = pd.DataFrame({"a": np.arange(100) % 7, "b": np.arange(100) % 5})

        assert run(frame).findings == ()


class TestTarget:
    def test_a_constant_target_is_critical_and_not_repeated_among_the_constants(self):
        result = run(planted_frame(), target="const_str")

        target = finding(result, "Target column is constant")
        assert target.severity is Severity.CRITICAL
        assert target.affected_columns == ("const_str",)
        assert "const_str" not in finding(result, "Constant columns").affected_columns

    def test_a_near_constant_target_is_not_flagged_because_that_is_class_imbalance(self):
        result = run(planted_frame(), target="near_995")

        assert "near_995" not in finding(result, "Near-constant columns").affected_columns
        column = next(c for c in result.metrics["columns"] if c["name"] == "near_995")
        assert (column["kind"], column["is_target"]) == ("near_constant", True)


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=7, row_threshold=10)

        first = check_constants(Dataset(data=planted_frame(), name="t"), config)
        second = check_constants(Dataset(data=planted_frame(), name="t"), config)

        assert first.config == config
        assert first == second
        assert AnalysisResult.from_json(first.to_json()) == first

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            run(pd.DataFrame({"a": []}))
