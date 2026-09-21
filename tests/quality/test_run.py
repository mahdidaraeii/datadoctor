import datetime

import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality import (
    check_constants,
    check_dtypes,
    check_duplicates,
    check_impossible_values,
    check_missingness,
    check_outliers,
    run_quality_checks,
)

N = 200
CONFIG = AnalysisConfig()
AS_OF = datetime.date(2025, 1, 1)
CHECKS = [
    "missingness",
    "duplicates",
    "constants",
    "outliers",
    "dtypes",
    "impossible_values",
]


def clean_frame() -> pd.DataFrame:
    """Columns with nothing wrong: varied values, no repeats, nothing missing."""
    # Uniform values, because normal ones always put a few points beyond the outlier fences.
    rs = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "height": rs.uniform(150, 190, size=N).round(1),
            "weight": rs.uniform(50, 90, size=N).round(1),
            "region": np.array(["north", "south", "east", "west"] * (N // 4)),
            "units": [f"U{k % 40}" for k in range(N)],
            "age": rs.randint(18, 80, size=N),
            "flag": rs.choice(["a", "b", "c"], size=N),
            "income": rs.uniform(30000, 70000, size=N).round(0),
            "visit": [f"2024-03-{1 + k % 28:02d}" for k in range(N)],
        }
    )


def planted_frame() -> pd.DataFrame:
    """The clean frame with one defect per analyzer. Each one is listed below."""
    frame = clean_frame()
    rs = np.random.RandomState(8)
    frame.loc[3, "weight"] = 900.0  # outliers: one extreme value
    frame["units"] = [str(10 + k % 90) for k in range(N)]  # dtypes: numbers stored as text,
    frame.loc[[5, 17, 40], "units"] = "n/a"  # with a stray token
    frame.loc[[8, 9, 10], "age"] = -4  # impossible values: negative age
    frame.loc[rs.choice(N, 70, replace=False), "income"] = np.nan  # missingness: 35%
    frame["flag"] = "same"  # constants: one value everywhere
    frame["visit"] = "2024-03-01"
    frame.loc[[12, 13], "visit"] = "2999-01-01"  # impossible values: dates in the future
    return pd.concat([frame, frame.iloc[:12]], ignore_index=True)  # duplicates: 12 copied rows


PLANTED = {
    ("missingness", ("income",)),
    ("missingness", ()),  # the same 35% seen as incomplete rows
    ("duplicates", ()),
    ("constants", ("flag",)),
    ("constants", ("visit",)),  # 2 of 212 dates differ: near-constant
    ("outliers", ("weight",)),
    ("dtypes", ("units",)),
    ("impossible_values", ("age",)),
    ("impossible_values", ("visit",)),
}


def run(frame, config=CONFIG, target=None, as_of=AS_OF):
    return run_quality_checks(Dataset(data=frame, name="t", target=target), config, as_of=as_of)


class TestPlantedDefects:
    def test_every_planted_defect_is_found_and_nothing_else(self):
        result = run(planted_frame())

        assert {(f.category, f.affected_columns) for f in result.findings} == PLANTED

    def test_a_frame_without_defects_has_no_findings_but_still_lists_the_checks_it_ran(self):
        result = run(clean_frame())

        assert result.findings == ()
        assert result.metrics["checks_run"] == CHECKS
        assert result.metrics["findings_by_severity"] == dict.fromkeys(
            ["critical", "high", "medium", "low", "info"], 0
        )


class TestOrder:
    def test_findings_run_from_most_to_least_severe_and_ties_keep_the_analyzer_order(self):
        result = run(planted_frame(), target="income")

        assert [(f.severity.value, f.category) for f in result.findings] == [
            ("high", "missingness"),  # a target with missing values is at least high
            ("medium", "missingness"),
            ("medium", "duplicates"),
            ("medium", "outliers"),
            ("medium", "dtypes"),
            ("medium", "impossible_values"),
            ("medium", "impossible_values"),
            ("low", "constants"),
            ("info", "constants"),
        ]


class TestMetrics:
    def test_severity_counts_match_the_findings(self):
        result = run(planted_frame(), target="income")

        by_severity = result.metrics["findings_by_severity"]
        assert list(by_severity) == ["critical", "high", "medium", "low", "info"]
        assert by_severity == {"critical": 0, "high": 1, "medium": 6, "low": 1, "info": 1}
        assert sum(by_severity.values()) == len(result.findings)

    def test_each_analyzer_keeps_its_own_metrics_under_its_own_name(self):
        dataset = Dataset(data=planted_frame(), name="t")
        result = run_quality_checks(dataset, CONFIG, as_of=AS_OF)

        alone = {
            "missingness": check_missingness(dataset, CONFIG),
            "duplicates": check_duplicates(dataset, CONFIG),
            "constants": check_constants(dataset, CONFIG),
            "outliers": check_outliers(dataset, CONFIG),
            "dtypes": check_dtypes(dataset, CONFIG),
            "impossible_values": check_impossible_values(dataset, CONFIG, as_of=AS_OF),
        }
        assert {name: result.metrics[name] for name in CHECKS} == {
            name: analysis.metrics for name, analysis in alone.items()
        }
        assert sorted(f.title for f in result.findings) == sorted(
            f.title for analysis in alone.values() for f in analysis.findings
        )


class TestAsOf:
    def test_the_reference_date_reaches_the_future_date_check(self):
        frame = planted_frame()

        early = run(frame, as_of=datetime.date(2000, 1, 1))
        late = run(frame, as_of=datetime.date(3000, 1, 1))

        assert early.metrics["impossible_values"]["as_of"] == "2000-01-01"
        assert late.metrics["impossible_values"]["as_of"] == "3000-01-01"
        assert any(f.title == "Dates in the future in visit" for f in early.findings)
        assert not any(f.title == "Dates in the future in visit" for f in late.findings)


class TestGuardrails:
    def test_a_sampled_run_says_so_in_its_findings(self):
        result = run(planted_frame(), config=AnalysisConfig(row_threshold=100))

        guardrail = [f for f in result.findings if f.category == "guardrail"]
        assert [(f.severity, f.title) for f in guardrail] == [
            (Severity.INFO, "Analysis ran on a sample of the rows")
        ]


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=5)

        first = run(planted_frame(), config=config)

        assert first.config == config
        assert run(planted_frame(), config=config).to_json() == first.to_json()

    def test_the_result_survives_json(self):
        result = run(planted_frame())

        assert AnalysisResult.from_json(result.to_json()) == result

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(
            DatasetError, match="cannot check the quality of a dataset with no rows"
        ):
            run(pd.DataFrame({"a": []}))
