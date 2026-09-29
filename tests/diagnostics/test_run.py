import numpy as np
import pandas as pd
import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult
from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.run import run_diagnostics
from datadoctor.diagnostics.splits import check_split_strategy
from datadoctor.eda import run_eda
from datadoctor.quality import run_quality_checks
from datadoctor.testing.synthetic import SyntheticConfig, make_synthetic_dataset

CONFIG = AnalysisConfig()
N = 200


def mixed_frame() -> pd.DataFrame:
    """A column each domain has something to say about.

    ``correlated_a``/``correlated_b`` are two numeric columns with a near-perfect linear
    relationship (an eda-only finding: strongly correlated features). ``income`` has missing
    values (a quality-only finding). ``target`` is a classification target imbalanced past the
    stratification threshold, which both eda (class imbalance) and split_strategy
    (stratification) report on the same root cause.
    """
    base = np.arange(N, dtype=float)
    frame = pd.DataFrame(
        {
            "correlated_a": base,
            "correlated_b": base * 2 + 1,
            "income": [None if k < 80 else 30000.0 + k for k in range(N)],
            "target": pd.Categorical(["rare"] * 6 + ["common"] * (N - 6)),
        }
    )
    return frame


def run(frame, config=CONFIG, target="target") -> AnalysisResult:
    return run_diagnostics(Dataset(data=frame, name="t", target=target), config)


class TestValidation:
    def test_zero_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="cannot run diagnostics.*no rows"):
            run(pd.DataFrame({"a": []}), target=None)


class TestFilter:
    def test_a_quality_only_finding_is_not_in_diagnose_findings(self):
        result = run(mixed_frame())

        assert "Some missingness in income" not in [f.title for f in result.findings]

    def test_an_eda_only_finding_is_not_in_diagnose_findings(self):
        result = run(mixed_frame())

        assert "Numeric columns are strongly correlated" not in [f.title for f in result.findings]

    def test_the_imbalance_and_stratification_merge_is_kept_with_both_sources(self):
        result = run(mixed_frame())

        (rec,) = result.metrics["recommendations"]
        categories = {f["category"] for f in rec["findings"]}
        assert categories == {"eda", "split_strategy"}
        assert len(result.findings) == 1
        assert result.findings[0].category == "recommendation"


class TestLeakage:
    def test_planted_duplicate_leakage_is_reported(self):
        sc = SyntheticConfig(n_rows=2000, n_features=3, random_seed=42)
        dataset = make_synthetic_dataset(sc, leakage="duplicate").dataset

        result = run_diagnostics(dataset, CONFIG)

        assert any(
            f.title == "A feature appears to be the target under another name or encoding"
            for f in result.findings
        )


class TestReadinessReuse:
    def test_readiness_metrics_match_computing_it_directly_with_the_same_splits_and_leakage(self):
        dataset = Dataset(data=mixed_frame(), name="t", target="target")
        splits = check_split_strategy(dataset, CONFIG)
        leakage = check_leakage(dataset, CONFIG)
        standalone = check_readiness(dataset, CONFIG, splits=splits, leakage=leakage)

        result = run_diagnostics(dataset, CONFIG)

        assert result.metrics["readiness"] == standalone.metrics


class TestQualityEdaReuse:
    def test_a_passed_in_empty_eda_result_is_used_as_is_not_recomputed(self):
        # mixed_frame's target is imbalanced enough that a freshly computed eda result would
        # merge with split_strategy's own stratification finding. Passing an empty eda result
        # in means that merge cannot happen -- proving it was reused, not recomputed, since a
        # fresh computation would have found the real imbalance.
        dataset = Dataset(data=mixed_frame(), name="t", target="target")
        empty_eda = AnalysisResult(findings=())

        result = run_diagnostics(dataset, CONFIG, eda=empty_eda)

        titles = [f.title for f in result.findings]
        assert "The target classes are imbalanced" not in titles
        assert "Stratified splitting is recommended for this target" in titles

    def test_a_passed_in_quality_result_produces_the_same_result_as_computing_it_fresh(self):
        dataset = Dataset(data=mixed_frame(), name="t", target="target")
        quality = run_quality_checks(dataset, CONFIG)
        eda = run_eda(dataset, CONFIG)

        reused = run_diagnostics(dataset, CONFIG, quality=quality, eda=eda)
        fresh = run_diagnostics(dataset, CONFIG)

        assert reused.to_json() == fresh.to_json()


class TestMetrics:
    def test_checks_run_lists_all_five_analyzers(self):
        result = run(mixed_frame())

        assert result.metrics["checks_run"] == [
            "quality",
            "eda",
            "split_strategy",
            "leakage",
            "readiness",
        ]

    def test_severity_counts_match_the_findings(self):
        result = run(mixed_frame())

        by_severity = result.metrics["findings_by_severity"]
        assert list(by_severity) == ["critical", "high", "medium", "low", "info"]
        assert sum(by_severity.values()) == len(result.findings)

    def test_recommendations_metric_has_one_entry_per_finding(self):
        result = run(mixed_frame())

        assert len(result.metrics["recommendations"]) == len(result.findings)


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=5)

        first = run(mixed_frame(), config=config)

        assert first.config == config
        assert run(mixed_frame(), config=config).to_json() == first.to_json()

    def test_the_result_survives_json(self):
        result = run(mixed_frame())

        assert AnalysisResult.from_json(result.to_json()) == result

    def test_a_dataset_with_no_defects_has_no_findings_but_still_lists_the_checks_it_ran(self):
        frame = pd.DataFrame(
            {"a": range(N), "b": range(N), "target": pd.Categorical(["x", "y"] * (N // 2))}
        )

        result = run(frame)

        assert result.findings == ()
        assert result.metrics["checks_run"] == [
            "quality",
            "eda",
            "split_strategy",
            "leakage",
            "readiness",
        ]
