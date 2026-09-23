import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity
from datadoctor.core.exceptions import DatasetError
from datadoctor.eda import check_relationships, check_univariate, run_eda

N = 200


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(output_dir=tmp_path)


def frame() -> pd.DataFrame:
    """A column each analyzer has something to say about.

    ``ident`` is not drawn by univariate and not compared by relationships (an INFO finding from
    each). ``churned`` is a 0/1-coded target imbalanced well past the MEDIUM cutoff (a MEDIUM
    finding from relationships), and, having only 2 distinct values, is itself drawn by
    univariate as a discrete numeric column.
    """
    return pd.DataFrame(
        {
            "ident": [f"ID-{k:04d}" for k in range(N)],
            "churned": np.array([0] * (N - 4) + [1] * 4, dtype=float),
        }
    )


def run(data, config, target="churned") -> AnalysisResult:
    return run_eda(Dataset(data=data, name="t", target=target), config)


class TestMerge:
    def test_both_analyzers_findings_are_present_most_severe_first(self, config):
        result = run(frame(), config)

        assert [f.category for f in result.findings] == ["eda", "eda", "eda"]
        assert [f.severity for f in result.findings] == [
            Severity.MEDIUM,
            Severity.INFO,
            Severity.INFO,
        ]
        assert result.findings[0].title == "The target classes are imbalanced"
        # ties keep the analyzer order given to run_eda: univariate before relationships.
        assert result.findings[1].title == "Some columns were not plotted"
        assert result.findings[2].title == "Some columns were not compared"

    def test_checks_run_and_severity_counts(self, config):
        result = run(frame(), config)

        assert result.metrics["checks_run"] == ["univariate", "relationships"]
        by_severity = result.metrics["findings_by_severity"]
        assert list(by_severity) == ["critical", "high", "medium", "low", "info"]
        assert by_severity == {"critical": 0, "high": 0, "medium": 1, "low": 0, "info": 2}
        assert sum(by_severity.values()) == len(result.findings)

    def test_each_analyzer_keeps_its_own_metrics_under_its_own_name(self, config):
        dataset = Dataset(data=frame(), name="t", target="churned")

        result = run_eda(dataset, config)

        assert result.metrics["univariate"] == check_univariate(dataset, config).metrics
        assert result.metrics["relationships"] == check_relationships(dataset, config).metrics

    def test_artifacts_from_both_analyzers_are_merged_without_collision(self, config):
        result = run(frame(), config)

        assert "target_distribution" in result.artifacts  # from relationships
        assert any(key.startswith("univariate_") for key in result.artifacts)  # from univariate
        for path in result.artifacts.values():
            assert path.exists()

    def test_a_dataset_without_rows_is_rejected(self, config):
        with pytest.raises(DatasetError, match="cannot explore a dataset with no rows"):
            run(pd.DataFrame({"a": []}), config, target=None)
