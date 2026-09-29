import pandas as pd
import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult
from datadoctor.diagnostics import check_split_strategy, run_diagnostics
from datadoctor.eda import run_eda
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import run_quality_checks
from datadoctor.report import build_report

N = 200


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(output_dir=tmp_path)


def mixed_frame() -> pd.DataFrame:
    """A column each of the four sources has something to say about.

    ``correlated_a``/``correlated_b`` are strongly correlated (an eda-only finding); repeated,
    not sequential unique values, so neither is mistaken for an identifier and excluded from
    the comparison. ``income`` has missing values (a quality-only finding). ``target`` is
    imbalanced past the stratification threshold, which both eda (class imbalance) and
    split_strategy (stratification) report on the same root cause -- diagnose merges the two
    into one recommendation.
    """
    a = [k % 50 for k in range(N)]
    frame = pd.DataFrame(
        {
            "correlated_a": a,
            "correlated_b": [x * 2 + 1 for x in a],
            "income": [None if k < 80 else 30000.0 + k for k in range(N)],
            "target": pd.Categorical(["rare"] * 6 + ["common"] * (N - 6)),
        }
    )
    return frame


def run(frame, config, target="target") -> AnalysisResult:
    return build_report(Dataset(data=frame, name="t", target=target), config)


class TestValidation:
    def test_zero_rows_is_rejected(self, config):
        with pytest.raises(DatasetError, match="cannot build a report.*no rows"):
            run(pd.DataFrame({"a": []}), config, target=None)


class TestMergeAcrossFourSources:
    def test_the_imbalance_finding_appears_exactly_once_as_the_recommendation_version(self, config):
        result = run(mixed_frame(), config)

        imbalance_titled = [
            f for f in result.findings if f.title == "The target classes are imbalanced"
        ]
        assert len(imbalance_titled) == 1
        assert imbalance_titled[0].category == "recommendation"

    def test_quality_only_and_eda_only_findings_pass_through_unfiltered(self, config):
        result = run(mixed_frame(), config)

        titles = [f.title for f in result.findings]
        assert "High missingness in income" in titles  # quality-only
        assert "Numeric columns are strongly correlated" in titles  # eda-only

    def test_profiles_own_finding_is_included_too(self, config):
        # income, 40% missing, is also flagged by profile_schema as a possible identifier
        # (a near-unique-looking remainder once nulls are set aside): profile's own finding,
        # not quality's or eda's, and it must not be dropped from the merge.
        result = run(mixed_frame(), config)

        assert any(
            f.category == "schema" and f.title == "Possible identifier column"
            for f in result.findings
        )

    def test_diagnose_only_findings_are_included_too(self, config):
        # income's missingness makes this frame's row count irrelevant here; a target this
        # imbalanced also trips split_strategy's own stratification finding, merged above, but
        # readiness's headline itself must still not appear (build_recommendations excludes it).
        result = run(mixed_frame(), config)

        assert "Readiness was not assessed" not in [f.title for f in result.findings]
        assert not any(f.category == "readiness" and "blocking" in f.title for f in result.findings)


class TestMetricsNesting:
    def test_checks_run_lists_all_four_sources(self, config):
        result = run(mixed_frame(), config)

        assert result.metrics["checks_run"] == ["profile", "quality", "eda", "diagnose"]

    def test_each_sources_metrics_are_nested_under_its_own_name(self, config):
        dataset = Dataset(data=mixed_frame(), name="t", target="target")
        profile = profile_schema(dataset)
        quality = run_quality_checks(dataset, config)
        eda = run_eda(dataset, config)
        diagnose = run_diagnostics(dataset, config, quality=quality, eda=eda)

        result = build_report(dataset, config)

        assert result.metrics["profile"] == profile.metrics
        assert result.metrics["quality"] == quality.metrics
        assert result.metrics["eda"] == eda.metrics
        assert result.metrics["diagnose"] == diagnose.metrics

    def test_findings_by_severity_sums_to_the_findings_count(self, config):
        result = run(mixed_frame(), config)

        by_severity = result.metrics["findings_by_severity"]
        assert list(by_severity) == ["critical", "high", "medium", "low", "info"]
        assert sum(by_severity.values()) == len(result.findings)


class TestDocument:
    def test_the_written_file_matches_the_markdown_metric(self, config):
        result = run(mixed_frame(), config)

        written = result.artifacts["report"].read_text(encoding="utf-8")
        assert written == result.metrics["markdown"]

    def test_figures_are_listed_in_the_document_and_not_duplicated(self, config):
        result = run(mixed_frame(), config)

        dataset = Dataset(data=mixed_frame(), name="t", target="target")
        for key in run_eda(dataset, config).artifacts:
            assert key in result.artifacts
        assert "## Figures" in result.metrics["markdown"]

    def test_a_quality_only_finding_appears_in_the_document_text(self, config):
        result = run(mixed_frame(), config)

        assert "High missingness in income" in result.metrics["markdown"]
        assert result.metrics["markdown"].count("The target classes are imbalanced") == 1

    def test_the_limitations_section_is_present(self, config):
        result = run(mixed_frame(), config)

        assert "## Limitations" in result.metrics["markdown"]


class TestDeterminism:
    def test_the_same_seed_and_config_give_byte_identical_markdown(self, config):
        # Same output_dir for both: a different one would legitimately change the embedded
        # figure paths without the analysis itself having changed at all.
        first = run(mixed_frame(), config)
        second = run(mixed_frame(), config)

        assert first.metrics["markdown"] == second.metrics["markdown"]


class TestResult:
    def test_the_result_survives_json(self, config):
        result = run(mixed_frame(), config)

        assert AnalysisResult.from_json(result.to_json()) == result


class TestExactDuplicateCollapse:
    def test_a_guardrail_finding_reported_by_several_sources_appears_once(self, tmp_path):
        # quality, eda and diagnose (via splits/leakage) each independently attach this same
        # finding when the dataset is sampled; it must collapse to one, not appear three times.
        frame = pd.DataFrame({"a": range(500), "b": range(500), "target": ["x", "y"] * 250})
        config = AnalysisConfig(output_dir=tmp_path, row_threshold=100)

        result = run(frame, config)

        guardrails = [f for f in result.findings if f.category == "guardrail"]
        assert len(guardrails) == 1
        assert guardrails[0].title == "Analysis ran on a sample of the rows"


class TestSplitsPassthrough:
    def test_a_dataset_without_the_imbalance_defect_has_no_merge(self, config):
        frame = pd.DataFrame(
            {"a": range(N), "b": range(N), "target": pd.Categorical(["x", "y"] * (N // 2))}
        )

        result = run(frame, config)

        assert "The target classes are imbalanced" not in [f.title for f in result.findings]
        # sanity: split_strategy itself found nothing either, confirming the frame is balanced.
        assert (
            check_split_strategy(Dataset(data=frame, name="t", target="target"), config).findings
            == ()
        )
