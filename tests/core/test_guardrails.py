import pandas as pd
import pytest

from datadoctor import AnalysisConfig, Dataset, Provenance, Severity
from datadoctor.core.guardrails import apply_guardrails


def make_dataset(n_rows, n_columns=1, index=None, config=None):
    """A dataset whose first column holds each row's position, so selections can be read back."""
    frame = pd.DataFrame({f"c{i}": range(n_rows) for i in range(n_columns)}, index=index)
    provenance = None if config is None else Provenance.capture(frame, config=config)
    return Dataset(data=frame, name="t", provenance=provenance)


def positions(result):
    return result.frame["c0"].tolist()


class TestRows:
    def test_at_the_threshold_nothing_is_sampled(self):
        dataset = make_dataset(10)

        result = apply_guardrails(dataset, AnalysisConfig(row_threshold=10))

        assert result.frame is dataset.data
        assert result.findings == ()

    def test_one_row_above_the_threshold_is_sampled_down_to_the_threshold(self):
        result = apply_guardrails(make_dataset(11), AnalysisConfig(row_threshold=10))

        assert len(result.frame) == 10
        assert len(result.findings) == 1

    def test_the_same_seed_selects_the_same_rows_in_every_environment(self):
        # Pinned from numpy's frozen RandomState stream. The CI matrix runs two numpy and two
        # pandas versions, so this also fails if selection ever depends on either of them.
        config = AnalysisConfig(random_seed=42, row_threshold=10)

        result = apply_guardrails(make_dataset(1000), config)

        assert positions(result) == [136, 411, 513, 521, 626, 660, 678, 737, 740, 859]

    def test_a_different_seed_selects_different_rows(self):
        first = apply_guardrails(make_dataset(100), AnalysisConfig(random_seed=1, row_threshold=10))
        second = apply_guardrails(
            make_dataset(100), AnalysisConfig(random_seed=2, row_threshold=10)
        )

        assert positions(first) != positions(second)

    def test_the_sample_keeps_original_order_without_repeats(self):
        result = apply_guardrails(make_dataset(100), AnalysisConfig(row_threshold=10))

        chosen = positions(result)
        assert chosen == sorted(set(chosen))
        assert all(0 <= position < 100 for position in chosen)

    def test_the_dataset_frame_is_left_unchanged(self):
        dataset = make_dataset(100)
        snapshot = dataset.data.copy(deep=True)

        result = apply_guardrails(dataset, AnalysisConfig(row_threshold=10))

        assert result.frame is not dataset.data
        pd.testing.assert_frame_equal(dataset.data, snapshot)

    def test_selection_is_by_position_even_when_index_labels_repeat(self):
        labels = ["a", "b"] * 10
        dataset = make_dataset(20, index=labels)

        result = apply_guardrails(dataset, AnalysisConfig(row_threshold=5))

        assert len(result.frame) == 5
        assert result.frame.index.tolist() == [labels[p] for p in positions(result)]

    def test_the_config_argument_decides_not_the_datasets_own_config(self):
        loaded_under = AnalysisConfig(row_threshold=1000)
        dataset = make_dataset(100, config=loaded_under)

        result = apply_guardrails(dataset, AnalysisConfig(row_threshold=10))

        assert len(result.frame) == 10


class TestColumns:
    def test_at_the_threshold_pairwise_work_is_allowed(self):
        result = apply_guardrails(make_dataset(5, n_columns=3), AnalysisConfig(column_threshold=3))

        assert result.pairwise is True
        assert result.findings == ()

    def test_one_column_above_the_threshold_skips_pairwise_work(self):
        result = apply_guardrails(make_dataset(5, n_columns=4), AnalysisConfig(column_threshold=3))

        assert result.pairwise is False
        assert len(result.findings) == 1

    def test_skipping_pairwise_work_does_not_touch_the_rows(self):
        dataset = make_dataset(5, n_columns=4)

        result = apply_guardrails(dataset, AnalysisConfig(column_threshold=3))

        assert result.frame is dataset.data


class TestFindings:
    def test_sampling_is_reported_with_the_numbers_that_caused_it(self):
        result = apply_guardrails(
            make_dataset(11), AnalysisConfig(random_seed=42, row_threshold=10)
        )

        (finding,) = result.findings

        assert finding.severity is Severity.INFO
        assert finding.category == "guardrail"
        assert finding.confidence == 1.0
        assert "11 rows" in finding.evidence
        assert "row threshold of 10" in finding.evidence
        assert "sample of 10 rows" in finding.evidence
        assert "seed 42" in finding.evidence
        assert "class balance" in finding.limitations

    def test_skipped_pairwise_work_is_reported_with_the_numbers_that_caused_it(self):
        result = apply_guardrails(make_dataset(5, n_columns=4), AnalysisConfig(column_threshold=3))

        (finding,) = result.findings

        assert finding.severity is Severity.INFO
        assert finding.category == "guardrail"
        assert "4 columns" in finding.evidence
        assert "column threshold of 3" in finding.evidence

    def test_both_limits_give_two_findings_rows_first(self):
        config = AnalysisConfig(row_threshold=10, column_threshold=3)

        result = apply_guardrails(make_dataset(11, n_columns=4), config)

        assert [f.title for f in result.findings] == [
            "Analysis ran on a sample of the rows",
            "Pairwise analyses were skipped",
        ]

    def test_config_must_be_an_analysis_config(self):
        with pytest.raises(TypeError, match="AnalysisConfig"):
            apply_guardrails(make_dataset(5), {"row_threshold": 5})
