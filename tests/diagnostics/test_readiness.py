from unittest.mock import patch

import pandas as pd
import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import Severity
from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.leakage_findings import DUPLICATE_TITLE
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.splits import check_split_strategy
from datadoctor.quality.constants import CONSTANT_TARGET_TITLE
from datadoctor.quality.missingness_findings import TARGET_MISSING_TITLE
from datadoctor.testing.synthetic import SyntheticConfig, make_synthetic_dataset

CONFIG = AnalysisConfig()
SC = SyntheticConfig(n_rows=2000, n_features=3, random_seed=42)


def entries(result, bucket):
    return {(e["source"], e["title"]) for e in result.metrics[bucket]}


class TestValidation:
    def test_zero_rows_is_rejected(self):
        dataset = Dataset(data=pd.DataFrame({"a": []}), name="empty")

        with pytest.raises(DatasetError, match="no rows"):
            check_readiness(dataset, CONFIG)


class TestCleanDataset:
    def test_nothing_blocking_and_nothing_attention(self):
        result = check_readiness(make_synthetic_dataset(SC).dataset, CONFIG)

        assert result.metrics["target_assessed"] is True
        assert result.metrics["blocking"] == []
        assert result.metrics["attention"] == []
        (headline,) = result.findings
        assert headline.title == "No blocking issues were found by these checks"
        assert headline.severity == Severity.INFO


class TestNoTarget:
    def test_reports_not_assessed_not_no_blocking_issues(self):
        frame = make_synthetic_dataset(SC).dataset.data
        dataset = Dataset(data=frame, name="no_target")

        result = check_readiness(dataset, CONFIG)

        assert result.metrics["target_assessed"] is False
        assert result.metrics["blocking"] == []
        assert result.metrics["attention"] == []
        (headline,) = result.findings
        assert headline.title == "Readiness was not assessed"
        assert "not assessed" in headline.title.lower()
        assert headline.title != "No blocking issues were found by these checks"


class TestConstantTargetAndDuplicateLeakage:
    def test_both_are_blocking(self):
        # A literally constant target makes correlation with it undefined, so leak_duplicate
        # cannot be detected on the *same* dataset a constant target requires: check_leakage's
        # own association machinery has nothing to compute against a zero-variance target. This
        # combines the two conditions the way check_readiness actually supports it: a genuine
        # duplicate-leakage result computed on a normal dataset, passed through as the optional
        # `leakage` argument, alongside a dataset whose target really is constant.
        duplicate_leakage_dataset = make_synthetic_dataset(
            SC, leakage="duplicate", task="regression"
        ).dataset
        leakage_result = check_leakage(duplicate_leakage_dataset, CONFIG)

        frame = pd.DataFrame({"feature": range(200), "target": [1.0] * 200})
        dataset = Dataset(data=frame, name="constant_target", target="target")

        result = check_readiness(dataset, CONFIG, leakage=leakage_result)

        assert entries(result, "blocking") == {
            ("quality", CONSTANT_TARGET_TITLE),
            ("leakage", DUPLICATE_TITLE),
        }
        assert result.metrics["attention"] == []


class TestHeadlineWording:
    def test_two_blockers_pluralize_the_title(self):
        dataset = make_synthetic_dataset(
            SC,
            temporal=True,
            leakage="duplicate",
            task="regression",
            effect_sizes={"temporal": 0.9},
        ).dataset

        result = check_readiness(dataset, CONFIG)

        (headline,) = [f for f in result.findings if f.title.endswith("blocking issues found")]
        assert headline.title == "2 blocking issues found"
        assert "A random split is likely to leak future information" in headline.evidence
        assert "A random split would leak future information" not in headline.evidence
        assert headline.interpretation.startswith("Any blocking issue would make")

    def test_attention_count_is_singular_for_one_item(self):
        frame = pd.DataFrame({"feature": range(200), "target": range(200)}).astype(float)
        frame.loc[0, "target"] = None
        dataset = Dataset(data=frame, name="one_attention", target="target")

        result = check_readiness(dataset, CONFIG)

        (headline,) = result.findings
        assert headline.evidence == (
            "No blocking issues were found by these checks. "
            "1 item needs attention, listed in the readiness metrics."
        )

    def test_attention_count_is_plural_for_more_than_one_item(self):
        # Small-n (splits) and non-critical target missingness (quality) fire together on a
        # small dataset with a couple of missing target values, giving two attention items from
        # two different sources without needing the leakage/splits pass-through parameters.
        frame = pd.DataFrame({"feature": range(50), "target": range(50)}).astype(float)
        frame.loc[[0, 1], "target"] = None
        dataset = Dataset(data=frame, name="two_attention", target="target")

        result = check_readiness(dataset, CONFIG)

        assert len(result.metrics["attention"]) == 2
        (headline,) = result.findings
        assert headline.evidence == (
            "No blocking issues were found by these checks. "
            "2 items need attention, listed in the readiness metrics."
        )

    def test_no_attention_sentence_when_attention_is_empty(self):
        result = check_readiness(make_synthetic_dataset(SC).dataset, CONFIG)

        (headline,) = result.findings
        assert headline.evidence == "No blocking issues were found by these checks."


class TestTargetMissingness:
    def test_exactly_one_missing_value_is_attention(self):
        # The floor of "any missing at all", distinct from the two-missing-values case below:
        # an off-by-one at this specific boundary (count > 1 instead of count > 0) would still
        # pass a test using two missing values.
        frame = pd.DataFrame({"feature": range(200), "target": range(200)}).astype(float)
        frame.loc[0, "target"] = None
        dataset = Dataset(data=frame, name="one_missing", target="target")

        result = check_readiness(dataset, CONFIG)

        assert entries(result, "attention") == {("quality", TARGET_MISSING_TITLE)}

    def test_two_missing_values_is_attention(self):
        frame = make_synthetic_dataset(SC, task="regression").dataset.data.copy()
        frame.loc[[0, 1], "target"] = None
        dataset = Dataset(data=frame, name="two_missing", target="target")

        result = check_readiness(dataset, CONFIG)

        assert entries(result, "attention") == {("quality", TARGET_MISSING_TITLE)}
        assert ("quality", TARGET_MISSING_TITLE) not in entries(result, "blocking")

    def test_eighty_percent_missing_is_blocking(self):
        frame = pd.DataFrame({"feature": range(200), "target": range(200)}).astype(float)
        frame.loc[: int(0.80 * 200) - 1, "target"] = None
        dataset = Dataset(data=frame, name="mostly_missing", target="target")

        result = check_readiness(dataset, CONFIG)

        assert entries(result, "blocking") == {("quality", TARGET_MISSING_TITLE)}
        assert ("quality", TARGET_MISSING_TITLE) not in entries(result, "attention")


class TestPassThrough:
    def test_splits_and_leakage_are_not_recomputed_when_supplied(self):
        dataset = make_synthetic_dataset(SC).dataset
        splits_result = check_split_strategy(dataset, CONFIG)
        leakage_result = check_leakage(dataset, CONFIG)

        with (
            patch("datadoctor.diagnostics.readiness.check_split_strategy") as mock_splits,
            patch("datadoctor.diagnostics.readiness.check_leakage") as mock_leakage,
        ):
            check_readiness(dataset, CONFIG, splits=splits_result, leakage=leakage_result)

            mock_splits.assert_not_called()
            mock_leakage.assert_not_called()

    def test_splits_and_leakage_are_computed_once_each_when_not_supplied(self):
        dataset = make_synthetic_dataset(SC).dataset

        with (
            patch(
                "datadoctor.diagnostics.readiness.check_split_strategy",
                wraps=check_split_strategy,
            ) as mock_splits,
            patch(
                "datadoctor.diagnostics.readiness.check_leakage", wraps=check_leakage
            ) as mock_leakage,
        ):
            check_readiness(dataset, CONFIG)

            mock_splits.assert_called_once()
            mock_leakage.assert_called_once()
