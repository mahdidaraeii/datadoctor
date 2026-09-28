import numpy as np
import pandas as pd
import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import Severity
from datadoctor.diagnostics.leakage import NEAR_PERFECT_THRESHOLD, check_leakage
from datadoctor.testing.synthetic import SyntheticConfig, make_synthetic_dataset

CONFIG = AnalysisConfig()
SC = SyntheticConfig(n_rows=2000, n_features=3, random_seed=42)

NEAR_PERFECT_TITLE = "A feature is a near-perfect single-column predictor"
DUPLICATE_TITLE = "A feature appears to be the target under another name or encoding"
NAMING_TITLE = "A column name suggests a post-outcome event"


def titles(result):
    return [f.title for f in result.findings]


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


def eta_squared_independent(feature: pd.Series, labels: pd.Series) -> float:
    """A ground-truth correlation ratio, computed independently of the module under test."""
    feature = feature.astype(float)
    overall_mean = feature.mean()
    total_ss = ((feature - overall_mean) ** 2).sum()
    group_means = feature.groupby(labels, observed=True).transform("mean")
    between_ss = ((group_means - overall_mean) ** 2).sum()
    return float((between_ss / total_ss) ** 0.5)


class TestValidation:
    def test_zero_rows_is_rejected(self):
        dataset = Dataset(data=pd.DataFrame({"a": []}), name="empty")

        with pytest.raises(DatasetError, match="no rows"):
            check_leakage(dataset, CONFIG)

    def test_config_must_be_an_analysis_config(self):
        dataset = make_synthetic_dataset(SC).dataset

        with pytest.raises(TypeError, match="AnalysisConfig"):
            check_leakage(dataset, {"row_threshold": 5})


class TestCleanDataset:
    def test_no_findings_on_a_dataset_with_no_planted_leakage(self):
        result = check_leakage(make_synthetic_dataset(SC).dataset, CONFIG)

        assert result.findings == ()

    def test_no_findings_for_a_regression_target_either(self):
        result = check_leakage(make_synthetic_dataset(SC, task="regression").dataset, CONFIG)

        assert result.findings == ()


class TestNoTarget:
    def test_every_check_is_skipped_without_a_target(self):
        frame = pd.DataFrame(
            {
                "leak_score": [0.0] * 100 + [1.0] * 100,
                "settlement_amount": range(200),
                "value": [0.0] * 100 + [1.0] * 100,
            }
        )
        dataset = Dataset(data=frame, name="no_target")

        result = check_leakage(dataset, CONFIG)

        assert result.findings == ()
        assert result.metrics["near_perfect"] is None
        assert result.metrics["duplicated"] is None
        assert result.metrics["suspicious_names"] is None


class TestNearPerfectPredictor:
    def test_regression_leak_future_fires_at_the_exact_requested_strength(self):
        dataset = make_synthetic_dataset(
            SC, leakage="future", task="regression", effect_sizes={"leakage": 0.95}
        ).dataset

        result = check_leakage(dataset, CONFIG)

        found = finding(result, NEAR_PERFECT_TITLE)
        assert found.severity == Severity.HIGH
        assert found.confidence == 0.6
        assert found.affected_columns == ("leak_future",)
        assert result.metrics["near_perfect"]["effects"]["leak_future"] == pytest.approx(
            0.95, abs=1e-9
        )

    def test_a_merely_strongly_predictive_regression_feature_does_not_fire(self):
        # Constructed deliberately, below NEAR_PERFECT_THRESHOLD: a real predictor this strong
        # is plausible (comparable to breast_cancer.csv's own strongest features), and the check
        # must not treat "very predictive" alone as leakage.
        dataset = make_synthetic_dataset(
            SC, leakage="future", task="regression", effect_sizes={"leakage": 0.85}
        ).dataset

        result = check_leakage(dataset, CONFIG)

        assert titles(result) == []
        assert result.metrics["near_perfect"]["effects"]["leak_future"] == pytest.approx(
            0.85, abs=1e-9
        )

    def test_classification_leak_future_does_not_fire_even_at_high_requested_strength(self):
        # The deliberate "strongly predictive, not leaking" control for classification: even a
        # near-deterministic continuous function of the latent (0.95, then 0.98 below) cannot
        # clear NEAR_PERFECT_THRESHOLD against a classification target, because the correlation
        # ratio between a continuous, overlapping-distribution column and a binary target has a
        # reachable ceiling around 0.8 -- well below the threshold -- regardless of how strong
        # the underlying relationship is. This is not a false negative to fix; it is the
        # documented scope of this check (see leakage.py's module notes).
        dataset = make_synthetic_dataset(
            SC, leakage="future", task="classification", effect_sizes={"leakage": 0.95}
        ).dataset
        frame = dataset.data

        observed = eta_squared_independent(frame["leak_future"], frame["target"])

        assert observed == pytest.approx(0.78, abs=0.05)
        assert observed < NEAR_PERFECT_THRESHOLD
        result = check_leakage(dataset, CONFIG)
        assert titles(result) == []

    def test_classification_leak_future_stays_silent_at_an_even_higher_strength(self):
        dataset = make_synthetic_dataset(
            SC, leakage="future", task="classification", effect_sizes={"leakage": 0.98}
        ).dataset
        frame = dataset.data

        observed = eta_squared_independent(frame["leak_future"], frame["target"])

        assert observed == pytest.approx(0.78, abs=0.05)
        assert observed < NEAR_PERFECT_THRESHOLD
        result = check_leakage(dataset, CONFIG)
        assert titles(result) == []


class TestSignificanceGate:
    def test_bonferroni_correction_can_suppress_an_effect_above_the_threshold(self):
        # A deliberately small, exact construction: 5 categories of 2 rows each (n_rows=10,
        # average group size exactly at MIN_AVERAGE_GROUP_SIZE) gives leak_candidate a sample
        # eta-squared of exactly 0.90 against the target, an effect of sqrt(0.90) = 0.9487,
        # comfortably above NEAR_PERFECT_THRESHOLD. Its raw p-value (0.0103) alone clears ALPHA,
        # but four more categorical candidates bring tests_run to 5, and the Bonferroni-adjusted
        # p (0.0103 * 5 = 0.0514) does not. This is a real case, not a hypothetical: few rows
        # and several categorical candidates is an ordinary shape for a small dataset.
        rng = np.random.RandomState(0)
        n_groups, group_size = 5, 2
        n_rows = n_groups * group_size
        codes = np.repeat(np.arange(n_groups), group_size)

        group_effect = rng.normal(size=n_groups)
        signal = group_effect[codes]
        signal = (signal - signal.mean()) / signal.std()
        raw = rng.normal(size=n_rows)
        group_means = np.array([raw[codes == g].mean() for g in range(n_groups)])
        residual = raw - group_means[codes]
        residual = (residual - residual.mean()) / residual.std()
        target = np.sqrt(0.90) * signal + np.sqrt(0.10) * residual

        frame = {"leak_candidate": [f"g{c}" for c in codes], "target": target}
        for i in range(4):
            noise_codes = rng.randint(0, n_groups, size=n_rows)
            frame[f"noise_{i}"] = [f"n{c}" for c in noise_codes]
        dataset = Dataset(data=pd.DataFrame(frame), name="bonferroni_leakage", target="target")

        result = check_leakage(dataset, CONFIG)

        assert result.metrics["near_perfect"]["effects"]["leak_candidate"] == pytest.approx(
            0.9486832980505138, abs=1e-9
        )
        assert result.metrics["near_perfect"]["flagged"] == []
        assert titles(result) == []


class TestDuplicatedTarget:
    def test_regression_leak_duplicate_fires_at_exactly_one(self):
        dataset = make_synthetic_dataset(SC, leakage="duplicate", task="regression").dataset

        result = check_leakage(dataset, CONFIG)

        found = finding(result, DUPLICATE_TITLE)
        assert found.severity == Severity.CRITICAL
        assert found.confidence == 0.9
        assert found.affected_columns == ("leak_duplicate",)
        assert result.metrics["duplicated"]["effects"]["leak_duplicate"] == pytest.approx(
            1.0, abs=1e-9
        )

    def test_classification_leak_duplicate_fires_at_exactly_one(self):
        dataset = make_synthetic_dataset(SC, leakage="duplicate", task="classification").dataset

        result = check_leakage(dataset, CONFIG)

        found = finding(result, DUPLICATE_TITLE)
        assert found.affected_columns == ("leak_duplicate",)
        assert result.metrics["duplicated"]["effects"]["leak_duplicate"] == pytest.approx(
            1.0, abs=1e-9
        )

    def test_the_ceiling_is_not_general_a_step_function_reaches_one(self):
        # Proves the classification ceiling above is specific to an overlapping distribution,
        # not a property of the correlation ratio itself: a numeric column with one distinct
        # value per class (zero within-class variance) has no such ceiling.
        n_per_class = 200
        classes = ["alpha", "beta", "gamma"]
        target, leak_score = [], []
        for code, label in enumerate(classes):
            target += [label] * n_per_class
            leak_score += [float(code)] * n_per_class
        frame = pd.DataFrame({"leak_score": leak_score, "target": pd.Categorical(target)})
        dataset = Dataset(data=frame, name="step_function", target="target")

        result = check_leakage(dataset, CONFIG)

        found = finding(result, DUPLICATE_TITLE)
        assert found.affected_columns == ("leak_score",)
        assert result.metrics["duplicated"]["effects"]["leak_score"] == pytest.approx(1.0, abs=1e-9)


class TestTierExclusivity:
    def test_a_duplicate_column_does_not_also_appear_as_near_perfect(self):
        dataset = make_synthetic_dataset(SC, leakage="duplicate", task="regression").dataset

        result = check_leakage(dataset, CONFIG)

        assert "leak_duplicate" not in result.metrics["near_perfect"]["flagged"]
        assert NEAR_PERFECT_TITLE not in titles(result)
        assert DUPLICATE_TITLE in titles(result)


class TestNearUniqueExclusion:
    def test_a_near_unique_column_with_incidental_duplicates_is_not_a_candidate(self):
        # The same construction that surfaced a real false-positive risk while building S26:
        # 15 singleton categories plus 5 categories repeated once, the repeated pair given
        # nearly identical target values. Cardinality is kept at exactly 20 (n_rows=25, average
        # group size 1.25) so profile_schema still types the column categorical via its own
        # cardinality<=20 rule, rather than "text": a higher cardinality here would make the
        # fixture accidentally test schema typing instead of this analyzer's own guard. Without
        # excluding sparse categorical candidates, this produces a spuriously high correlation
        # ratio (0.9999) with no general relationship between the column and the target --
        # exactly the kind of "suspiciously high" number this analyzer is looking for.
        rng = np.random.RandomState(0)
        ids = [f"id_{i}" for i in range(15)]
        target = list(rng.normal(size=15))
        for i in range(5):
            value = rng.normal()
            ids += [f"pair_{i}", f"pair_{i}"]
            target += [value, value + rng.normal() * 0.01]
        frame = pd.DataFrame({"candidate": ids, "target": target})
        dataset = Dataset(data=frame, name="near_unique", target="target")

        result = check_leakage(dataset, CONFIG)

        assert "candidate" not in result.metrics["near_perfect"]["candidates"]
        assert result.findings == ()

    def test_a_high_cardinality_column_is_not_a_candidate_even_when_well_populated(self):
        # 60 categories, 4 rows each: average group size (4) comfortably clears
        # MIN_AVERAGE_GROUP_SIZE, so this is testing the separate MAX_CATEGORIES exclusion, not
        # the sparse-data one above.
        n_categories, per_category = 60, 4
        candidate, target = [], []
        for i in range(n_categories):
            candidate += [f"cat_{i}"] * per_category
            target += [float(i)] * per_category
        frame = pd.DataFrame({"candidate": candidate, "target": target})
        dataset = Dataset(data=frame, name="high_cardinality", target="target")

        result = check_leakage(dataset, CONFIG)

        assert "candidate" not in result.metrics["near_perfect"]["candidates"]


class TestSuspiciousNaming:
    def test_a_post_outcome_name_fires_regardless_of_association(self):
        rng = np.random.RandomState(0)
        frame = pd.DataFrame(
            {
                "discharge_status": rng.choice(["a", "b"], size=200),
                "target": rng.choice([0, 1], size=200),
            }
        )
        dataset = Dataset(data=frame, name="suspicious_name", target="target")

        result = check_leakage(dataset, CONFIG)

        found = finding(result, NAMING_TITLE)
        assert found.severity == Severity.MEDIUM
        assert found.confidence == 0.3
        assert found.affected_columns == ("discharge_status",)
        assert NEAR_PERFECT_TITLE not in titles(result)
        assert DUPLICATE_TITLE not in titles(result)

    def test_camelcase_names_are_tokenized(self):
        frame = pd.DataFrame({"settlementDate": range(200), "target": np.arange(200) % 2})
        dataset = Dataset(data=frame, name="camel_case", target="target")

        result = check_leakage(dataset, CONFIG)

        assert result.metrics["suspicious_names"]["flagged"] == ["settlementDate"]

    def test_matching_is_whole_word_not_substring(self):
        # "predischarge" contains "discharge" as a substring but is a different word.
        frame = pd.DataFrame({"predischarge_value": range(200), "target": np.arange(200) % 2})
        dataset = Dataset(data=frame, name="substring_check", target="target")

        result = check_leakage(dataset, CONFIG)

        assert result.metrics["suspicious_names"]["flagged"] == []

    def test_an_ordinary_name_does_not_fire(self):
        result = check_leakage(make_synthetic_dataset(SC).dataset, CONFIG)

        assert NAMING_TITLE not in titles(result)
        assert result.metrics["suspicious_names"]["flagged"] == []
