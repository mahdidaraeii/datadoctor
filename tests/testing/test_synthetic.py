import numpy as np
import pandas as pd
import pytest

from datadoctor.core.exceptions import ConfigError
from datadoctor.testing.synthetic import SyntheticConfig, make_synthetic_dataset

TOLERANCE = 1e-9


def _config(**overrides):
    defaults = {"n_rows": 2000, "n_features": 3, "random_seed": 42}
    defaults.update(overrides)
    return SyntheticConfig(**defaults)


def _eta_squared(feature: pd.Series, labels: pd.Series) -> float:
    feature = feature.astype(float)
    overall_mean = feature.mean()
    total_ss = ((feature - overall_mean) ** 2).sum()
    group_means = feature.groupby(labels, observed=True).transform("mean")
    between_ss = ((group_means - overall_mean) ** 2).sum()
    return between_ss / total_ss


def _mean_group_modal_share(frame: pd.DataFrame) -> float:
    counts = frame.groupby(["group_id", "target"], observed=True).size().unstack(fill_value=0)
    return (counts.max(axis=1) / counts.sum(axis=1)).mean()


class TestSyntheticConfig:
    @pytest.mark.parametrize("value", [9, 1.5, "10", None, True])
    def test_bad_n_rows_is_rejected(self, value):
        with pytest.raises(ConfigError, match="n_rows"):
            SyntheticConfig(n_rows=value, n_features=1, random_seed=0)

    @pytest.mark.parametrize("value", [0, 1.5, "1", None, True])
    def test_bad_n_features_is_rejected(self, value):
        with pytest.raises(ConfigError, match="n_features"):
            SyntheticConfig(n_rows=100, n_features=value, random_seed=0)

    @pytest.mark.parametrize("value", [-1, 1.5, "0", True])
    def test_bad_random_seed_is_rejected(self, value):
        with pytest.raises(ConfigError, match="random_seed"):
            SyntheticConfig(n_rows=100, n_features=1, random_seed=value)

    @pytest.mark.parametrize("value", [1, 1.5, "2", True])
    def test_bad_n_classes_is_rejected(self, value):
        with pytest.raises(ConfigError, match="n_classes"):
            SyntheticConfig(n_rows=100, n_features=1, random_seed=0, n_classes=value)


class TestValidation:
    def test_config_must_be_a_synthetic_config(self):
        with pytest.raises(TypeError, match="SyntheticConfig"):
            make_synthetic_dataset({"n_rows": 100})

    def test_unknown_task_is_rejected(self):
        with pytest.raises(ConfigError, match="task"):
            make_synthetic_dataset(_config(), task="ranking")

    def test_unknown_leakage_mechanism_is_rejected(self):
        with pytest.raises(ConfigError, match="leakage"):
            make_synthetic_dataset(_config(), leakage="past")

    @pytest.mark.parametrize("value", [-0.1, 1.0, 1.1])
    def test_label_noise_out_of_range_is_rejected(self, value):
        with pytest.raises(ConfigError, match="label_noise"):
            make_synthetic_dataset(_config(), label_noise=value)

    @pytest.mark.parametrize("value", [0, -1, 2000])
    def test_groups_out_of_range_is_rejected(self, value):
        with pytest.raises(ConfigError, match="groups"):
            make_synthetic_dataset(_config(), groups=value)

    def test_unknown_effect_sizes_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown effect_sizes keys"):
            make_synthetic_dataset(_config(), leakage="future", effect_sizes={"drift": 0.5})

    @pytest.mark.parametrize("value", [-1.1, 1.1])
    def test_leakage_effect_size_out_of_range_is_rejected(self, value):
        with pytest.raises(ConfigError, match="effect_sizes"):
            make_synthetic_dataset(_config(), leakage="future", effect_sizes={"leakage": value})

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_group_effect_size_out_of_range_is_rejected(self, value):
        with pytest.raises(ConfigError, match="effect_sizes"):
            make_synthetic_dataset(_config(), groups=10, effect_sizes={"group": value})


class TestDeterminism:
    def test_same_config_and_kwargs_give_byte_identical_output(self):
        kwargs = dict(
            leakage=("future", "duplicate"),
            temporal=True,
            groups=20,
            label_noise=0.05,
            task="regression",
        )
        first = make_synthetic_dataset(_config(), **kwargs)
        second = make_synthetic_dataset(_config(), **kwargs)

        pd.testing.assert_frame_equal(first.dataset.data, second.dataset.data)
        assert first.flipped_indices == second.flipped_indices

    def test_a_different_seed_changes_the_output(self):
        first = make_synthetic_dataset(_config(random_seed=1))
        second = make_synthetic_dataset(_config(random_seed=2))

        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(first.dataset.data, second.dataset.data)


class TestPinnedOutput:
    def test_exact_values_for_a_fixed_seed(self):
        # Pinned from numpy's frozen RandomState stream, the same style as
        # test_guardrails.py's row-selection pin. The exact-correlation and exact-eta-squared
        # tests are blind to a bug that relabels which row or group gets which value (the
        # aggregate statistic comes out the same either way), so this pins the actual per-row
        # wiring: which date and which group each row lands on, not just the resulting
        # association strength.
        config = SyntheticConfig(n_rows=20, n_features=1, random_seed=7)

        result = make_synthetic_dataset(
            config,
            temporal=True,
            groups=4,
            task="regression",
            effect_sizes={"temporal": 0.5, "group": 0.5},
        )
        frame = result.dataset.data

        assert frame["event_time"].astype(str).tolist()[:5] == [
            "2020-01-14",
            "2020-01-12",
            "2020-01-03",
            "2020-01-18",
            "2020-01-19",
        ]
        assert frame["group_id"].tolist()[:5] == [
            "group_003",
            "group_000",
            "group_001",
            "group_003",
            "group_003",
        ]
        np.testing.assert_allclose(
            frame["target"].to_numpy()[:3],
            [1.932120288838592, -1.4572189656606902, 0.2645942721362703],
        )
        np.testing.assert_allclose(
            frame["feature_0"].to_numpy()[:3],
            [2.1830908622109977, -1.3860041892375947, -1.77182221406411],
        )


class TestCleanDataset:
    def test_no_mechanism_columns_are_present_unless_requested(self):
        frame = make_synthetic_dataset(_config()).dataset.data

        assert set(frame.columns) == {"feature_0", "feature_1", "feature_2", "target"}

    def test_target_is_set_on_the_dataset(self):
        dataset = make_synthetic_dataset(_config()).dataset

        assert dataset.target == "target"


class TestLeakageFuture:
    def test_regression_hits_the_requested_correlation_exactly(self):
        result = make_synthetic_dataset(
            _config(), leakage="future", task="regression", effect_sizes={"leakage": 0.9}
        )
        frame = result.dataset.data

        r = np.corrcoef(frame["leak_future"], frame["target"])[0, 1]

        assert r == pytest.approx(0.9, abs=TOLERANCE)

    def test_classification_leak_stands_out_from_an_ordinary_feature(self):
        frame = make_synthetic_dataset(
            _config(), leakage="future", task="classification", effect_sizes={"leakage": 0.98}
        ).dataset.data

        leak_association = _eta_squared(frame["leak_future"], frame["target"])
        feature_association = _eta_squared(frame["feature_0"], frame["target"])

        assert leak_association > 0.5
        assert leak_association > feature_association


class TestLeakageDuplicate:
    def test_regression_is_an_exact_affine_recoding_of_the_target(self):
        frame = make_synthetic_dataset(
            _config(), leakage="duplicate", task="regression"
        ).dataset.data

        recovered = (frame["leak_duplicate"] - 7.0) / 1000.0

        np.testing.assert_allclose(recovered.to_numpy(), frame["target"].to_numpy())

    def test_classification_is_a_bijection_with_the_target(self):
        frame = make_synthetic_dataset(
            _config(), leakage="duplicate", task="classification"
        ).dataset.data

        pairs = frame[["target", "leak_duplicate"]].drop_duplicates()

        assert pairs["target"].nunique() == len(pairs)
        assert pairs["leak_duplicate"].nunique() == len(pairs)


class TestTemporal:
    def test_regression_hits_the_requested_correlation_exactly(self):
        frame = make_synthetic_dataset(
            _config(), temporal=True, task="regression", effect_sizes={"temporal": 0.7}
        ).dataset.data

        time_ordinal = frame["event_time"].astype("int64")
        r = np.corrcoef(time_ordinal, frame["target"])[0, 1]

        assert r == pytest.approx(0.7, abs=TOLERANCE)

    def test_classification_drift_is_detectable(self):
        frame = make_synthetic_dataset(
            _config(), temporal=True, task="classification", effect_sizes={"temporal": 0.9}
        ).dataset.data

        time_ordinal = frame["event_time"].astype("int64")
        association = _eta_squared(pd.Series(time_ordinal), frame["target"])

        assert association > 0.2


class TestGroups:
    def test_regression_hits_the_requested_eta_squared_exactly(self):
        frame = make_synthetic_dataset(
            _config(), groups=40, task="regression", effect_sizes={"group": 0.6}
        ).dataset.data

        association = _eta_squared(frame["target"], frame["group_id"])

        assert association == pytest.approx(0.6, abs=TOLERANCE)

    def test_classification_groups_are_target_homogeneous(self):
        frame = make_synthetic_dataset(
            _config(), groups=40, task="classification", effect_sizes={"group": 0.95}
        ).dataset.data

        assert _mean_group_modal_share(frame) > 0.8


class TestLabelNoise:
    def test_no_noise_leaves_the_target_untouched(self):
        result = make_synthetic_dataset(_config())

        assert result.flipped_indices == ()

    def test_flipped_indices_are_exactly_the_rows_that_differ(self):
        clean = make_synthetic_dataset(_config()).dataset.data["target"].to_numpy()
        result = make_synthetic_dataset(_config(), label_noise=0.1)
        noisy = result.dataset.data["target"].to_numpy()

        differing = tuple(int(i) for i in np.flatnonzero(clean != noisy))

        assert differing == result.flipped_indices
        assert len(result.flipped_indices) == round(0.1 * 2000)

    def test_regression_flipped_indices_are_exactly_the_rows_that_differ(self):
        clean = make_synthetic_dataset(_config(), task="regression").dataset.data
        result = make_synthetic_dataset(_config(), task="regression", label_noise=0.1)
        noisy = result.dataset.data

        differing = tuple(
            int(i) for i in np.flatnonzero(clean["target"].to_numpy() != noisy["target"].to_numpy())
        )

        assert differing == result.flipped_indices

    def test_classification_flip_always_changes_the_class(self):
        clean = make_synthetic_dataset(_config(n_classes=3)).dataset.data["target"]
        result = make_synthetic_dataset(_config(n_classes=3), label_noise=0.2)
        noisy = result.dataset.data["target"]

        for position in result.flipped_indices:
            assert noisy.iloc[position] != clean.iloc[position]
