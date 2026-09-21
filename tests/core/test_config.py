from pathlib import Path

import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.exceptions import ConfigError, SerializationError


class TestConstruction:
    def test_defaults(self):
        config = AnalysisConfig()

        assert config.random_seed == 42
        assert config.row_threshold == 100_000
        assert config.column_threshold == 100

    def test_explicit_values(self):
        config = AnalysisConfig(random_seed=7, row_threshold=500, column_threshold=12)

        assert (config.random_seed, config.row_threshold, config.column_threshold) == (7, 500, 12)

    def test_zero_seed_is_allowed(self):
        assert AnalysisConfig(random_seed=0).random_seed == 0

    def test_is_immutable(self):
        config = AnalysisConfig()

        with pytest.raises(AttributeError):
            config.random_seed = 1

    def test_positional_construction_is_rejected(self):
        with pytest.raises(TypeError):
            AnalysisConfig(1, 2, 3)


class TestOutputDir:
    def test_the_default_is_a_relative_outputs_directory(self):
        assert AnalysisConfig().output_dir == Path("outputs")

    def test_a_string_and_a_path_mean_the_same_and_are_stored_as_a_path(self, tmp_path):
        assert AnalysisConfig(output_dir="plots") == AnalysisConfig(output_dir=Path("plots"))
        assert AnalysisConfig(output_dir=str(tmp_path)).output_dir == tmp_path

    def test_an_empty_string_is_rejected(self):
        with pytest.raises(ConfigError, match="output_dir"):
            AnalysisConfig(output_dir="  ")

    @pytest.mark.parametrize("value", [5, None, ["plots"], True])
    def test_other_types_are_rejected(self, value):
        with pytest.raises(TypeError, match="output_dir"):
            AnalysisConfig(output_dir=value)


class TestValidation:
    def test_negative_seed_is_rejected(self):
        with pytest.raises(ConfigError, match="random_seed"):
            AnalysisConfig(random_seed=-1)

    @pytest.mark.parametrize("name", ["row_threshold", "column_threshold"])
    @pytest.mark.parametrize("value", [0, -5])
    def test_non_positive_thresholds_are_rejected(self, name, value):
        with pytest.raises(ConfigError, match=name):
            AnalysisConfig(**{name: value})

    @pytest.mark.parametrize("name", ["random_seed", "row_threshold", "column_threshold"])
    @pytest.mark.parametrize("value", [1.5, "10", None, True])
    def test_non_integers_are_rejected(self, name, value):
        with pytest.raises(TypeError, match=name):
            AnalysisConfig(**{name: value})


class TestSerialization:
    def test_dict_round_trip(self):
        config = AnalysisConfig(random_seed=7, row_threshold=500, column_threshold=12)

        assert AnalysisConfig.from_dict(config.to_dict()) == config

    def test_json_round_trip(self):
        config = AnalysisConfig(random_seed=7, row_threshold=500, column_threshold=12)

        assert AnalysisConfig.from_json(config.to_json()) == config

    def test_to_dict_has_exactly_the_documented_keys(self):
        assert set(AnalysisConfig().to_dict()) == {
            "random_seed",
            "row_threshold",
            "column_threshold",
            "output_dir",
        }

    def test_the_output_directory_is_stored_as_text_with_forward_slashes(self):
        config = AnalysisConfig(output_dir=Path("results") / "plots")

        assert config.to_dict()["output_dir"] == "results/plots"
        assert AnalysisConfig.from_json(config.to_json()) == config

    def test_a_record_saved_before_output_dir_existed_still_loads(self):
        saved = AnalysisConfig().to_dict()
        del saved["output_dir"]

        assert AnalysisConfig.from_dict(saved).output_dir == Path("outputs")

    def test_missing_keys_fall_back_to_defaults(self):
        assert AnalysisConfig.from_dict({"random_seed": 3}) == AnalysisConfig(random_seed=3)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(SerializationError, match="unknown keys: row_limit"):
            AnalysisConfig.from_dict({"row_limit": 10})

    def test_invalid_stored_value_is_still_validated(self):
        with pytest.raises(ConfigError):
            AnalysisConfig.from_dict({"row_threshold": 0})
