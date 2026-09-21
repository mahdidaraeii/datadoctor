import platform
from datetime import datetime, timezone

import pandas as pd
import pytest

import datadoctor
from datadoctor import AnalysisConfig, Dataset, Provenance


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestCapture:
    def test_records_shape_versions_and_a_current_utc_timestamp(self, frame):
        before = datetime.now(timezone.utc).replace(microsecond=0)

        provenance = Provenance.capture(frame)

        after = datetime.now(timezone.utc)
        recorded = datetime.strptime(provenance.loaded_at, "%Y-%m-%dT%H:%M:%SZ")
        assert before <= recorded.replace(tzinfo=timezone.utc) <= after
        assert provenance.shape == (3, 2)
        assert provenance.package_version == datadoctor.__version__
        assert provenance.python_version == platform.python_version()
        assert provenance.pandas_version == pd.__version__

    def test_records_the_config_and_its_seed(self, frame):
        config = AnalysisConfig(random_seed=7, row_threshold=500, column_threshold=12)

        assert Provenance.capture(frame, config=config).config == config

    def test_defaults_to_the_default_config_and_no_file(self, frame):
        provenance = Provenance.capture(frame)

        assert provenance.config == AnalysisConfig()
        assert provenance.file_sha256 is None
        assert provenance.file_size_bytes is None
        assert provenance.converted_tokens == {}
        assert provenance.unnamed_columns == ()
        assert provenance.promoted_index == ()
        assert provenance.leading_zeros == {}

    def test_config_must_be_an_analysis_config(self, frame):
        with pytest.raises(TypeError, match="AnalysisConfig"):
            Provenance.capture(frame, config={"random_seed": 7})

    def test_json_round_trip_keeps_every_field(self, frame):
        provenance = Provenance.capture(
            frame,
            config=AnalysisConfig(random_seed=7),
            converted_tokens={"city": {"NA": 2, "null": 1}},
            unnamed_columns=("Unnamed: 1",),
            promoted_index=("key",),
            leading_zeros={"zip": {"values": 3, "checked": 10, "width": 5}},
        )

        assert Provenance.from_json(provenance.to_json()) == provenance

    def test_a_record_saved_before_leading_zeros_were_recorded_still_loads(self, frame):
        saved = Provenance.capture(frame).to_dict()
        del saved["leading_zeros"]

        assert Provenance.from_dict(saved).leading_zeros == {}


class TestDatasetProvenance:
    def test_a_dataset_without_provenance_gets_a_default_one(self, frame):
        dataset = Dataset(data=frame, name="t")

        assert dataset.provenance is not None
        assert dataset.provenance.shape == (3, 2)
        assert dataset.provenance.file_sha256 is None

    def test_an_explicit_provenance_is_kept(self, frame):
        provenance = Provenance.capture(frame, config=AnalysisConfig(random_seed=9))

        assert Dataset(data=frame, name="t", provenance=provenance).provenance is provenance
