import pandas as pd
import pytest

from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [21, 35, 58], "income": [30.5, None, 72.0], "churn": [0, 1, 0]})


class TestConstruction:
    def test_metadata_fields(self, frame):
        dataset = Dataset(data=frame, name="customers", source="data/customers.csv", target="churn")

        assert dataset.data is frame
        assert dataset.name == "customers"
        assert dataset.source == "data/customers.csv"
        assert dataset.target == "churn"

    def test_source_and_target_are_optional(self, frame):
        dataset = Dataset(data=frame, name="customers")

        assert dataset.source is None
        assert dataset.target is None

    def test_frame_is_neither_copied_nor_modified(self, frame):
        snapshot = frame.copy(deep=True)

        dataset = Dataset(data=frame, name="customers", target="churn")

        assert dataset.data is frame
        pd.testing.assert_frame_equal(frame, snapshot)

    def test_is_immutable(self, frame):
        dataset = Dataset(data=frame, name="customers")

        with pytest.raises(AttributeError):
            dataset.name = "other"

    def test_equality_is_by_identity_and_does_not_compare_frames(self, frame):
        first = Dataset(data=frame, name="customers")
        second = Dataset(data=frame, name="customers")

        assert first == first
        assert first != second


class TestValidation:
    def test_target_must_be_a_column(self, frame):
        with pytest.raises(DatasetError, match="'salary'"):
            Dataset(data=frame, name="customers", target="salary")

    def test_target_lookup_is_exact(self, frame):
        with pytest.raises(DatasetError):
            Dataset(data=frame, name="customers", target="Churn")

    @pytest.mark.parametrize("data", [None, [[1, 2]], {"a": [1]}, "a.csv"])
    def test_data_must_be_a_dataframe(self, data):
        with pytest.raises(TypeError, match="DataFrame"):
            Dataset(data=data, name="customers")
