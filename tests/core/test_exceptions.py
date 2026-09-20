import pytest

from datadoctor.core.exceptions import (
    ConfigError,
    DataDoctorError,
    DatasetError,
    SerializationError,
)


@pytest.mark.parametrize("error", [ConfigError, DatasetError, SerializationError])
def test_concrete_errors_are_both_package_errors_and_value_errors(error):
    assert issubclass(error, DataDoctorError)
    assert issubclass(error, ValueError)

    with pytest.raises(ValueError):
        raise error("boom")
    with pytest.raises(DataDoctorError):
        raise error("boom")


def test_base_error_is_not_a_value_error():
    assert not issubclass(DataDoctorError, ValueError)
