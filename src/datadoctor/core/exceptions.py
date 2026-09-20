"""Exception hierarchy for DataDoctor.

Every package error derives from ``DataDoctorError`` so interfaces can catch one type. Errors
about a bad value also derive from ``ValueError``, so callers that catch ``ValueError`` keep
working.
"""


class DataDoctorError(Exception):
    """Base class for all errors raised by DataDoctor."""


class ConfigError(DataDoctorError, ValueError):
    """An analysis configuration value is invalid."""


class DatasetError(DataDoctorError, ValueError):
    """A dataset is inconsistent with what was asked of it."""


class SerializationError(DataDoctorError, ValueError):
    """A value could not be converted to or from JSON."""


class DataLoadError(DataDoctorError):
    """A data file could not be read into a dataset."""
