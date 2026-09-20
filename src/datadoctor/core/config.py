"""Analysis configuration."""

from dataclasses import dataclass
from typing import Any

from datadoctor.core.exceptions import ConfigError
from datadoctor.core.serialization import JsonSerializable, require_keys

_FIELDS = ("random_seed", "row_threshold", "column_threshold")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisConfig(JsonSerializable):
    """Settings that control an analysis run.

    Attributes:
        random_seed: Seed for every source of randomness. Same input and seed give the same
            output. Must be zero or greater.
        row_threshold: Above this many rows, analyses work on a deterministic sample.
        column_threshold: Above this many columns, pairwise computations are skipped.
    """

    random_seed: int = 42
    row_threshold: int = 100_000
    column_threshold: int = 100

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if self.random_seed < 0:
            raise ConfigError(f"random_seed must be zero or greater, got {self.random_seed}")
        for name in ("row_threshold", "column_threshold"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        require_keys("AnalysisConfig", data, optional=_FIELDS)
        return cls(**data)
