"""Analysis configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datadoctor.core.exceptions import ConfigError
from datadoctor.core.serialization import JsonSerializable, require_keys

_INT_FIELDS = ("random_seed", "row_threshold", "column_threshold")
_FIELDS = (*_INT_FIELDS, "output_dir")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisConfig(JsonSerializable):
    """Settings that control an analysis run.

    Attributes:
        random_seed: Seed for every source of randomness. Same input and seed give the same
            output. Must be zero or greater.
        row_threshold: Above this many rows, analyses that apply the size guardrails work on a
            seeded sample of exactly this many rows. Analyses that need exact counts, such as
            the schema profile, always use every row.
        column_threshold: Above this many columns, pairwise computations are skipped.
        output_dir: The directory that saved files, such as plots, are written to. It is created
            when the first file is saved. A string or a path is accepted and stored as a ``Path``.
            Relative paths are relative to the working directory. Files with the same name are
            replaced.
    """

    random_seed: int = 42
    row_threshold: int = 100_000
    column_threshold: int = 100
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if self.random_seed < 0:
            raise ConfigError(f"random_seed must be zero or greater, got {self.random_seed}")
        for name in ("row_threshold", "column_threshold"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not isinstance(self.output_dir, str | os.PathLike):
            raise TypeError(f"output_dir must be a string or a path, got {self.output_dir!r}")
        if isinstance(self.output_dir, str) and not self.output_dir.strip():
            raise ConfigError("output_dir must not be empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def to_dict(self) -> dict[str, Any]:
        """Return the seed, both thresholds and ``output_dir`` as a POSIX path string."""
        data = {name: getattr(self, name) for name in _INT_FIELDS}
        return {**data, "output_dir": self.output_dir.as_posix()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        """Rebuild from a ``to_dict`` result. Every key is optional; missing ones use defaults."""
        require_keys("AnalysisConfig", data, optional=_FIELDS)
        return cls(**data)
