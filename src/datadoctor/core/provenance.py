"""Provenance: where a dataset came from and the conditions it was loaded under."""

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.serialization import JsonSerializable, require_keys

_CHUNK_SIZE = 1 << 20
_REQUIRED = (
    "shape",
    "loaded_at",
    "package_version",
    "python_version",
    "pandas_version",
    "config",
)
_OPTIONAL = (
    "file_sha256",
    "file_size_bytes",
    "converted_tokens",
    "unnamed_columns",
    "promoted_index",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance(JsonSerializable):
    """A record of how a dataset was obtained, for reproducing and auditing a result.

    Attributes:
        file_sha256: SHA-256 of the source file's bytes as lowercase hex, or ``None`` when the
            dataset did not come from a file. The hash depends only on the bytes, never on the
            path. The file is hashed in its own pass, not from the stream that was parsed, so a
            file changed between the two would show a hash that does not match the parsed data.
        file_size_bytes: Size of the source file in bytes, or ``None`` without a file.
        shape: ``(n_rows, n_columns)`` of the loaded data.
        loaded_at: When the data was loaded, in UTC as ``YYYY-MM-DDTHH:MM:SSZ``.
        package_version: The DataDoctor version.
        python_version: The Python version, such as ``3.13.7``.
        pandas_version: The pandas version. Results can differ between pandas major versions.
        config: The ``AnalysisConfig`` the dataset was loaded under, including its random seed.
        converted_tokens: Literal cell texts that were read as missing, as
            ``{column: {token: count}}``. A genuine value spelled like a missing-value token
            (the country code ``NA``) is converted too, so this shows when that may have happened.
        unnamed_columns: Columns whose header cell was empty and that were named
            ``Unnamed: <position>``.
        promoted_index: Columns created from a stored index, which parquet files can carry.
    """

    file_sha256: str | None = None
    file_size_bytes: int | None = None
    shape: tuple[int, int]
    loaded_at: str
    package_version: str
    python_version: str
    pandas_version: str
    config: AnalysisConfig
    converted_tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    unnamed_columns: tuple[str, ...] = ()
    promoted_index: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(
            self,
            "converted_tokens",
            {column: dict(tokens) for column, tokens in self.converted_tokens.items()},
        )
        object.__setattr__(self, "unnamed_columns", tuple(self.unnamed_columns))
        object.__setattr__(self, "promoted_index", tuple(self.promoted_index))

    @classmethod
    def capture(
        cls,
        frame: pd.DataFrame,
        *,
        config: AnalysisConfig | None = None,
        file: Path | None = None,
        converted_tokens: dict[str, dict[str, int]] | None = None,
        unnamed_columns: tuple[str, ...] = (),
        promoted_index: tuple[str, ...] = (),
    ) -> "Provenance":
        """Record the current conditions for a loaded frame, hashing ``file`` if there is one."""
        from datadoctor import __version__

        if config is not None and not isinstance(config, AnalysisConfig):
            raise TypeError(f"config must be an AnalysisConfig, got {type(config).__name__}")
        digest, size = (None, None) if file is None else _hash_file(file)
        return cls(
            file_sha256=digest,
            file_size_bytes=size,
            shape=(len(frame), frame.shape[1]),
            loaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            package_version=__version__,
            python_version=platform.python_version(),
            pandas_version=pd.__version__,
            config=AnalysisConfig() if config is None else config,
            converted_tokens={} if converted_tokens is None else converted_tokens,
            unnamed_columns=unnamed_columns,
            promoted_index=promoted_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "file_size_bytes": self.file_size_bytes,
            "shape": list(self.shape),
            "loaded_at": self.loaded_at,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "pandas_version": self.pandas_version,
            "config": self.config.to_dict(),
            "converted_tokens": {
                column: dict(tokens) for column, tokens in self.converted_tokens.items()
            },
            "unnamed_columns": list(self.unnamed_columns),
            "promoted_index": list(self.promoted_index),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        require_keys("Provenance", data, required=_REQUIRED, optional=_OPTIONAL)
        return cls(**{**data, "config": AnalysisConfig.from_dict(data["config"])})


def _hash_file(file: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with file.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
