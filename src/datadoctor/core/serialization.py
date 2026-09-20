"""Strict JSON support shared by every result type."""

import json
from collections.abc import Iterable
from typing import Any, TypeVar

from datadoctor.core.exceptions import SerializationError

T = TypeVar("T", bound="JsonSerializable")


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not valid JSON")


class JsonSerializable:
    """Mixin adding strict JSON text support to a class with ``to_dict`` and ``from_dict``.

    Output is standard JSON: NaN and infinity are rejected in both directions.
    """

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        try:
            return json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SerializationError(f"cannot serialize {type(self).__name__}: {exc}") from exc

    @classmethod
    def from_json(cls: type[T], text: str) -> T:
        """Rebuild an instance from a string produced by ``to_json``."""
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except ValueError as exc:
            raise SerializationError(f"invalid JSON for {cls.__name__}: {exc}") from exc
        return cls.from_dict(data)


def require_keys(
    kind: str,
    data: Any,
    *,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
) -> None:
    """Check that ``data`` is a mapping with all required keys and no unknown ones."""
    if not isinstance(data, dict):
        raise SerializationError(f"{kind} must be a JSON object, got {type(data).__name__}")
    required = tuple(required)
    missing = [key for key in required if key not in data]
    if missing:
        raise SerializationError(f"{kind} is missing keys: {', '.join(missing)}")
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        raise SerializationError(f"{kind} has unknown keys: {', '.join(unknown)}")


def ensure_json_roundtrip(label: str, value: Any) -> None:
    """Raise ``SerializationError`` unless ``value`` survives a strict JSON round trip unchanged.

    This rejects NaN and infinity, types JSON cannot represent (such as numpy scalars), and
    values that come back different (tuples become lists, integer dict keys become strings).
    """
    try:
        restored = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"{label} is not JSON-serializable: {exc}") from exc
    if restored != value:
        raise SerializationError(f"{label} does not survive a JSON round trip unchanged")
