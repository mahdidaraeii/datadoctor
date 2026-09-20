"""Result types returned by analyzers."""

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datadoctor.core.exceptions import SerializationError
from datadoctor.core.serialization import JsonSerializable, ensure_json_roundtrip, require_keys


class Severity(str, Enum):
    """How much a finding matters if it is real.

    Severity is impact. It says nothing about how sure we are; that is
    ``Finding.confidence``. The two are never combined.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        """Integer rank, higher is more severe. Use this, never string order."""
        return _RANK[self]


_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

_FINDING_REQUIRED = (
    "category",
    "severity",
    "confidence",
    "title",
    "evidence",
    "interpretation",
    "limitations",
)
_FINDING_OPTIONAL = ("affected_columns", "recommendation")


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding(JsonSerializable):
    """One diagnostic result: what was computed, what it might mean, what is uncertain.

    Attributes:
        category: Kind of diagnostic, for example ``"data_quality"``.
        severity: Impact if the finding is real.
        confidence: Certainty that the finding is real, from 0.0 to 1.0.
        title: Short label.
        evidence: What was actually computed. Facts only.
        interpretation: What the evidence might mean. Label heuristics as such.
        limitations: What is uncertain or was not tested.
        affected_columns: Columns the finding refers to, possibly none.
        recommendation: Suggested action, if there is one.
    """

    category: str
    severity: Severity
    confidence: float
    title: str
    evidence: str
    interpretation: str
    limitations: str
    affected_columns: tuple[str, ...] = ()
    recommendation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            raise TypeError(f"severity must be a Severity, got {self.severity!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence!r}")
        if isinstance(self.affected_columns, str):
            raise TypeError("affected_columns must be a sequence of names, not a single string")
        object.__setattr__(self, "affected_columns", tuple(self.affected_columns))

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "title": self.title,
            "evidence": self.evidence,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
            "affected_columns": list(self.affected_columns),
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        require_keys("Finding", data, required=_FINDING_REQUIRED, optional=_FINDING_OPTIONAL)
        try:
            severity = Severity(data["severity"])
        except ValueError as exc:
            raise SerializationError(f"unknown severity {data['severity']!r}") from exc
        return cls(**{**data, "severity": severity})


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Return findings ordered from most to least severe.

    The sort is stable and uses severity only. Confidence is deliberately not a
    tie-breaker, so equally severe findings keep their input order.
    """
    return sorted(findings, key=lambda finding: finding.severity.rank, reverse=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult(JsonSerializable):
    """Everything an analysis produced: findings, numbers and files.

    Attributes:
        findings: Diagnostic results, in the order the analyzers produced them.
        metrics: Named numbers and small structures. Every value must survive a strict JSON
            round trip, so use ``None`` for an undefined value, never NaN, and plain Python types.
        artifacts: Files written during the analysis, such as plots, keyed by a stable name.
    """

    findings: tuple[Finding, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        for finding in findings:
            if not isinstance(finding, Finding):
                raise TypeError(f"findings must contain Finding objects, got {finding!r}")
        object.__setattr__(self, "findings", findings)

        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise TypeError(f"metric names must be strings, got {key!r}")
            ensure_json_roundtrip(f"metric {key!r}", value)
        object.__setattr__(self, "metrics", dict(self.metrics))

        for key in self.artifacts:
            if not isinstance(key, str):
                raise TypeError(f"artifact names must be strings, got {key!r}")
        object.__setattr__(
            self, "artifacts", {key: Path(path) for key, path in self.artifacts.items()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": copy.deepcopy(self.metrics),
            "artifacts": {key: path.as_posix() for key, path in self.artifacts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        require_keys("AnalysisResult", data, optional=("findings", "metrics", "artifacts"))
        return cls(
            findings=tuple(Finding.from_dict(item) for item in data.get("findings", [])),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
        )
