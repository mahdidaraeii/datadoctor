"""Result types returned by analyzers."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
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


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Return findings ordered from most to least severe.

    The sort is stable and uses severity only. Confidence is deliberately not a
    tie-breaker, so equally severe findings keep their input order.
    """
    return sorted(findings, key=lambda finding: finding.severity.rank, reverse=True)
