"""Recommendations engine: a ranked, deduplicated action list across every analyzer.

``build_recommendations`` takes the already-computed results from quality, eda, splits, leakage
and readiness and produces a ranked list of :class:`Recommendation` entries. It computes nothing
new: no new ``Finding``, no new statistic, only a re-sorting and a narrow, explicit
deduplication of what those five analyzers already reported.

Readiness's own findings are almost entirely excluded. Its headline is a summary of what the
other four already say, not a root cause of its own; including it here would double-count
whatever it summarizes. Only readiness's target-unusable finding participates, since it is the
one finding readiness produces that no other analyzer also produces. It is identified by
elimination: the headline's own title is dynamic (it includes the blocking count), so rather than
pattern-match that text, everything from readiness that is *not* the target-unusable title is
dropped, since readiness's own findings tuple only ever holds these two kinds.

Deduplication is not "same column": two different problems on the same column are two different
recommendations, full stop -- that is the default, not an edge case. Merging is a narrow,
explicit exception: a fixed registry of ``(category, title)`` pairs, each individually verified
by reading the analyzers' own code (not by title-matching alone) to represent the same fact
reported twice:

- ``eda``'s and ``readiness``'s target-type findings fire under the identical condition (the
  target's schema type is not numeric, categorical or boolean), even though neither imports a
  shared constant to prove it -- confirmed by inspection.
- ``eda``'s class-imbalance finding and ``split_strategy``'s stratification finding import the
  *same* ``IMBALANCE_MEDIUM``/``IMBALANCE_LOW``/``DISCRETE_MAX_DISTINCT`` thresholds from
  ``eda.relationships`` and fire on the same condition; on ``wine_quality.csv`` both report the
  smallest class at 0.1% of the rows.

Two findings only merge when both conditions hold: their ``(category, title)`` pair is in the
*same* registered group, and their ``affected_columns`` match exactly. Nothing else merges,
regardless of shared columns: a missingness finding and an outlier finding naming the same column
are a different problem each and stay as two entries.

Before that group-based merge runs, an exact-duplicate ``Finding`` -- the same object, byte for
byte, most commonly the guardrail notice that up to three of these five analyzers can
independently attach when a dataset is sampled -- is collapsed first, the same ``dict.fromkeys``
mechanism ``quality.run`` and ``eda.run`` already use for their own cross-analyzer ties.

A merged entry never loses a reason: both original ``Finding`` objects are kept, verbatim, in its
``findings`` tuple. Its severity is the higher of the two. Its confidence travels paired with
whichever original finding has that higher severity, or, when severities tie, whichever has the
higher confidence -- never maximized independently of severity, so a severity is never reported
alongside a confidence that did not actually accompany it in either source finding.

Ranking is severity first, then confidence, both descending, extending ``sort_findings``'s own
principle across analyzers rather than within one. Ties beyond that keep the fixed order the
analyzers are read in: quality, eda, splits, leakage, readiness -- the same "stable sort over a
documented order" mechanism ``quality.run``'s own merged findings list already relies on.

``Recommendation.to_finding`` flattens one entry back into a single ``Finding``, for callers
(``diagnostics.run``, S30) that want recommendations as their primary ``findings`` output rather
than a separate structure. Title comes from the first source finding; evidence, interpretation,
limitations and recommendation text are each joined across every source, so a merge never drops
a reason a caller reading only the flattened form would otherwise lose; severity and confidence
are the ones already computed onto the ``Recommendation`` itself. Title and severity can come
from different sources when they disagree -- the flattened form is a summary, not a rewrite.

Joining text across two sources can produce a near-restate rather than two genuinely different
sentences -- eda's and split_strategy's own phrasing of the same class-imbalance fact, for
example, differ only in whether the minority label is named. ``_join`` drops one side of a pair
only when one text, trailing periods stripped, is a literal substring of the other, keeping the
longer (strictly more informative) one; two texts that are not in a substring relationship are
always both kept, even when they describe the same underlying fact in unrelated words.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.core.serialization import JsonSerializable, ensure_json_roundtrip, require_keys
from datadoctor.diagnostics.readiness_findings import TARGET_UNUSABLE_TITLE as READINESS_UNUSABLE
from datadoctor.diagnostics.splits_findings import STRATIFIED_TITLE
from datadoctor.eda.relationships_findings import IMBALANCE_TITLE
from datadoctor.eda.relationships_findings import TARGET_UNUSABLE_TITLE as EDA_UNUSABLE

# Splits' and leakage's other findings (temporal, grouped, small-n, near-perfect, duplicate,
# naming) need no entry here: they merge with nothing and rank purely on their own severity and
# confidence. Only titles that are part of a registered group below are referenced at all.

# Each group is a set of (category, title) pairs verified, by reading the analyzers' own code, to
# represent the same fact. See the module docstring for why each group is here.
#
# Each group's own recommendation -- category "recommendation", the title Recommendation.to_finding
# gives it -- is registered alongside its raw sources, not in a separate group: to_finding's title
# always comes from findings[0], which for both groups below is the eda finding, so the
# recommendation's title is always that same eda title. Keeping it in the same group (rather than
# a new one) is required, not stylistic: _group_index returns the first group a key is found in,
# so if ("eda", IMBALANCE_TITLE) were also registered in a second, separate group, _group_index
# would never return that second group for it, and drop_findings_superseded_by_recommendations
# below would never see the two as sharing a group. See that function for what this is for.
_ROOT_CAUSE_GROUPS: tuple[frozenset[tuple[str, str]], ...] = (
    frozenset(
        {("eda", EDA_UNUSABLE), ("readiness", READINESS_UNUSABLE), ("recommendation", EDA_UNUSABLE)}
    ),
    frozenset(
        {
            ("eda", IMBALANCE_TITLE),
            ("split_strategy", STRATIFIED_TITLE),
            ("recommendation", IMBALANCE_TITLE),
        }
    ),
)


def _group_index(finding: Finding) -> int | None:
    key = (finding.category, finding.title)
    for index, group in enumerate(_ROOT_CAUSE_GROUPS):
        if key in group:
            return index
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class Recommendation(JsonSerializable):
    """One ranked action item: one or more findings that share a root cause.

    Attributes:
        severity: The higher of the merged findings' severities, or the one finding's own.
        confidence: The confidence of whichever finding contributed ``severity`` (the more
            severe one, or the more confident one if severities tie). Never independently
            maximized: a severity is never paired with a confidence that did not accompany it.
        columns: The union of the merged findings' ``affected_columns``.
        findings: The original findings, verbatim and in full, one for a singleton entry and
            two when two analyzers reported the same root cause. Nothing here is paraphrased.
    """

    severity: Severity
    confidence: float
    columns: tuple[str, ...]
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dict, ``findings`` recursively."""
        return {
            "severity": self.severity.value,
            "confidence": self.confidence,
            "columns": list(self.columns),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recommendation":
        """Rebuild from a ``to_dict`` result."""
        require_keys(
            "Recommendation", data, required=("severity", "confidence", "columns", "findings")
        )
        return cls(
            severity=Severity(data["severity"]),
            confidence=data["confidence"],
            columns=tuple(data["columns"]),
            findings=tuple(Finding.from_dict(f) for f in data["findings"]),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            raise TypeError(f"severity must be a Severity, got {self.severity!r}")
        if not self.findings:
            raise ValueError("a recommendation must hold at least one finding")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "findings", tuple(self.findings))
        ensure_json_roundtrip("Recommendation.confidence", self.confidence)

    def to_finding(self) -> Finding:
        """Flatten this recommendation into one ``Finding``.

        See the module docstring for what is joined, what is kept as-is, and why the title and
        severity can come from different source findings when this entry is a merge.
        """
        return Finding(
            category="recommendation",
            severity=self.severity,
            confidence=self.confidence,
            title=self.findings[0].title,
            evidence=_join(f.evidence for f in self.findings),
            interpretation=_join(f.interpretation for f in self.findings),
            limitations=_join(f.limitations for f in self.findings),
            affected_columns=self.columns,
            recommendation=_join(f.recommendation for f in self.findings) or None,
        )


def _join(texts: Iterable[str | None]) -> str:
    # dict.fromkeys drops an exact repeat (both sources phrasing a fact identically) while
    # keeping the order the source findings were read in.
    exact = list(dict.fromkeys(text for text in texts if text))
    # A near-restate isn't byte-identical, so dict.fromkeys above lets it through. When one
    # string, trailing periods stripped, is a substring of another, it says nothing the longer
    # one doesn't already say, so only the longer one is kept. Two texts that genuinely differ
    # are never touched by this.
    kept: list[str] = []
    for text in exact:
        if any(_contains(longer, text) for longer in kept):
            continue
        kept = [longer for longer in kept if not _contains(text, longer)]
        kept.append(text)
    return "; ".join(kept)


def _contains(outer: str, inner: str) -> bool:
    """Whether ``inner``, trailing periods stripped, is a substring of ``outer``, likewise."""
    return inner.rstrip(".") in outer.rstrip(".")


def build_recommendations(
    *,
    quality: AnalysisResult,
    eda: AnalysisResult,
    splits: AnalysisResult,
    leakage: AnalysisResult,
    readiness: AnalysisResult,
) -> tuple[Recommendation, ...]:
    """Merge and rank the findings from every analyzer into one action list.

    See the module docstring for exactly what merges, what never does, and how a merged entry's
    severity and confidence are chosen. Readiness's headline is excluded entirely; only its
    target-unusable finding (when present) takes part.

    Args:
        quality: The result of ``run_quality_checks``.
        eda: The result of ``run_eda``.
        splits: The result of ``check_split_strategy``.
        leakage: The result of ``check_leakage``.
        readiness: The result of ``check_readiness``.

    Returns:
        Every input finding, deduplicated and merged where a root cause is shared, ranked most
        severe first and by confidence within a severity.
    """
    ordered_findings = (
        *quality.findings,
        *eda.findings,
        *splits.findings,
        *leakage.findings,
        *(f for f in readiness.findings if f.title == READINESS_UNUSABLE),
    )
    # dict.fromkeys drops exact repeats (the same Finding object reported by more than one
    # analyzer, such as a shared guardrail notice) and keeps the first of each, so ties below
    # stay in this same, deterministic order.
    unique_findings = tuple(dict.fromkeys(ordered_findings))

    groups: dict[tuple[int, frozenset[str]], list[Finding]] = {}
    singles: list[Finding] = []
    for finding in unique_findings:
        index = _group_index(finding)
        if index is None:
            singles.append(finding)
            continue
        key = (index, frozenset(finding.affected_columns))
        groups.setdefault(key, []).append(finding)

    entries = [_merge((finding,)) for finding in singles]
    entries.extend(_merge(tuple(group)) for group in groups.values())

    return tuple(sorted(entries, key=lambda entry: (-entry.severity.rank, -entry.confidence)))


def _merge(findings: tuple[Finding, ...]) -> Recommendation:
    winner = max(findings, key=lambda finding: (finding.severity.rank, finding.confidence))
    columns = tuple(dict.fromkeys(c for finding in findings for c in finding.affected_columns))
    return Recommendation(
        severity=winner.severity, confidence=winner.confidence, columns=columns, findings=findings
    )


def drop_findings_superseded_by_recommendations(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Drop a raw finding when a recommendation covering the same root cause is also present.

    A ``category="recommendation"`` finding (``Recommendation.to_finding()``) for one of the
    registered groups already carries everything its raw eda source finding does -- evidence,
    interpretation, limitations and recommendation text joined across every finding that
    contributed to it, see the module docstring -- and more. A caller that combines findings
    from several already-merged results, such as ``report`` combining ``quality``, ``eda`` and
    ``diagnose`` together, would otherwise show the same fact twice: once as eda's own raw
    finding, once again inside diagnose's fuller, already-joined recommendation.

    Only a finding whose ``(category, title)`` is in the same registered group as a present
    recommendation, and whose ``affected_columns`` match exactly, is dropped. A recommendation
    itself is never dropped, and anything not in a registered group -- everything from quality,
    most of eda, and any singleton recommendation -- is never touched.

    ``findings`` is assumed already free of exact byte-identical repeats (``dict.fromkeys``);
    this runs after that step, not instead of it.
    """
    findings = tuple(findings)
    covered = {
        (_group_index(finding), frozenset(finding.affected_columns))
        for finding in findings
        if finding.category == "recommendation" and _group_index(finding) is not None
    }
    return tuple(
        finding
        for finding in findings
        if finding.category == "recommendation"
        or (_group_index(finding), frozenset(finding.affected_columns)) not in covered
    )
