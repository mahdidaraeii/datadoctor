"""The findings ``check_readiness`` reports itself, and the wording of each.

check_readiness composes facts and findings that other analyzers already computed; it does not
re-emit them as new findings of its own (see ``readiness.py``'s module notes). Its own findings
are exactly these two: a headline summarizing the overall state, and a target-type finding for
the one blocking condition that has no other analyzer to source it from.
"""

from datadoctor.core.result import Finding, Severity
from datadoctor.diagnostics.splits_findings import GROUPED_TITLE, TEMPORAL_TITLE

LISTED = 5

# The source finding's own title, softened for readiness's summary where the title as written
# would overstate what was actually found. The split-strategy check found an association
# between the time column and the target, not a certainty that the split leaks; only this one
# title is known to overstate in this way, so only it is remapped.
_HEADLINE_PHRASING = {
    TEMPORAL_TITLE: "A random split is likely to leak future information",
}


def _headline_text(entry: dict) -> str:
    text = _HEADLINE_PHRASING.get(entry["title"], entry["title"])
    return f"{text} ({', '.join(entry['columns'])})" if entry["columns"] else text


def _listed_entries(entries: list[dict]) -> str:
    shown = [_headline_text(e) for e in entries[:LISTED]]
    text = "; ".join(shown)
    more = len(entries) - LISTED
    return text + (f", and {more} more" if more > 0 else "")


def not_assessed_finding() -> Finding:
    """No target is set, so none of the checks readiness composes could run."""
    return Finding(
        category="readiness",
        severity=Severity.INFO,
        confidence=1.0,
        title="Readiness was not assessed",
        evidence=(
            "No target column is set. Every check readiness composes needs one: target "
            "constancy, target missingness, target usability, split strategy and leakage."
        ),
        interpretation="Readiness cannot be judged without knowing what a model would predict.",
        limitations="This says nothing about the data itself, only that it was not checked.",
        recommendation="Set a target column, then run this check again.",
    )


def target_unusable_finding(name: str, kind: str) -> Finding:
    """The target's schema-inferred type is not one this project treats as modelable."""
    return Finding(
        category="readiness",
        severity=Severity.CRITICAL,
        confidence=1.0,
        title="Target column's type cannot be modeled",
        evidence=(
            f"Column {name} is typed {kind}, which this project does not treat as a regression "
            "or classification target."
        ),
        interpretation=(
            "Without a numeric or categorical target, no supervised model can be fit or "
            "evaluated, and none of the other readiness checks that depend on a task can run."
        ),
        limitations=(
            "The type comes from the schema profile's own heuristic; a column mistyped by that "
            "heuristic may be usable once corrected, and this finding would not know."
        ),
        affected_columns=(name,),
        recommendation=(
            "Confirm the target column, or recode it into a numeric or categorical form."
        ),
    )


def headline_finding(blocking: list[dict], attention: list[dict]) -> Finding:
    """A one-finding summary of what readiness composed, never the underlying evidence itself."""
    if not blocking:
        severity = Severity.MEDIUM if attention else Severity.INFO
        evidence = "No blocking issues were found by these checks."
        if attention:
            count = len(attention)
            noun = "item" if count == 1 else "items"
            verb = "needs" if count == 1 else "need"
            evidence += f" {count} {noun} {verb} attention, listed in the readiness metrics."
        return Finding(
            category="readiness",
            severity=severity,
            confidence=1.0,
            title="No blocking issues were found by these checks",
            evidence=evidence,
            interpretation=(
                "This does not show the dataset is ready to model, only that these specific "
                "checks did not find a blocker. They do not cover every way a dataset can fail."
            ),
            limitations=(
                "This is a summary of what the composed checks found, not a new analysis. Each "
                "source analyzer's own finding, named in the readiness metrics, has the full "
                "evidence and reasoning behind any attention item."
            ),
            recommendation=("Review the attention items before modeling." if attention else None),
        )

    untrusted_split = any(entry["title"] in (TEMPORAL_TITLE, GROUPED_TITLE) for entry in blocking)
    recommendation = "Resolve every blocking issue before modeling."
    if untrusted_split:
        recommendation += (
            " Do not trust evaluation scores until a time-based or grouped split is used, as "
            "the flagged split-strategy issue recommends."
        )
    count = len(blocking)
    return Finding(
        category="readiness",
        severity=Severity.CRITICAL,
        confidence=1.0,
        title=f"{count} blocking {'issue' if count == 1 else 'issues'} found",
        evidence=_listed_entries(blocking) + ".",
        interpretation=(
            "Any blocking issue would make a model's results unreliable or meaningless on its "
            "own. The source analyzer named in the readiness metrics has the full evidence and "
            "reasoning behind it."
        ),
        limitations=(
            "This is a summary of what the composed checks found, not a new analysis: see each "
            "source analyzer's own finding for what was actually computed."
        ),
        affected_columns=tuple(sorted({c for entry in blocking for c in entry["columns"]})),
        recommendation=recommendation,
    )
