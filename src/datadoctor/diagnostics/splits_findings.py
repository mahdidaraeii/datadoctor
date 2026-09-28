"""The findings ``check_split_strategy`` reports, and the wording of each.

Every builder takes plain values, so this module knows nothing about how they were computed.
"""

from datadoctor.core.result import Finding, Severity

LISTED = 5

# Titles, public so other modules (diagnostics.readiness) can recognize a specific finding
# without hardcoding the string.
TEMPORAL_TITLE = "A random split would leak future information"
GROUPED_TITLE = "A random split could leak an entity across train and test"
SMALL_N_TITLE = "Too few rows for a stable train/test split"
STRATIFIED_TITLE = "Stratified splitting is recommended for this target"


def _listed(names: list[str]) -> str:
    text = ", ".join(names[:LISTED])
    return text + (f" and {len(names) - LISTED} more" if len(names) > LISTED else "")


def temporal_split_finding(columns: list[str]) -> Finding:
    """A datetime column is associated with the target beyond a small effect."""
    plural = len(columns) != 1
    return Finding(
        category="split_strategy",
        severity=Severity.HIGH,
        confidence=1.0,
        title=TEMPORAL_TITLE,
        evidence=(
            f"The target is associated with {_listed(columns)}, "
            f"{'date or time columns' if plural else 'a date or time column'}, beyond a small "
            "effect."
        ),
        interpretation=(
            "When the target's distribution drifts over time, a random split can put a later "
            "row in training and an earlier one in the test set, or the reverse, so a random "
            "split's reported performance can be optimistic compared to predicting the future "
            "from the past."
        ),
        limitations=(
            "The effect-size and significance cutoffs are the same conventions used for every "
            "other association reported in this project, not a judgment on this data."
        ),
        affected_columns=tuple(columns),
        recommendation="Split train and test chronologically instead of at random.",
    )


def grouped_split_finding(columns: list[str], confidence: float) -> Finding:
    """A named, repeated entity column is associated with the target beyond a small effect."""
    plural = len(columns) != 1
    return Finding(
        category="split_strategy",
        severity=Severity.HIGH,
        confidence=confidence,
        title=GROUPED_TITLE,
        evidence=(
            f"{_listed(columns)}, identified as a likely entity column by name and repeated "
            f"values, {'are' if plural else 'is'} associated with the target beyond a small "
            "effect."
        ),
        interpretation=(
            "If the same entity's rows land on both sides of the split, the model can appear to "
            "generalize by recognizing the entity rather than by learning the relationship."
        ),
        limitations=(
            "The column was identified by a name pattern (ending in id, key or similar) with "
            "repeated values, not by confirmed entity structure, which is why this finding's "
            "confidence is below 1.0. An entity column named some other way, such as patient, "
            "household or subject, is not found by this pattern and is not checked."
        ),
        affected_columns=tuple(columns),
        recommendation="Group rows by entity and keep each entity entirely in train or test.",
    )


def small_n_finding(n_rows: int, expected_test_rows: float, severity: Severity) -> Finding:
    """Too few rows to expect a stable train/test split at a conventional ratio."""
    return Finding(
        category="split_strategy",
        severity=severity,
        confidence=1.0,
        title=SMALL_N_TITLE,
        evidence=(
            f"{n_rows} rows. A conventional test split would hold about "
            f"{expected_test_rows:.0f} of them."
        ),
        interpretation=(
            "A small test set makes reported performance noisy: a handful of rows can swing a "
            "metric substantially, and a small training set limits what a model can learn."
        ),
        limitations="The conventional split fraction and the row-count floor are conventions.",
        recommendation=(
            "Prefer cross-validation over a single held-out split, and report a range of scores "
            "rather than a single number."
        ),
    )


def stratified_split_finding(name: str, minority_share: float, severity: Severity) -> Finding:
    """A classification target's minority class is small enough to warrant stratification."""
    return Finding(
        category="split_strategy",
        severity=severity,
        confidence=1.0,
        title=STRATIFIED_TITLE,
        evidence=f"The smallest class holds {minority_share:.1%} of the rows.",
        interpretation=(
            "A plain random split can under- or over-represent the minority class in the test "
            "set by chance, especially with a small test set."
        ),
        limitations="The severity thresholds here are the same conventions used elsewhere.",
        affected_columns=(name,),
        recommendation="Use stratified sampling for train/test splits and cross-validation.",
    )
