"""The notes that ``check_univariate`` reports about what it did not draw, and why.

They are INFO findings: facts about the run, not defects in the data.
"""

from datadoctor.core.result import Finding, Severity

# How many columns a note spells out per reason. All are in the metrics.
LISTED = 5

_REASONS = {
    "identifier": "identifiers, where every value is different",
    "text": "free text",
    "datetime": "dates and times",
    "unknown": "columns with mixed, nested or no values",
    "no_finite_values": "numeric columns with no finite value",
}


def _listed(names: list[str]) -> str:
    text = ", ".join(names[:LISTED])
    return text + (f" and {len(names) - LISTED} more" if len(names) > LISTED else "")


def not_plotted_finding(by_reason: dict[str, list[str]]) -> Finding:
    """Columns that get statistics but no plot, because a chart would say nothing."""
    total = sum(len(names) for names in by_reason.values())
    parts = [f"{_REASONS[reason]} ({_listed(names)})" for reason, names in by_reason.items()]
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns were not plotted",
        evidence=f"{total} {'column was' if total == 1 else 'columns were'} not plotted: "
        + "; ".join(parts)
        + ".",
        interpretation=(
            "A chart of values that are all different, of free text or of dates would show "
            "nothing useful in this kind of plot, so these columns are left out of the figures."
        ),
        limitations=(
            "Their summary statistics are still in the metrics. Distributions of dates over "
            "time are not drawn here."
        ),
    )


def page_limit_finding(not_drawn: dict[str, list[str]], limit: int) -> Finding:
    """Columns left out because the figures for their kind are already full."""
    total = sum(len(names) for names in not_drawn.values())
    parts = [f"{kind} ({_listed(names)})" for kind, names in not_drawn.items()]
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns did not fit on the figures",
        evidence=(
            f"Each kind of column is drawn on at most {limit} figures, and {total} "
            f"{'column' if total == 1 else 'columns'} did not fit: " + "; ".join(parts) + "."
        ),
        interpretation=(
            "Very wide datasets would need many figures, so the first columns of each kind, in "
            "file order, are drawn and the rest are only in the metrics."
        ),
        limitations="Their summary statistics are still in the metrics.",
        recommendation="Select the columns of interest and run the analysis on those.",
    )


def labels_hidden_finding(names: list[str]) -> Finding:
    """Categorical columns whose values were kept out of the figures on privacy grounds."""
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Category labels were hidden for columns that may hold personal data",
        evidence=(
            f"{len(names)} categorical {'column' if len(names) == 1 else 'columns'}: "
            f"{_listed(names)}. Their bars and counts are drawn, but the values are not named."
        ),
        interpretation=(
            "The privacy check flagged these columns as possibly holding personal data, and a "
            "frequency chart would write those values into an image file."
        ),
        limitations=(
            "The privacy check is a heuristic, so a flagged column may hold no personal data, "
            "and a column it missed is drawn with its labels."
        ),
        recommendation=(
            "Run the privacy check for the details, and draw the labels yourself if the values "
            "are not personal data."
        ),
    )
