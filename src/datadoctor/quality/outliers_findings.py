"""The findings that ``check_outliers`` reports, and the wording of each.

Values are never shown. A finding gives counts, quartiles and fences; the lowest and highest values
stay in the metrics, because findings are what gets shared.
"""

from datadoctor.core.result import Finding, Severity

# How many columns a finding spells out. The rest are counted, and all are in the metrics.
LISTED = 5


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _has(count: int) -> str:
    return "has" if count == 1 else "have"


def _num(value: float) -> str:
    return f"{value:,.5g}"


def _more(total: int, shown: int) -> str:
    return f" and {total - shown} more" if total > shown else ""


def _infinite_note(infinite: list[dict]) -> str:
    if not infinite:
        return ""
    return (
        f" Infinite values in {len(infinite)} {_plural(len(infinite), 'column')} were left out of "
        "the statistics."
    )


def _limitations(infinite: list[dict]) -> str:
    return (
        "The fences assume little about the shape of the data. The z-score assumes it is roughly "
        "bell-shaped and is pulled by the outliers it looks for. In skewed data both flag the "
        "long tail of a legitimate distribution. The cutoffs (1.5 and 3 interquartile ranges, "
        "|z| of 3) are conventions. Each column is examined alone, so a value that is unusual only "
        "in combination with others is not found. Values are not shown here; the lowest and "
        "highest are in the metrics." + _infinite_note(infinite)
    )


def extreme_finding(columns: list[dict], infinite: list[dict]) -> Finding:
    """Columns with values beyond the wide fences and beyond 3 standard deviations."""
    ranked = sorted(columns, key=lambda c: (-c["extreme"], c["name"]))
    shown = [
        f"{c['name']} ({c['extreme']:,} {_plural(c['extreme'], 'value')}; quartiles "
        f"{_num(c['q1'])} and {_num(c['q3'])}, wide fences {_num(c['extreme_low'])} to "
        f"{_num(c['extreme_high'])})"
        for c in ranked[:LISTED]
    ]
    return Finding(
        category="outliers",
        severity=Severity.MEDIUM,
        confidence=0.7,
        title="Extreme values in numeric columns",
        evidence=(
            f"{len(columns)} {_plural(len(columns), 'column')} {_has(len(columns))} values more "
            "than 3 interquartile ranges beyond the quartiles and more than 3 standard "
            "deviations from the mean: " + "; ".join(shown) + _more(len(columns), len(shown)) + "."
        ),
        interpretation=(
            "Values this far from the rest are candidates for data errors, such as a wrong unit, "
            "a typo or a faulty sensor, or for real but rare events. They are candidates, not "
            "errors: one column alone cannot tell which."
        ),
        limitations=_limitations(infinite),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Look at the rows behind these values and decide whether each is an error or a real "
            "case. Do not remove them automatically."
        ),
    )


def candidates_finding(columns: list[dict], infinite: list[dict]) -> Finding:
    """Columns with milder candidates only."""
    ranked = sorted(columns, key=lambda c: (-c["rate"], c["name"]))
    shown = [
        f"{c['name']} ({c['flagged']:,} {_plural(c['flagged'], 'value')}, {c['rate']:.1%}: "
        f"{c['iqr_flagged']:,} by the IQR rule, {c['z_flagged']:,} by z-score, "
        f"{c['both_flagged']:,} by both; quartiles {_num(c['q1'])} and {_num(c['q3'])}, fences "
        f"{_num(c['mild_low'])} to {_num(c['mild_high'])})"
        for c in ranked[:LISTED]
    ]
    return Finding(
        category="outliers",
        severity=Severity.LOW,
        confidence=0.4,
        title="Outlier candidates in numeric columns",
        evidence=(
            f"{len(columns)} {_plural(len(columns), 'column')} {_has(len(columns))} values outside "
            "1.5 interquartile ranges from the quartiles or more than 3 standard deviations "
            "from the mean: " + "; ".join(shown) + _more(len(columns), len(shown)) + "."
        ),
        interpretation=(
            "These values are unusual for their column, but mild candidates are common in real "
            "data. When many values are flagged, the column is usually skewed or heavy-tailed, "
            "which is a property of the distribution and not a fault. They are candidates, not "
            "errors."
        ),
        limitations=_limitations(infinite),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Check the columns with the highest rates first. Consider a transformation, such as "
            "a logarithm, for skewed columns before treating the tail as errors."
        ),
    )


_REASON = {
    "too_few_values": "fewer than 30 finite values",
    "zero_iqr": "an interquartile range of zero, so at least half of the values are identical",
}


def skipped_finding(columns: list[dict], numeric: int) -> Finding:
    """Numeric columns whose outliers could not be assessed, so silence is never read as clean."""
    by_reason: dict[str, list[str]] = {}
    for column in columns:
        by_reason.setdefault(column["reason"], []).append(column["name"])
    parts = []
    for reason, names in by_reason.items():
        listed = ", ".join(names[:LISTED]) + _more(len(names), LISTED)
        parts.append(f"{len(names)} ({listed}) {_has(len(names))} {_REASON[reason]}")
    return Finding(
        category="outliers",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some numeric columns could not be checked for outliers",
        evidence=(
            f"{len(columns)} of {numeric} numeric columns, not counting flags and ratings with "
            "fewer than 10 distinct values, could not be checked: " + "; ".join(parts) + "."
        ),
        interpretation=(
            "For these columns the analysis says nothing about outliers. A lack of findings for "
            "them is not evidence that there are none."
        ),
        limitations=(
            "The fence and z-score rules give unreliable answers below 30 values or when the "
            "middle half of the values is identical, so they are not applied there. Columns "
            "with fewer than 10 distinct values are not listed, since they are flags or "
            "ratings and not measurements."
        ),
        affected_columns=tuple(name for names in by_reason.values() for name in names),
        recommendation="Inspect these columns directly if outliers matter for them.",
    )


def infinite_finding(columns: list[dict]) -> Finding:
    """Columns holding infinite values, which were left out of every statistic."""
    shown = [f"{c['name']} ({c['n_infinite']:,})" for c in columns[:LISTED]]
    return Finding(
        category="outliers",
        severity=Severity.INFO,
        confidence=1.0,
        title="Infinite values were left out of the outlier statistics",
        evidence=(
            f"{len(columns)} numeric {_plural(len(columns), 'column')} "
            f"{'holds' if len(columns) == 1 else 'hold'} infinite values: "
            + ", ".join(shown)
            + _more(len(columns), len(shown))
            + "."
        ),
        interpretation=(
            "Infinite values are usually the result of a division by zero or an overflow, and "
            "they would make every statistic undefined, so they were not used."
        ),
        limitations="The counts are of values equal to positive or negative infinity.",
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Find out where the infinite values come from before analyzing these columns."
        ),
    )
