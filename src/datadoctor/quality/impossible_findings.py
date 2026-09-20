"""The findings that ``check_impossible_values`` reports, and the wording of each.

Findings give counts and never the offending values; the lowest and highest are in the metrics.
"""

from datadoctor.core.result import Finding, Severity

# From this share of a column's values, an impossible value is HIGH: it is systematic, not a slip.
HIGH_SHARE = 0.05


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _grade(rate: float) -> Severity:
    return Severity.HIGH if rate >= HIGH_SHARE else Severity.MEDIUM


def for_check(check: dict) -> list[Finding]:
    """The findings for one checked column, empty when nothing was found."""
    if check["rule"] == "age":
        found = []
        if check["negative"]:
            found.append(_negative_age(check))
        if check["above_maximum"]:
            found.append(_high_age(check))
        return found
    if check["rule"] == "non_negative":
        return [_negative_quantity(check)] if check["negative"] else []
    found = []
    if check["invalid"]:
        found.append(_invalid_dates(check))
    if check["future"]:
        found.append(_future_dates(check))
    return found


def _share(count: int, check: dict) -> str:
    checked = check["checked"]
    return f"{count:,} {_plural(count, 'value')} ({count / checked:.1%} of {checked:,})"


def _negative_age(check: dict) -> Finding:
    name = check["name"]
    return Finding(
        category="impossible_values",
        severity=_grade(check["rate"]),
        confidence=0.9,
        title=f"Negative values in age column {name}",
        evidence=f"Column {name} has {_share(check['negative'], check)} below zero.",
        interpretation=(
            "An age cannot be negative. Such values are usually entry errors, a code such as -1 "
            "for unknown, or a subtraction the wrong way round. This is a heuristic: it rests "
            "on the column name containing the word age."
        ),
        limitations=(
            "The rule looks at the name only, so a column called age might hold something else. "
            "Values are not shown; the lowest is in the metrics."
        ),
        affected_columns=(name,),
        recommendation=(
            "Find out what the negative values mean. If they are codes for unknown, treat them "
            "as missing. Nothing was changed."
        ),
    )


def _high_age(check: dict) -> Finding:
    name = check["name"]
    return Finding(
        category="impossible_values",
        severity=Severity.LOW,
        confidence=0.6,
        title=f"Implausibly high values in age column {name}",
        evidence=(
            f"Column {name} has {_share(check['above_maximum'], check)} above 120. The oldest "
            "verified human lived to 122."
        ),
        interpretation=(
            "Ages this high are implausible, and are usually entry errors, a different unit "
            "such as months, or a placeholder. This is a heuristic based on the column name and "
            "on a fixed bound of 120."
        ),
        limitations=(
            "The bound of 120 is a convention. Historical or fictional data can legitimately go "
            "higher. Values are not shown; the highest is in the metrics."
        ),
        affected_columns=(name,),
        recommendation="Check the unit and the source of these values. Nothing was changed.",
    )


def _negative_quantity(check: dict) -> Finding:
    name = check["name"]
    return Finding(
        category="impossible_values",
        severity=_grade(check["rate"]),
        confidence=0.7,
        title=f"Negative values in {name}",
        evidence=f"Column {name} has {_share(check['negative'], check)} below zero.",
        interpretation=(
            "A length, weight, duration, distance, population or count cannot be negative. "
            "Such values are usually entry errors or a code for unknown. This is a heuristic: "
            "it rests on the column name."
        ),
        limitations=(
            "The rule looks at the name only. A column with such a name can legitimately hold "
            "negative values, for example a change or a difference. Values are not shown; the "
            "lowest is in the metrics."
        ),
        affected_columns=(name,),
        recommendation=(
            "Check what the column measures. If the negative values are codes for unknown, "
            "treat them as missing. Nothing was changed."
        ),
    )


def _invalid_dates(check: dict) -> Finding:
    name = check["name"]
    return Finding(
        category="impossible_values",
        severity=Severity.HIGH if check["invalid_rate"] >= HIGH_SHARE else Severity.MEDIUM,
        confidence=0.9,
        title=f"Dates that do not exist in {name}",
        evidence=(
            f"Column {name} has {_share(check['invalid'], check)} shaped like dates "
            "(YYYY-MM-DD) that are not real calendar dates, such as a 30th of February or a "
            "13th month. The other values are valid dates, and were checked for the future."
        ),
        interpretation=(
            "A date that does not exist is an entry error or a corrupted value. The rest of the "
            "column reads as ISO dates, so this is unlikely to be free text."
        ),
        limitations=(
            "Only text in ISO format (YYYY-MM-DD, optionally followed by a time) is checked, and "
            "the time part is not validated. Values are not shown."
        ),
        affected_columns=(name,),
        recommendation=(
            "Find where these values come from and correct or remove them at the source. "
            "Nothing was changed."
        ),
    )


def _future_dates(check: dict) -> Finding:
    name = check["name"]
    return Finding(
        category="impossible_values",
        severity=_grade(check["rate"]),
        confidence=0.5,
        title=f"Dates in the future in {name}",
        evidence=(
            f"Column {name} has {_share(check['future'], check)} later than {check['as_of']}."
        ),
        interpretation=(
            "For events that already happened, a date in the future is impossible. These are "
            "usually clock errors, wrong date formats or placeholder dates. This is a heuristic: "
            "it does not know whether the column is meant to hold future dates."
        ),
        limitations=(
            f"Dates are compared as calendar dates, ignoring the time of day, and timezone-aware "
            f"values are converted to UTC. The reference date is {check['as_of']}, which is the "
            "date the data was loaded unless another was given. Columns with names that suggest "
            "future dates, such as due or expiry, are not checked. Values are not shown; the "
            "latest date is in the metrics."
        ),
        affected_columns=(name,),
        recommendation=(
            "Check whether this column should ever hold future dates. If not, find out where "
            "the later dates come from. Nothing was changed."
        ),
    )
