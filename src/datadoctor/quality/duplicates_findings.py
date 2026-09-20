"""The findings that ``check_duplicates`` reports, and the wording of each.

Values are never printed, only counts and row positions counted from 0.
"""

from datadoctor.core.result import Finding, Severity

# How many row positions a finding spells out. The rest are counted.
LISTED = 5
# Rows: the share of rows that are extra copies, lower bound inclusive. Any repeat is at least LOW.
_ROW_GRADES = (
    (0.50, Severity.CRITICAL),
    (0.10, Severity.HIGH),
    (0.01, Severity.MEDIUM),
)
# Identifier columns: from this share of rows a repeated id is HIGH, and CRITICAL if any repeated
# id has conflicting rows.
IDENTIFIER_SHARE = 0.01


def grade(rate: float) -> Severity:
    """The severity of a share of rows that are extra copies of earlier rows."""
    for floor, severity in _ROW_GRADES:
        if rate >= floor:
            return severity
    return Severity.LOW


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _positions(positions: list[int]) -> str:
    return ", ".join(str(position) for position in positions)


def _left_out_sentence(left_out: list[str]) -> str:
    if not left_out:
        return ""
    return (
        f" Columns {', '.join(left_out)} were left out because their values cannot be compared, "
        "so rows that differ only in them count as duplicates."
    )


def rows_finding(duplicates, compared: int, left_out: list[str]) -> Finding:
    """Rows that repeat an earlier row, identical in every compared column."""
    d = duplicates
    return Finding(
        category="duplicates",
        severity=grade(d.rate),
        confidence=1.0,
        title="Duplicate rows",
        evidence=(
            f"{d.extra:,} rows repeat an earlier row ({d.rate:.1%} of all rows), in {d.groups:,} "
            f"{_plural(d.groups, 'group')} of identical rows. The largest group has "
            f"{d.largest_group:,} rows. First repeats at row positions "
            f"{_positions(d.first_positions)} (counted from 0)."
        ),
        interpretation=(
            "Repeated rows count the same record more than once in every statistic, and they can "
            "leak between training and test data. They are not necessarily errors: a table with "
            "only a few categorical columns, or aggregated data, repeats rows by nature."
        ),
        limitations=(
            f"{compared} columns were compared, and rows count as identical only if every value "
            "matches exactly, with all missing values equal to each other. Near-duplicates, such "
            "as differences in case or spacing, are not found." + _left_out_sentence(left_out)
        ),
        recommendation=(
            "Check whether the repeats are real repeated events. If they are not, remove the "
            "extra copies before splitting the data."
        ),
    )


def hidden_finding(
    duplicates, ignored: list[str], confidence: float, left_out: list[str]
) -> Finding:
    """Rows that repeat once identifier columns are ignored, which a plain comparison cannot see."""
    d = duplicates
    columns = ", ".join(ignored)
    return Finding(
        category="duplicates",
        severity=grade(d.rate),
        confidence=confidence,
        title="Duplicate rows hidden by identifier columns",
        evidence=(
            f"Ignoring {_plural(len(ignored), 'the identifier column')} {columns}, "
            f"{d.extra:,} rows repeat an earlier row ({d.rate:.1%} of all rows), in {d.groups:,} "
            f"{_plural(d.groups, 'group')}. The largest group has {d.largest_group:,} rows. First "
            f"repeats at row positions {_positions(d.first_positions)} (counted from 0)."
        ),
        interpretation=(
            "Every row has a different identifier, so an exact comparison of rows finds nothing, "
            "yet the rest of the record is repeated. These are probably the same records entered "
            "more than once, or copied rows given a new id. This rests on those columns really "
            "being identifiers, which is a heuristic."
        ),
        limitations=(
            f"The identifier columns were detected by a heuristic, and this finding is only as "
            f"sure as the least certain of them (confidence {confidence:.2f}). If one of them is "
            "really a measurement, the repeats may be false. Near-duplicates are not found."
            + _left_out_sentence(left_out)
        ),
        recommendation=(
            "Check whether the repeated records are genuine. If not, remove the extra copies "
            "before splitting the data or joining on the id."
        ),
    )


def identifier_finding(duplicates) -> Finding:
    """Values that repeat in a column detected as an identifier."""
    d = duplicates
    if d.rate >= IDENTIFIER_SHARE and d.conflicting_groups:
        severity = Severity.CRITICAL
    elif d.rate >= IDENTIFIER_SHARE:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM

    parts = []
    if d.exact_copy_groups:
        parts.append(
            "Exact copies are a duplication problem: the same record appears more than once."
        )
    if d.conflicting_groups:
        parts.append(
            "The same id with different data is an integrity problem: at least one of those rows "
            "is wrong, or the column is not really unique."
        )
    recommendation = (
        "Find out which of the conflicting rows is correct before using this column as a key or "
        "joining on it."
        if d.conflicting_groups
        else "Remove the extra copies before using this column as a key or joining on it."
    )
    return Finding(
        category="duplicates",
        severity=severity,
        confidence=d.confidence,
        title=f"Repeated values in identifier column {d.name}",
        evidence=(
            f"Column {d.name} has {d.duplicated_values:,} values that appear more than once, "
            f"{d.extra_rows:,} extra rows in all ({d.rate:.1%} of all rows). In "
            f"{d.exact_copy_groups:,} of these groups the rows are exact copies, and in "
            f"{d.conflicting_groups:,} the rows differ in other columns. First repeats at row "
            f"positions {_positions(d.first_positions)} (counted from 0). Values are not shown."
        ),
        interpretation=" ".join(parts),
        limitations=(
            f"{d.name} was detected as an identifier by a heuristic (confidence "
            f"{d.confidence:.2f}). Ids that differ only in case or spacing are not matched, and "
            "composite keys spread over several columns are not checked."
        ),
        affected_columns=(d.name,),
        recommendation=recommendation,
    )


def left_out_finding(left_out: list[str], compared: int) -> Finding:
    """Columns whose values cannot be compared."""
    names = ", ".join(left_out)
    verb = "holds" if len(left_out) == 1 else "hold"
    evidence = (
        f"{len(left_out)} {_plural(len(left_out), 'column')} {verb} values that cannot be "
        f"compared, such as lists or dicts inside cells: {names}. "
    )
    evidence += (
        f"They were left out, and the other {compared} columns were compared."
        if compared
        else "No column could be compared, so duplicate rows were not looked for."
    )
    return Finding(
        category="duplicates",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns could not be compared",
        evidence=evidence,
        interpretation=(
            "Rows that differ only in these columns look identical here, so duplicate counts can "
            "be too high, and a lack of duplicates cannot be trusted for these columns."
        ),
        limitations="Nested values are compared as a whole by no rule, so they are skipped.",
        affected_columns=tuple(left_out),
        recommendation=(
            "Flatten the nested values into ordinary columns if they should count in the "
            "comparison."
        ),
    )
