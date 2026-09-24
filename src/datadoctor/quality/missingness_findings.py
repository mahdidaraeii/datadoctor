"""The findings that ``check_missingness`` reports, and the wording of each.

Every builder takes plain values, so this module knows nothing about how they were computed.
The cutoffs here are conventions, and the findings that use them say so.
"""

from datadoctor.core.result import Finding, Severity
from datadoctor.quality import dependence as stat

# A rate at or above the first value gets the severity beside it. Below the last, no finding.
GRADES = (
    (0.80, Severity.CRITICAL),
    (0.40, Severity.HIGH),
    (0.20, Severity.MEDIUM),
    (0.05, Severity.LOW),
)
# Correlated missingness from this absolute phi is graded as strong.
STRONG_PHI = 0.8
# NA is only reported as a possible genuine value when this share of the other values in the
# column look like 2 or 3 letter uppercase codes.
CODE_PATTERN = r"[A-Z]{2,3}"
CODE_SHARE = 0.95
# How many names or pairs a finding spells out. The rest are counted.
LISTED = 5

REASONS = {
    "pairwise_skipped": "pairwise analyses were skipped because of the column threshold",
    "all_missing": "every value is missing, so nothing is observed to compare",
    "none_in_sample": "no value is missing in the rows that were analyzed",
    "small_group": "too few missing or observed values in every comparison (under 20 in a group)",
    "small_expected": "expected counts too small in every comparison (under 5)",
    "no_variation": "no variation in the columns compared",
    "too_many_categories": "the only categorical comparisons had too many levels",
    "no_predictors": "there is no other numeric or categorical column to compare against",
}
_WORD = {
    Severity.CRITICAL: "Severe",
    Severity.HIGH: "High",
    Severity.MEDIUM: "Moderate",
    Severity.LOW: "Some",
}


def grade(rate: float) -> Severity | None:
    """The severity of a missing rate, or ``None`` when it is too low to report."""
    for floor, severity in GRADES:
        if rate >= floor:
            return severity
    return None


def plural(count: int, word: str) -> str:
    """``word`` as-is for a count of 1, with a trailing ``s`` otherwise."""
    return word if count == 1 else f"{word}s"


def rate_finding(name: str, count: int, n_rows: int, is_target: bool) -> Finding | None:
    """A finding for one column's missing rate, or ``None`` when it needs none."""
    if count == 0:
        return None
    rate = count / n_rows
    severity = grade(rate)
    evidence = f"Column {name} contains {rate:.1%} missing values ({count:,} of {n_rows:,} rows)."
    if is_target:
        raised = severity is None or severity.rank < Severity.HIGH.rank
        return Finding(
            category="missingness",
            severity=Severity.HIGH if raised else severity,
            confidence=1.0,
            title="Target column has missing values",
            evidence=evidence,
            interpretation=(
                "Rows without a target value cannot be used to train or evaluate a supervised "
                "model, and dropping them can bias the result if they are missing for a reason."
            ),
            limitations=(
                "The rate says how much is missing, not why. The severity is a convention: any "
                "missing target value is treated as at least HIGH."
            ),
            affected_columns=(name,),
            recommendation=(
                "Find out why target values are missing before deciding how to handle those rows."
            ),
        )
    if severity is None:
        return None
    return Finding(
        category="missingness",
        severity=severity,
        confidence=1.0,
        title=f"{_WORD[severity]} missingness in {name}",
        evidence=evidence,
        interpretation=(
            "Dropping these rows or imputing the values can bias estimates if the values are "
            "not missing at random."
        ),
        limitations=(
            "The rate says how much is missing, not why. The severity comes from fixed cutoffs "
            "(5%, 20%, 40%, 80%) that are conventions, not properties of the data."
        ),
        affected_columns=(name,),
        recommendation=(
            "Investigate why values are missing before choosing between dropping, imputing and "
            "modeling the missingness."
        ),
    )


def rows_finding(rows_with_missing: int, n_rows: int, rate: float) -> Finding:
    """The share of rows that have at least one missing value."""
    return Finding(
        category="missingness",
        severity=grade(rate),
        confidence=1.0,
        title="Many rows have at least one missing value",
        evidence=(
            f"{rows_with_missing:,} of {n_rows:,} rows ({rate:.1%}) have at least one missing "
            "value, so dropping every incomplete row would discard that many rows."
        ),
        interpretation=(
            "Complete-case analysis would shrink the data and can bias it if the missing values "
            "are not missing at random."
        ),
        limitations=(
            "Counts every column equally, including columns that may not matter for the analysis. "
            "The severity comes from the same conventional cutoffs as the per-column rates."
        ),
        recommendation=(
            "Decide which columns are needed, then check how many rows are incomplete in those."
        ),
    )


def pairs_finding(
    pairs: list[tuple[str, str, float, int]],
    *,
    above_threshold: int,
    tested: int,
    skipped: int,
    sampled: bool,
) -> Finding:
    """Columns whose missingness is correlated. ``pairs`` are (a, b, phi, rows missing in both)."""
    shown = [
        f"{a} and {b} (phi {phi:.2f}, both missing in {both:,} rows)"
        for a, b, phi, both in pairs[:LISTED]
    ]
    more = above_threshold - len(shown)
    strong = max(abs(phi) for _, _, phi, _ in pairs) >= STRONG_PHI
    verb = "is" if above_threshold == 1 else "are"
    columns = tuple(dict.fromkeys(name for a, b, _, _ in pairs for name in (a, b)))
    return Finding(
        category="missingness",
        severity=Severity.MEDIUM if strong else Severity.LOW,
        confidence=0.9 if strong else 0.7,
        title="Missingness is correlated between columns",
        evidence=(
            f"{above_threshold:,} {plural(above_threshold, 'pair')} of columns {verb} missing "
            f"together more often than chance allows, among {tested:,} pairs tested. "
            "Strongest: " + "; ".join(shown) + (f"; and {more:,} more." if more > 0 else ".")
        ),
        interpretation=(
            "Columns that go missing together usually share a cause, such as one form section, "
            "one source system or one merge, so the missingness is structured and not random "
            "noise."
        ),
        limitations=(
            f"Correlation of missing indicators does not show the cause. {skipped:,} pairs were "
            "not tested because their expected counts were too small. Pairs need |phi| of at "
            f"least {stat.MIN_ABS_PHI} and a Bonferroni-adjusted p-value below {stat.ALPHA}; "
            "both cutoffs are conventions."
            + (" Computed on a sample of the rows." if sampled else "")
        ),
        affected_columns=columns,
        recommendation=(
            "Look for the shared cause and treat these columns as a group when deciding how to "
            "handle missing values."
        ),
    )


def dependence_finding(
    name: str,
    detected: list[stat.Predictor],
    *,
    comparisons: int,
    tests_run: int,
    sampled: bool,
) -> Finding:
    """A column whose missingness depends on other columns. ``detected`` is strongest first."""
    shown = "; ".join(_describe(predictor, name) for predictor in detected[:3])
    strongest = min(predictor.p_adjusted for predictor in detected)
    return Finding(
        category="missingness",
        severity=Severity.MEDIUM,
        confidence=0.8 if strongest < 0.001 else 0.6,
        title=f"Missingness in {name} depends on other columns",
        evidence=(
            f"Whether {name} is missing is associated with the values of {len(detected)} other "
            f"{plural(len(detected), 'column')}, among {comparisons} comparisons. "
            f"Strongest: {shown}."
        ),
        interpretation=(
            "Missingness that depends on observed values is not missing completely at random "
            "(MCAR). It may be missing at random given those columns, or not at random; the "
            "data alone cannot tell these apart. Dropping such rows or filling the values with "
            "a constant can bias estimates."
        ),
        limitations=(
            "Only observed columns were examined, so dependence on the missing values "
            "themselves cannot be detected. Comparisons are Bonferroni-adjusted over "
            f"{tests_run:,} tests, and the effect-size cutoffs (|d| at least {stat.MIN_ABS_D}, "
            f"Cramer's V at least {stat.MIN_CRAMERS_V}) are conventions."
            + (" Computed on a sample of the rows." if sampled else "")
        ),
        affected_columns=(name,),
        recommendation=(
            "Do not treat this column's missingness as random. Consider imputing with the "
            "associated columns, or modeling the missingness, and check with someone who knows "
            "how the data was collected."
        ),
    )


def _describe(predictor: stat.Predictor, column: str) -> str:
    small = predictor.p_adjusted < 1e-15
    p = "adjusted p < 1e-15" if small else f"adjusted p = {predictor.p_adjusted:.2g}"
    if predictor.kind == "numeric":
        direction = "higher" if predictor.effect > 0 else "lower"
        return (
            f"{predictor.name} (mean {abs(predictor.effect):.2f} standard deviations {direction} "
            f"where {column} is missing, {p})"
        )
    return f"{predictor.name} (Cramer's V = {predictor.effect:.2f}, {p})"


def untested_finding(by_reason: dict[str, list[str]], considered: int) -> Finding:
    """Columns that could not be tested, grouped by why, so silence is never read as a result.

    ``considered`` is how many columns had a missing rate high enough to be looked at.
    """
    total = sum(len(columns) for columns in by_reason.values())
    floor = GRADES[-1][0]
    parts = []
    for reason, columns in by_reason.items():
        listed = ", ".join(columns[:LISTED])
        if len(columns) > LISTED:
            listed += f" and {len(columns) - LISTED} more"
        parts.append(f"{len(columns)} ({listed}) because {REASONS[reason]}")
    return Finding(
        category="missingness",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns could not be tested for dependence",
        evidence=(
            f"{total} of {considered} columns with a missing rate of {floor:.0%} or more could "
            "not be tested for whether their missingness depends on other columns: "
            + "; ".join(parts)
            + "."
        ),
        interpretation=(
            "For these columns the analysis says nothing about whether their missingness is "
            "random. A lack of findings for them is not evidence that it is."
        ),
        limitations=(
            "Only comparisons that met the minimum group and count sizes were run. Too few "
            "missing values, or too few observed ones, are the usual cause. Columns with a lower "
            "missing rate are not listed here; their labels are in the metrics."
        ),
        affected_columns=tuple(name for columns in by_reason.values() for name in columns),
        recommendation=(
            "Treat the missingness mechanism of these columns as unknown, or examine them directly."
        ),
    )


def token_finding(name: str, count: int) -> Finding:
    """A token read as missing that may be a genuine 2 or 3 letter code."""
    return Finding(
        category="missingness",
        severity=Severity.MEDIUM,
        confidence=0.6,
        title="Literal token read as missing",
        evidence=(
            f'Column {name} contained "NA" in {count:,} rows, which was read as a missing value. '
            "At least 95% of the other values in the column are 2 or 3 letter uppercase codes, "
            "and NA is itself a valid 2 letter code (for example Namibia's country code)."
        ),
        interpretation=(
            "These cells may be a genuine value that was mistaken for a missing-value marker. "
            "This is a heuristic based on the token and the format of the column."
        ),
        limitations=(
            "The loader cannot tell a genuine NA from a marker for missing data. If NA marks "
            "missing values in this file, there is nothing to correct."
        ),
        affected_columns=(name,),
        recommendation=(
            "Check whether these cells are genuine values. Until they are read as values, every "
            "analysis treats them as missing."
        ),
    )
