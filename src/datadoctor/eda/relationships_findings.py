"""The findings that ``check_relationships`` reports, and the wording of each.

Every builder takes plain values, so this module knows nothing about how they were computed.
The cutoffs here are conventions, and the findings that use them say so.
"""

from datadoctor.core.result import Finding, Severity

# How many pairs or names a finding spells out. The rest are counted, and all are in the metrics.
LISTED = 5

_NOT_USED_REASONS = {
    "identifier": "identifiers, where every value is different",
    "text": "free text",
    "datetime": "dates and times",
    "unknown": "columns with mixed, nested or no values",
}
_NOT_TESTED_REASONS = {
    "too_many_categories": "too many distinct values to compare",
    "not_enough_data": "too few rows with both values present, or no variation to compare",
}
_TARGET_UNUSABLE_REASONS = {
    "identifier": "an identifier, where every value is different",
    "text": "free text",
    "datetime": "a date or time",
    "unknown": "a column of mixed, nested or no values",
}
_KIND_WORDING = {
    "pearson": "Pearson correlation",
    "eta_squared": "correlation ratio",
    "cramers_v": "Cramer's V",
}


def _more(total: int, shown: int) -> str:
    return f" and {total - shown} more" if total > shown else ""


def _listed(names: list[str]) -> str:
    text = ", ".join(names[:LISTED])
    return text + (f" and {len(names) - LISTED} more" if len(names) > LISTED else "")


def correlated_columns_finding(pairs: list[tuple[str, str, float]], threshold: float) -> Finding:
    """Column pairs whose Pearson correlation reaches ``threshold``, a candidate for redundancy."""
    shown = [f"{a} / {b} (r={r:.2f})" for a, b, r in pairs[:LISTED]]
    names = sorted({name for pair in pairs for name in pair[:2]})
    return Finding(
        category="eda",
        severity=Severity.LOW,
        confidence=1.0,
        title="Numeric columns are strongly correlated",
        evidence=(
            f"{len(pairs)} {'pair' if len(pairs) == 1 else 'pairs'} of columns have an absolute "
            f"Pearson correlation of at least {threshold:.1f}: "
            + "; ".join(shown)
            + _more(len(pairs), len(shown))
            + "."
        ),
        interpretation=(
            "Two strongly correlated columns carry much of the same information. Keeping both "
            "in a model rarely helps and can make coefficients unstable or hard to interpret."
        ),
        limitations=(
            f"The {threshold:.1f} cutoff is a convention. Pearson correlation only measures a "
            "straight-line relationship, so a strong non-linear relationship can be missed."
        ),
        affected_columns=tuple(names),
        recommendation=(
            "Consider dropping or combining one column from each pair, or using a model that "
            "handles correlated features well."
        ),
    )


def top_predictors_finding(associations: list[dict]) -> Finding:
    """The features whose association with the target is strongest, most to least."""
    shown = [
        f"{a['name']} ({_KIND_WORDING[a['kind']]}={a['effect']:.2f})" for a in associations[:LISTED]
    ]
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Columns most associated with the target",
        evidence=(
            f"{len(associations)} of the tested "
            f"{'column is' if len(associations) == 1 else 'columns are'} associated with the "
            "target beyond a small effect: "
            + "; ".join(shown)
            + _more(len(associations), len(shown))
            + "."
        ),
        interpretation=(
            "A numeric feature and the target are compared by Pearson correlation. A categorical "
            "feature or target is compared by the correlation ratio (a numeric feature grouped by "
            "a categorical one) or Cramer's V (two categorical columns), both on the same 0 to 1 "
            "scale as an absolute correlation. This is association, not causation, and says "
            "nothing about how a feature would behave in a model alongside the others."
        ),
        limitations=(
            "Only pairs beyond a small effect size and significant after correcting for the "
            "number of features tested are listed. A column left out was not shown to be "
            "unrelated to the target, only not detected here."
        ),
        affected_columns=tuple(a["name"] for a in associations),
    )


def class_imbalance_finding(
    label: str | None, minority_share: float, severity: Severity
) -> Finding:
    """The minority class of a classification target holds a small share of the rows."""
    named = f"'{label}' " if label is not None else ""
    return Finding(
        category="eda",
        severity=severity,
        confidence=1.0,
        title="The target classes are imbalanced",
        evidence=f"The smallest class {named}holds {minority_share:.1%} of the rows.",
        interpretation=(
            "Imbalance is expected for many real classification problems, such as churn, fraud "
            "or disease detection, and is not a defect by itself. It is a signal to use "
            "stratified splitting when evaluating a model, and a metric that is not misled by "
            "the majority class, such as precision, recall or a class-weighted score, rather "
            "than plain accuracy."
        ),
        limitations="The severity thresholds here are conventions, not a judgment on this data.",
        recommendation=(
            "Use stratified sampling for train/test splits and cross-validation, and choose an "
            "evaluation metric that accounts for the imbalance."
        ),
    )


def not_used_finding(by_reason: dict[str, list[str]]) -> Finding:
    """Columns left out of the correlation matrix and the target comparison entirely."""
    total = sum(len(names) for names in by_reason.values())
    parts = [
        f"{_NOT_USED_REASONS[reason]} ({_listed(names)})" for reason, names in by_reason.items()
    ]
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns were not compared",
        evidence=f"{total} {'column was' if total == 1 else 'columns were'} not compared: "
        + "; ".join(parts)
        + ".",
        interpretation=(
            "A correlation or association needs numbers or a small set of categories, so these "
            "columns are left out of the correlation matrix and out of the target comparison."
        ),
        limitations="Nothing here says whether these columns relate to the others or the target.",
    )


def not_tested_finding(by_reason: dict[str, list[str]]) -> Finding:
    """Numeric or categorical columns that were not compared against the target."""
    total = sum(len(names) for names in by_reason.values())
    parts = [
        f"{_NOT_TESTED_REASONS[reason]} ({_listed(names)})" for reason, names in by_reason.items()
    ]
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns could not be compared with the target",
        evidence=(
            f"{total} {'column was' if total == 1 else 'columns were'} not compared with the "
            "target: " + "; ".join(parts) + "."
        ),
        interpretation=(
            "A lack of finding for these columns says nothing about whether they relate to the "
            "target; the comparison was not possible, not negative."
        ),
        limitations=(
            "Categorical columns with more than a conventional number of levels are skipped."
        ),
    )


def target_unusable_finding(name: str, kind: str) -> Finding:
    """The target column's type is not suited to correlation or association testing here."""
    return Finding(
        category="eda",
        severity=Severity.INFO,
        confidence=1.0,
        title="The target could not be compared against the other columns",
        evidence=(
            f"Column {name} is {_TARGET_UNUSABLE_REASONS[kind]}, so it was not used as a target "
            "here."
        ),
        interpretation=(
            "Correlation and association testing here work with numbers and a small set of "
            "categories, which this column's type does not fit."
        ),
        limitations="The correlation matrix of the other columns is not affected by this.",
        affected_columns=(name,),
    )
