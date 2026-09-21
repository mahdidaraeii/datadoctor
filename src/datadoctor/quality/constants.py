"""Constant and near-constant columns."""

import numpy as np

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.quality.codes import MISSING, column_codes

# A column is near-constant when its most common value covers at least this share of its
# non-null values (and it is not constant). A convention, not a property of the data.
NEAR_CONSTANT_SHARE = 0.99
# How many column names a finding spells out. The rest are counted, and all are in the metrics.
LISTED = 5


def check_constants(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Find columns that hold one value, or nearly one value.

    Values are counted among the non-null values, so missing values stay with the missingness
    diagnostics: a column with one value and many missing cells is constant among the values it
    has, and its finding says how many are missing. A column with no values at all is skipped,
    since the missingness diagnostics already report it. All missing values are equal to each
    other, and ``0.0`` equals ``-0.0``.

    Every row is used, so the size guardrails of ``config`` do not apply. ``config`` is recorded
    in the result.

    ``metrics`` holds one entry per column with its ``kind``: ``constant``, ``near_constant``,
    ``varying``, ``all_null`` or ``not_checked`` (values that cannot be compared, such as lists),
    with the number of non-null values, the number of distinct values and the share of the most
    common one.

    Findings:

    - a constant target column is CRITICAL, since there is nothing to learn or evaluate;
    - one LOW finding for all other constant columns;
    - one INFO finding for all other near-constant columns. A highly skewed column can be
      legitimate, so it is not treated as a defect. A near-constant target is not flagged, since a
      rare class is class imbalance, which is a different question;
    - one INFO finding for columns that could not be checked.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for constant columns in a dataset with no rows")

    names = [str(label) for label in frame.columns]
    target = None if dataset.target is None else list(frame.columns).index(dataset.target)
    columns = [_describe(frame.iloc[:, i], names[i], i == target) for i in range(n_columns)]

    def of_kind(kind: str, target_too: bool = False) -> list[dict]:
        return [c for c in columns if c["kind"] == kind and (target_too or not c["is_target"])]

    findings: list[Finding] = []
    constant_target = next((c for c in columns if c["is_target"] and c["kind"] == "constant"), None)
    if constant_target is not None:
        findings.append(_constant_target_finding(constant_target, n_rows))
    if of_kind("constant"):
        findings.append(_constants_finding(of_kind("constant")))
    if of_kind("near_constant"):
        findings.append(_near_constants_finding(of_kind("near_constant")))
    if of_kind("not_checked", target_too=True):
        findings.append(_not_checked_finding(of_kind("not_checked", target_too=True)))

    metrics = {"n_rows": n_rows, "n_columns": n_columns, "columns": columns}
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _describe(series, name: str, is_target: bool) -> dict:
    codes = column_codes(series)
    entry = {
        "name": name,
        "kind": "not_checked",
        "non_null": None,
        "distinct": None,
        "top_share": None,
        "missing": int(series.isna().sum()),
        "is_target": is_target,
    }
    if codes is None:
        return entry

    present = codes[codes != MISSING]
    entry["non_null"] = int(present.size)
    if present.size == 0:
        entry["kind"] = "all_null"
        return entry
    counts = np.bincount(present)
    distinct = int((counts > 0).sum())
    share = float(counts.max() / present.size)
    entry["distinct"] = distinct
    entry["top_share"] = share
    if distinct == 1:
        entry["kind"] = "constant"
    elif share >= NEAR_CONSTANT_SHARE:
        entry["kind"] = "near_constant"
    else:
        entry["kind"] = "varying"
    return entry


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _verb(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _listed(names: list[str]) -> str:
    text = ", ".join(names[:LISTED])
    if len(names) > LISTED:
        text += f" and {len(names) - LISTED} more"
    return text


def _constant_target_finding(column: dict, n_rows: int) -> Finding:
    name = column["name"]
    return Finding(
        category="constants",
        severity=Severity.CRITICAL,
        confidence=1.0,
        title="Target column is constant",
        evidence=(
            f"Column {name} has a single value in all {column['non_null']:,} of its non-null "
            f"values ({column['missing']:,} of {n_rows:,} rows are missing)."
        ),
        interpretation=(
            "A target that never varies gives a supervised model nothing to learn and nothing to "
            "be evaluated against."
        ),
        limitations=(
            "Only the values that are present were counted. If the target has many missing "
            "values, the rows that would show the other values may simply be missing."
        ),
        affected_columns=(name,),
        recommendation=(
            "Check that this is the right column, and that no filter or join removed the other "
            "outcomes."
        ),
    )


def _constants_finding(columns: list[dict]) -> Finding:
    names = [c["name"] for c in columns]
    with_missing = [c for c in columns if c["missing"]]
    verb = "has" if len(with_missing) == 1 else "have"
    missing_note = (
        f" {len(with_missing)} of them also {verb} missing values, "
        "so they are constant only among the values they have."
        if with_missing
        else ""
    )
    return Finding(
        category="constants",
        severity=Severity.LOW,
        confidence=1.0,
        title="Constant columns",
        evidence=(
            f"{len(names)} {_plural(len(names), 'column')} "
            f"{_verb(len(names), 'holds', 'hold')} a single value: {_listed(names)}." + missing_note
        ),
        interpretation=(
            "A column with one value cannot tell rows apart, so it carries no information for "
            "modeling. It can also break methods that divide by the spread of a column, such as "
            "standardization."
        ),
        limitations=(
            "Only non-null values were compared, and all missing values are treated as equal. A "
            "column that is constant here may vary in data that was not loaded."
        ),
        affected_columns=tuple(names),
        recommendation="Drop these columns unless they are needed for another reason.",
    )


def _near_constants_finding(columns: list[dict]) -> Finding:
    names = [c["name"] for c in columns]
    shares = [f"{c['name']} ({c['top_share']:.1%})" for c in columns[:LISTED]]
    more = len(columns) - len(shares)
    listed = ", ".join(shares) + (f" and {more} more" if more > 0 else "")
    return Finding(
        category="constants",
        severity=Severity.INFO,
        confidence=1.0,
        title="Near-constant columns",
        evidence=(
            f"{len(names)} {_plural(len(names), 'column')} "
            f"{_verb(len(names), 'has', 'have')} one value in at least "
            f"{NEAR_CONSTANT_SHARE:.0%} of {_verb(len(names), 'its', 'their')} non-null values. "
            f"Share of the most common value: {listed}."
        ),
        interpretation=(
            "A highly skewed column can be legitimate and is not a defect by itself. Zero-inflated "
            "counts and rare-event flags are common, and the rare values can carry most of the "
            "signal. Such a column mainly carries information in the few rows that differ."
        ),
        limitations=(
            f"The {NEAR_CONSTANT_SHARE:.0%} cutoff is a convention. Whether the rare values "
            "matter depends on how they relate to what is being predicted, which this check does "
            "not measure."
        ),
        affected_columns=tuple(names),
        recommendation=(
            "Check whether the rare values matter before dropping a column, and keep it if they do."
        ),
    )


def _not_checked_finding(columns: list[dict]) -> Finding:
    names = [c["name"] for c in columns]
    return Finding(
        category="constants",
        severity=Severity.INFO,
        confidence=1.0,
        title="Some columns could not be checked",
        evidence=(
            f"{len(names)} {_plural(len(names), 'column')} "
            f"{_verb(len(names), 'holds', 'hold')} values that cannot be compared, "
            f"such as lists or dicts inside cells: {_listed(names)}."
        ),
        interpretation=(
            "Nothing is known about whether these columns are constant, so a lack of findings for "
            "them says nothing."
        ),
        limitations="Nested values are not compared, so these columns were skipped.",
        affected_columns=tuple(names),
        recommendation=(
            "Flatten the nested values into ordinary columns if they should be checked."
        ),
    )
