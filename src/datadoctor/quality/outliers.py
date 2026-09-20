"""Outlier candidates in numeric columns, by the IQR and z-score rules.

These are candidates, never errors: a value far from the rest may be a mistake or a real, rare
event, and one column at a time cannot tell which.
"""

import numpy as np

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.result import AnalysisResult, Finding
from datadoctor.profiling.schema import profile_schema
from datadoctor.quality import outliers_findings as report

# A column needs this many finite values. Below about ten, no value can reach |z| > 3 at all,
# because |z| is bounded by (n - 1) / sqrt(n), and small samples give unstable quartiles.
MIN_VALUES = 30
# A column with fewer distinct values is a flag, a rating or a small count, not a measurement.
MIN_DISTINCT = 10
# Tukey's fences, in interquartile ranges beyond the quartiles, and the z-score cutoff.
MILD_FENCE = 1.5
EXTREME_FENCE = 3.0
Z_CUTOFF = 3.0
# How many row positions the metrics list per column.
POSITIONS = 10

TOO_FEW_VALUES = "too_few_values"
TOO_FEW_DISTINCT = "too_few_distinct"
ZERO_IQR = "zero_iqr"


def check_outliers(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Find outlier candidates in every numeric column.

    A value is a candidate if it lies beyond Tukey's fences (1.5 interquartile ranges outside the
    quartiles) or has a z-score above 3 in absolute value. It is extreme if it is beyond the wide
    fences (3.0 interquartile ranges) and its z-score is also above 3. Quartiles use linear
    interpolation. Only finite values are used, and the number of infinite values left out is
    recorded.

    A column is skipped, and the reason recorded, if it has fewer than ``MIN_VALUES`` finite
    values, fewer than ``MIN_DISTINCT`` distinct values, or an interquartile range of zero. In
    each case the method would give confident nonsense: a binary column has an interquartile range
    of zero and would flag every 1. Identifier and boolean columns are not numeric here.

    Every row is used, so the size guardrails of ``config`` do not apply. ``config`` is recorded.

    Findings show counts, quartiles and fences. They never show the values themselves, because a
    value can identify someone. The lowest and highest values, the fences and the first row
    positions are in ``metrics`` for anyone who needs them.

    Findings:

    - one MEDIUM finding for columns with extreme values;
    - one LOW finding for columns with milder candidates only;
    - one INFO finding for columns skipped for a zero interquartile range or too few values.
      Columns with few distinct values stay silent, since they are not measurements;
    - one INFO finding when infinite values were left out.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for outliers in a dataset with no rows")

    schema = profile_schema(dataset).metrics["columns"]
    columns = [
        _assess(str(frame.columns[i]), frame.iloc[:, i])
        for i, entry in enumerate(schema)
        if entry["semantic_type"] == "numeric"
    ]

    assessed = [c for c in columns if c["status"] == "assessed"]
    extreme = [c for c in assessed if c["extreme"]]
    milder = [c for c in assessed if c["flagged"] and not c["extreme"]]
    unassessable = [c for c in columns if c["reason"] in (TOO_FEW_VALUES, ZERO_IQR)]
    infinite = [c for c in columns if c["n_infinite"]]

    findings: list[Finding] = []
    if extreme:
        findings.append(report.extreme_finding(extreme, infinite))
    if milder:
        findings.append(report.candidates_finding(milder, infinite))
    if unassessable:
        measurements = sum(1 for c in columns if c["reason"] != TOO_FEW_DISTINCT)
        findings.append(report.skipped_finding(unassessable, measurements))
    if infinite:
        findings.append(report.infinite_finding(infinite))

    metrics = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "numeric_columns": len(columns),
        "assessed": len(assessed),
        "columns": columns,
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _assess(name: str, series) -> dict:
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(values)
    entry = {
        "name": name,
        "status": "skipped",
        "reason": None,
        "n": int(finite.sum()),
        "n_infinite": int((~np.isnan(values) & ~finite).sum()),
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
        "iqr": None,
        "mild_low": None,
        "mild_high": None,
        "extreme_low": None,
        "extreme_high": None,
        "iqr_flagged": 0,
        "iqr_extreme": 0,
        "z_flagged": 0,
        "both_flagged": 0,
        "flagged": 0,
        "extreme": 0,
        "rate": 0.0,
        "first_positions": [],
    }
    x = values[finite]
    if x.size < MIN_VALUES:
        entry["reason"] = TOO_FEW_VALUES
        return entry
    if np.unique(x).size < MIN_DISTINCT:
        entry["reason"] = TOO_FEW_DISTINCT
        return entry
    q1, q3 = np.percentile(x, [25, 75], method="linear")
    iqr = q3 - q1
    if iqr == 0:
        entry["reason"] = ZERO_IQR
        return entry

    mild_low, mild_high = q1 - MILD_FENCE * iqr, q3 + MILD_FENCE * iqr
    extreme_low, extreme_high = q1 - EXTREME_FENCE * iqr, q3 + EXTREME_FENCE * iqr
    beyond_mild = (x < mild_low) | (x > mild_high)
    beyond_extreme = (x < extreme_low) | (x > extreme_high)
    z_flagged = np.abs((x - x.mean()) / x.std(ddof=1)) > Z_CUTOFF
    flagged = beyond_mild | z_flagged
    extreme = beyond_extreme & z_flagged

    entry.update(
        status="assessed",
        min=float(x.min()),
        max=float(x.max()),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        mild_low=float(mild_low),
        mild_high=float(mild_high),
        extreme_low=float(extreme_low),
        extreme_high=float(extreme_high),
        iqr_flagged=int(beyond_mild.sum()),
        iqr_extreme=int(beyond_extreme.sum()),
        z_flagged=int(z_flagged.sum()),
        both_flagged=int((beyond_mild & z_flagged).sum()),
        flagged=int(flagged.sum()),
        extreme=int(extreme.sum()),
        rate=float(flagged.sum() / x.size),
        first_positions=[int(p) for p in np.flatnonzero(finite)[flagged][:POSITIONS]],
    )
    return entry
