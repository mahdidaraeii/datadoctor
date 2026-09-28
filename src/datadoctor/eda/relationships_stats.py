"""Statistics for how columns relate to each other and to the target.

Every function works on plain arrays or a plain frame and returns numbers, never a ``Finding``.
Deciding what the numbers mean is the job of ``relationships.py`` and ``relationships_findings.py``.
The cutoffs below are conventions, not laws, and the findings built from them say so.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

# A Pearson correlation needs at least this many rows with both values present to be reported.
MIN_PAIRS = 20
# Column pairs are reported as correlated from this absolute Pearson correlation up.
MIN_ABS_CORRELATION = 0.8
# A one-way ANOVA group needs at least this many values to contribute a variance.
MIN_GROUP_SIZE = 2
MIN_GROUPS = 2
# The chi-square approximation behind Cramer's V needs every expected cell count to be at least
# this, the same convention used for the missingness dependence tests.
MIN_EXPECTED_COUNT = 5.0
# A categorical column with more levels than this is not tested against the target: the
# contingency table would be too sparse to mean anything, and mostly-unique columns are already
# typed as identifiers and excluded before this point.
MAX_CATEGORIES = 50
# A histogram of the target covers this many interquartile ranges beyond the quartiles, the same
# convention the univariate histograms use.
FENCE_IQRS = 3.0


class Eta(NamedTuple):
    """A one-way ANOVA: how much of a numeric column's variance sits between groups.

    ``eta2`` is the effect size, 0 to 1, comparable to a squared correlation. ``p`` is the raw
    significance of the F-test, not yet adjusted for multiple comparisons.
    """

    eta2: float
    p: float


class CramersV(NamedTuple):
    """A chi-square test of independence between two categorical columns.

    ``v`` is the effect size, 0 to 1, comparable to an absolute correlation. ``p`` is the raw
    significance of the test, not yet adjusted for multiple comparisons.
    """

    v: float
    p: float


def correlation_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Pearson correlation between every pair of columns of ``frame``, as a plain array.

    The result is addressed by position, not by column label, since column labels are not
    guaranteed to be unique text everywhere a ``Dataset`` can come from. A pair with fewer than
    ``MIN_PAIRS`` rows where both columns have a value is NaN, since the estimate would be
    unreliable, and so is a column with no variance, against everything including itself.
    """
    return frame.corr(method="pearson", min_periods=MIN_PAIRS).to_numpy(dtype="float64")


def correlated_pairs(names: list[str], matrix: np.ndarray) -> list[tuple[str, str, float]]:
    """Column pairs whose absolute correlation reaches ``MIN_ABS_CORRELATION``, strongest first.

    ``names`` are the display names of the columns behind ``matrix``, in the same order.
    """
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            r = matrix[i, j]
            if not np.isnan(r) and abs(r) >= MIN_ABS_CORRELATION:
                pairs.append((a, names[j], float(r)))
    pairs.sort(key=lambda item: -abs(item[2]))
    return pairs


def eta_squared(values: np.ndarray, codes: np.ndarray) -> Eta | None:
    """How much of a numeric column's variance sits between the groups named by ``codes``.

    ``values`` and ``codes`` must be the same length and already have any row with a missing
    value or code (negative, as ``pandas.factorize`` marks a missing group) removed. ``None``
    when fewer than ``MIN_GROUPS`` groups have at least ``MIN_GROUP_SIZE`` values, or when there
    is no variance at all to divide between and within the groups.
    """
    groups = [values[codes == code] for code in np.unique(codes)]
    groups = [group for group in groups if group.size >= MIN_GROUP_SIZE]
    if len(groups) < MIN_GROUPS:
        return None
    pooled = np.concatenate(groups)
    grand_mean = pooled.mean()
    ss_between = sum(group.size * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    ss_total = ss_between + ss_within
    if ss_total <= 0:
        return None
    # dof_within is pooled.size - len(groups), and every kept group has at least MIN_GROUP_SIZE
    # values, so dof_within is always at least (MIN_GROUP_SIZE - 1) * len(groups), positive
    # whenever MIN_GROUP_SIZE is at least 2 as it is here. No separate check is needed for it.
    dof_between, dof_within = len(groups) - 1, pooled.size - len(groups)
    if ss_within <= 0:
        f = np.inf
    else:
        f = (ss_between / dof_between) / (ss_within / dof_within)
    return Eta(float(ss_between / ss_total), float(stats.f.sf(f, dof_between, dof_within)))


def cramers_v(a_codes: np.ndarray, b_codes: np.ndarray) -> CramersV | None:
    """Association between two categorical columns, from their group codes.

    The codes must be the same length and already have any row missing from either column
    removed (a negative code, as ``pandas.factorize`` marks a missing group). ``None`` when
    either column has fewer than two groups, or when an expected cell count in the contingency
    table is below ``MIN_EXPECTED_COUNT``, since the chi-square approximation does not hold then.
    """
    table = pd.crosstab(a_codes, b_codes).to_numpy(dtype="float64")
    rows, columns = table.shape
    if rows < 2 or columns < 2:
        return None
    n = table.sum()
    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / n
    if expected.min() < MIN_EXPECTED_COUNT:
        return None
    chi2 = float(((table - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, (rows - 1) * (columns - 1)))
    v = float(np.sqrt(chi2 / (n * (min(rows, columns) - 1))))
    return CramersV(v, p)


class ColumnAssociation(NamedTuple):
    """One column's association with the target: which statistic, and its value and p-value.

    ``effect`` is always on the same 0 to 1 scale as an absolute correlation, whichever
    statistic produced it: Pearson's own absolute value, the correlation ratio (the square root
    of eta-squared, not eta-squared itself), or Cramer's V. A threshold applied to ``effect``
    means the same thing regardless of which kind of column pair produced it. ``p`` is the raw
    significance, not yet adjusted for how many columns were tested; that adjustment is the
    caller's job, since it depends on how many other columns it also tested.
    """

    kind: str
    effect: float
    p: float


def column_association(
    series: pd.Series,
    numeric: bool,
    task: str,
    target_values: np.ndarray | None,
    target_codes: np.ndarray | None,
) -> ColumnAssociation | None:
    """A single column's association with the target, dispatched by its type and the target's.

    ``numeric`` is whether ``series`` itself is numeric; a categorical or boolean column passes
    ``False``. ``task`` is ``"regression"`` (the target is ``target_values``, a plain numeric
    array) or anything else (the target is ``target_codes``, from ``pandas.factorize``, with a
    negative code marking a missing value). Numeric against numeric is Pearson correlation,
    numeric against categorical is the correlation ratio, categorical against categorical is
    Cramer's V.

    Returns ``None`` when there is not enough overlapping, varying data to compute the statistic
    reliably. This does not check ``series``'s cardinality: a categorical column with too many
    distinct values to test meaningfully (see ``MAX_CATEGORIES``) is the caller's concern, to be
    checked before calling this, since a good reason to skip a column is not the same as this
    function being unable to compute an answer.
    """
    if numeric:
        feature = series.to_numpy(dtype="float64", na_value=np.nan)
        if task == "regression":
            pearson = _pearson(feature, target_values)
            if pearson is None:
                return None
            return ColumnAssociation("pearson", pearson[0], pearson[1])
        mask = np.isfinite(feature) & (target_codes >= 0)
        result = eta_squared(feature[mask], target_codes[mask])
        if result is None:
            return None
        return ColumnAssociation("eta_squared", float(np.sqrt(result.eta2)), result.p)

    codes, _ = pd.factorize(series)
    if task == "regression":
        mask = (codes >= 0) & np.isfinite(target_values)
        result = eta_squared(target_values[mask], codes[mask])
        if result is None:
            return None
        return ColumnAssociation("eta_squared", float(np.sqrt(result.eta2)), result.p)
    mask = (codes >= 0) & (target_codes >= 0)
    result = cramers_v(codes[mask], target_codes[mask])
    if result is None:
        return None
    return ColumnAssociation("cramers_v", result.v, result.p)


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """A feature's Pearson correlation and p-value against a target, or ``None`` if unreliable.

    ``None`` below ``MIN_PAIRS`` rows with both values present, or if either side has no
    variance there, since the correlation is undefined.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < MIN_PAIRS:
        return None
    x, y = a[mask], b[mask]
    if x.std() == 0 or y.std() == 0:
        return None
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def wide_fence_range(finite: np.ndarray, q1: float, q3: float) -> dict[str, float | int] | None:
    """The range a histogram of ``finite`` should cover, if a few extreme values would squeeze
    the rest of it into a few bars.

    The range is from ``FENCE_IQRS`` interquartile ranges below the first quartile to as many
    above the third, cut back to the data where the data ends sooner. Values outside it are
    counted, not dropped. ``None`` when nothing lies outside, and when the interquartile range
    is zero, since the fences would then have no width. This is the same rule the univariate
    histograms use, and the ``eda.univariate`` tests pin its behaviour in detail.
    """
    spread = q3 - q1
    if spread <= 0:
        return None
    low = max(float(finite.min()), q1 - FENCE_IQRS * spread)
    high = min(float(finite.max()), q3 + FENCE_IQRS * spread)
    below, above = int((finite < low).sum()), int((finite > high).sum())
    if not below and not above:
        return None
    return {"low": low, "high": high, "below": below, "above": above}
