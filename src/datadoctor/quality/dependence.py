"""Statistics for how missing values relate to each other and to the observed data.

Every function works on plain arrays. Deciding what the numbers mean, and what to tell the user,
is the job of ``missingness.py``. The cutoffs below are conventions, not laws, and the findings
built from them say so.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

# Two missing-value indicators count as correlated from this absolute phi coefficient up.
MIN_ABS_PHI = 0.5
# Significance level, applied after a Bonferroni correction over the tests actually run.
ALPHA = 0.05
# The chi-square approximation needs every expected cell count to be at least this.
MIN_EXPECTED_COUNT = 5.0
# A numeric comparison needs at least this many values in each group.
MIN_GROUP_SIZE = 20
# With many rows almost any difference is significant, so a difference must also be this large:
# Cohen's d for numeric columns, Cramer's V for categorical ones (medium-sized effects).
MIN_ABS_D = 0.5
MIN_CRAMERS_V = 0.3
# Categorical predictors with more levels than this are not tested.
MAX_CATEGORIES = 50
# Only this many correlated pairs are listed. The true total is always reported.
MAX_LISTED_PAIRS = 200

SMALL_GROUP = "small_group"
SMALL_EXPECTED = "small_expected"
NO_VARIATION = "no_variation"
TOO_MANY_CATEGORIES = "too_many_categories"


class Pair(NamedTuple):
    """Two indicator columns whose missingness is correlated, as positions among the indicators."""

    a: int
    b: int
    phi: float
    n_both: int
    p_adjusted: float


class PairSummary(NamedTuple):
    """Correlation of missing-value indicators. ``above_threshold`` counts every reported pair."""

    tested: int
    skipped: int
    above_threshold: int
    listed: list[Pair]


class Predictor(NamedTuple):
    """An observed column that the missingness of another column depends on."""

    name: str
    kind: str
    effect: float
    p_adjusted: float


class ColumnDependence(NamedTuple):
    """What was learned about one column's missingness.

    ``tested`` is how many comparisons ran, ``skipped`` counts the ones that could not, by reason,
    and ``detected`` holds the predictors that passed both the significance and effect-size cutoffs.
    """

    tested: int
    skipped: dict[str, int]
    detected: list[Predictor]


class DependenceSummary(NamedTuple):
    """Dependence results for every indicator column, in indicator order."""

    tests_run: int
    columns: list[ColumnDependence]


def correlated_pairs(indicators: np.ndarray) -> PairSummary:
    """Find pairs of columns whose missing-value indicators are strongly correlated.

    ``indicators`` holds 0/1 values, one column per data column, and every column must have both
    values. The correlation of two 0/1 columns is the phi coefficient, and ``n * phi**2`` follows a
    chi-square distribution with one degree of freedom. Pairs with an expected cell count below
    ``MIN_EXPECTED_COUNT`` are skipped because that approximation does not hold for them.
    """
    n, q = indicators.shape
    if q < 2:
        return PairSummary(0, 0, 0, [])

    ones = indicators.sum(axis=0)
    zeros = n - ones
    both = indicators.T @ indicators
    expected = [
        np.outer(ones, ones) / n,
        np.outer(ones, zeros) / n,
        np.outer(zeros, ones) / n,
        np.outer(zeros, zeros) / n,
    ]
    upper = np.triu(np.ones((q, q), dtype=bool), k=1)
    valid = upper & (np.minimum.reduce(expected) >= MIN_EXPECTED_COUNT)
    tested = int(valid.sum())

    phi = (n * both - np.outer(ones, ones)) / np.sqrt(np.outer(ones * zeros, ones * zeros))
    p_adjusted = np.minimum(1.0, stats.chi2.sf(n * phi**2, 1) * tested)
    selected = valid & (np.abs(phi) >= MIN_ABS_PHI) & (p_adjusted < ALPHA)

    rows, cols = np.nonzero(selected)
    pairs = [
        Pair(int(a), int(b), float(phi[a, b]), int(both[a, b]), float(p_adjusted[a, b]))
        for a, b in zip(rows, cols, strict=True)
    ]
    pairs.sort(key=lambda pair: (-abs(pair.phi), pair.a, pair.b))
    return PairSummary(tested, int(upper.sum()) - tested, len(pairs), pairs[:MAX_LISTED_PAIRS])


def dependence_tests(
    frame: pd.DataFrame,
    indicators: np.ndarray,
    indicator_positions: list[int],
    numeric_positions: list[int],
    categorical_positions: list[int],
) -> DependenceSummary:
    """Test whether each column's missingness depends on the observed values of the others.

    For every indicator column and every predictor column, the rows are split by whether the
    indicator column is missing. A numeric predictor is compared between the two groups with
    Welch's t-test, with Cohen's d as the effect size. A categorical predictor is tested for
    independence with a chi-square test, with Cramer's V as the effect size. A column is never
    tested against itself.

    p-values are Bonferroni-adjusted over all comparisons that ran. A predictor is detected when
    the adjusted p-value is below ``ALPHA`` and the effect reaches ``MIN_ABS_D`` or
    ``MIN_CRAMERS_V``.

    Args:
        frame: The rows to test, one column per data column.
        indicators: 0/1 missing-value indicators, one column per column in ``indicator_positions``.
        indicator_positions: The position in ``frame`` of each indicator column.
        numeric_positions: Positions of the numeric columns used as predictors.
        categorical_positions: Positions of the categorical or boolean columns used as predictors.
    """
    q = indicators.shape[1]
    own = {position: j for j, position in enumerate(indicator_positions)}
    skipped: list[dict[str, int]] = [{} for _ in range(q)]
    records: list[tuple[int, str, str, float, float]] = []  # indicator, name, kind, effect, p

    def skip(j: int, reason: str, count: int = 1) -> None:
        if count:
            skipped[j][reason] = skipped[j].get(reason, 0) + count

    observed = 1.0 - indicators
    if numeric_positions:
        _numeric_tests(frame, numeric_positions, indicators, observed, own, records, skip)
    for position in categorical_positions:
        _categorical_test(frame, position, indicators, observed, own, records, skip)

    tests_run = len(records)
    tested = [0] * q
    detected: list[list[Predictor]] = [[] for _ in range(q)]
    for j, name, kind, effect, p in records:
        tested[j] += 1
        p_adjusted = min(1.0, p * tests_run)
        floor = MIN_ABS_D if kind == "numeric" else MIN_CRAMERS_V
        if p_adjusted < ALPHA and abs(effect) >= floor:
            detected[j].append(Predictor(name, kind, effect, p_adjusted))
    for found in detected:
        found.sort(key=lambda predictor: (predictor.p_adjusted, -abs(predictor.effect)))

    return DependenceSummary(
        tests_run, [ColumnDependence(tested[j], skipped[j], detected[j]) for j in range(q)]
    )


def _numeric_tests(frame, positions, indicators, observed, own, records, skip) -> None:
    columns = [frame.iloc[:, i].to_numpy(dtype="float64", na_value=np.nan) for i in positions]
    values = np.column_stack(columns)
    present = np.isfinite(values)
    centered = np.zeros_like(values)
    for c in range(values.shape[1]):
        column_present = present[:, c]
        if column_present.any():
            centered[column_present, c] = (
                values[column_present, c] - values[column_present, c].mean()
            )
    squared = centered * centered
    present_f = present.astype(np.float64)

    n1 = indicators.T @ present_f
    n0 = observed.T @ present_f
    s1, s0 = indicators.T @ centered, observed.T @ centered
    q1, q0 = indicators.T @ squared, observed.T @ squared

    groups_ok = (n1 >= MIN_GROUP_SIZE) & (n0 >= MIN_GROUP_SIZE)
    safe1, safe0 = np.where(groups_ok, n1, 2.0), np.where(groups_ok, n0, 2.0)
    mean1, mean0 = s1 / safe1, s0 / safe0
    var1 = np.maximum((q1 - s1**2 / safe1) / (safe1 - 1), 0.0)
    var0 = np.maximum((q0 - s0**2 / safe0) / (safe0 - 1), 0.0)
    se1, se0 = var1 / safe1, var0 / safe0
    se2 = se1 + se0
    varying = groups_ok & (se2 > 0)

    safe_se2 = np.where(varying, se2, 1.0)
    t = (mean1 - mean0) / np.sqrt(safe_se2)
    dof = safe_se2**2 / (se1**2 / (safe1 - 1) + se0**2 / (safe0 - 1) + (~varying))
    p = 2 * stats.t.sf(np.abs(t), dof)
    pooled_var = ((safe1 - 1) * var1 + (safe0 - 1) * var0) / (safe1 + safe0 - 2)
    d = (mean1 - mean0) / np.sqrt(np.where(varying, pooled_var, 1.0))

    for c, position in enumerate(positions):
        name = str(frame.columns[position])
        for j in range(indicators.shape[1]):
            if own.get(position) == j:
                continue
            if not groups_ok[j, c]:
                skip(j, SMALL_GROUP)
            elif not varying[j, c]:
                skip(j, NO_VARIATION)
            else:
                records.append((j, name, "numeric", float(d[j, c]), float(p[j, c])))


def _categorical_test(frame, position, indicators, observed, own, records, skip) -> None:
    q = indicators.shape[1]
    others = [j for j in range(q) if own.get(position) != j]
    codes, uniques = pd.factorize(frame.iloc[:, position])
    k = len(uniques)
    if k < 2:
        for j in others:
            skip(j, NO_VARIATION)
        return
    if k > MAX_CATEGORIES:
        for j in others:
            skip(j, TOO_MANY_CATEGORIES)
        return

    onehot = np.zeros((len(codes), k))
    rows = np.nonzero(codes >= 0)[0]
    onehot[rows, codes[rows]] = 1.0
    missing_counts = indicators.T @ onehot
    observed_counts = observed.T @ onehot
    grand = (missing_counts + observed_counts).sum(axis=1)
    safe_grand = np.where(grand > 0, grand, 1.0)
    category_totals = missing_counts + observed_counts
    expected_missing = (
        missing_counts.sum(axis=1, keepdims=True) * category_totals / safe_grand[:, None]
    )
    expected_observed = (
        observed_counts.sum(axis=1, keepdims=True) * category_totals / safe_grand[:, None]
    )
    smallest = np.minimum(expected_missing.min(axis=1), expected_observed.min(axis=1))
    usable = (grand > 0) & (smallest >= MIN_EXPECTED_COUNT)

    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (
            (missing_counts - expected_missing) ** 2 / expected_missing
            + (observed_counts - expected_observed) ** 2 / expected_observed
        ).sum(axis=1)
    p = stats.chi2.sf(np.where(usable, statistic, 0.0), k - 1)
    cramers_v = np.sqrt(np.where(usable, statistic, 0.0) / safe_grand)

    name = str(frame.columns[position])
    for j in others:
        if usable[j]:
            records.append((j, name, "categorical", float(cramers_v[j]), float(p[j])))
        else:
            skip(j, SMALL_EXPECTED)
