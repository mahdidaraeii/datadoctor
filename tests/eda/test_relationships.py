from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity, load_dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.eda import check_relationships
from datadoctor.eda import relationships_plots as plots

REAL_FILE = Path(__file__).parents[2] / "data" / "telecom_customers.csv"


@pytest.fixture
def figures(monkeypatch):
    """The figures as they were drawn, by file name, so that their contents can be checked."""
    saved = {}
    real = plots.save_figure

    def capture(fig, path):
        saved[Path(path).name] = fig
        return real(fig, path)

    monkeypatch.setattr(plots, "save_figure", capture)
    return saved


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(output_dir=tmp_path / "plots")


def run(frame, config, target=None):
    return check_relationships(Dataset(data=frame, name="t", target=target), config)


def finding(result, title):
    (found,) = [f for f in result.findings if f.title == title]
    return found


def findings(result):
    return [f.title for f in result.findings]


def visible(fig):
    return [axis for axis in fig.axes if axis.get_visible()]


def association(result, name):
    return next(a for a in result.metrics["target"]["associations"] if a["name"] == name)


class TestCorrelationMatrix:
    def test_a_perfect_positive_and_a_perfect_negative_correlation(self, config, figures):
        # Non-whole-number, non-sequential values, so none of these columns is typed as an
        # identifier and excluded from the matrix.
        base = np.linspace(0, 1, 30) ** 1.3
        frame = pd.DataFrame({"a": base, "b": base * 2 + 1, "c": -base, "d": range(30)})

        result = run(frame, config)

        corr = result.metrics["correlation"]
        assert corr["computed"] is True
        columns = corr["columns"]
        matrix = corr["matrix"]
        assert matrix[columns.index("a")][columns.index("b")] == pytest.approx(1.0)
        assert matrix[columns.index("a")][columns.index("c")] == pytest.approx(-1.0)
        assert matrix[columns.index("a")][columns.index("a")] == pytest.approx(1.0)

    def r_of(self, config, overlap):
        # 40 rows, but only the first ``overlap`` have both "a" and "b", by construction always
        # a perfect correlation where they do overlap, so a reported cell would be exactly 1.0.
        rs = np.random.RandomState(0)
        a = rs.normal(size=40)
        b = a.copy()
        b[overlap:] = np.nan
        frame = pd.DataFrame({"a": a, "b": b, "c": rs.normal(size=40)})
        matrix = run(frame, config).metrics["correlation"]
        columns, values = matrix["columns"], matrix["matrix"]
        return values[columns.index("a")][columns.index("b")]

    def test_the_boundary_is_exactly_twenty_rows_in_common(self, config, figures):
        assert self.r_of(config, overlap=19) is None
        assert self.r_of(config, overlap=20) == pytest.approx(1.0)

    def test_a_constant_column_has_no_defined_correlation(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=30), "flat": [5.0] * 30})

        matrix = run(frame, config).metrics["correlation"]
        columns, values = matrix["columns"], matrix["matrix"]
        assert values[columns.index("a")][columns.index("flat")] is None
        assert values[columns.index("flat")][columns.index("flat")] is None

    def test_duplicate_column_labels_do_not_crash_the_matrix(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(rs.normal(size=(30, 3)))
        frame.columns = ["a", "a", "b"]

        result = run(frame, config)

        assert len(result.metrics["correlation"]["columns"]) == 3


class TestCorrelatedColumnsFinding:
    def test_pairs_at_or_above_the_threshold_are_reported(self, config, figures):
        rs = np.random.RandomState(0)
        base = rs.normal(size=30)
        frame = pd.DataFrame({"a": base, "b": base, "c": rs.normal(size=30)})

        result = run(frame, config)

        found = finding(result, "Numeric columns are strongly correlated")
        assert found.severity is Severity.LOW
        assert found.confidence == 1.0
        assert set(found.affected_columns) == {"a", "b"}
        assert "a / b" in found.evidence

    def test_pairs_below_the_threshold_are_not_reported(self, config, figures):
        rs = np.random.RandomState(0)
        a = rs.normal(size=200)
        b = a * 0.3 + rs.normal(size=200)  # a weak relationship, r well under 0.8
        frame = pd.DataFrame({"a": a, "b": b})

        result = run(frame, config)

        assert "Numeric columns are strongly correlated" not in findings(result)

    def test_the_threshold_is_exactly_zero_point_eight_inclusive(self, config, figures):
        # Two unit, mean-zero, mutually orthogonal vectors a and o: r*a + sqrt(1-r^2)*o then has
        # a sample correlation with a of exactly r, by construction, not just approximately.
        n = 40
        a = np.linspace(-1, 1, n)
        a = (a - a.mean()) / np.sqrt(((a - a.mean()) ** 2).sum())
        raw = np.array([((-1) ** k) * (k % 5) for k in range(n)], dtype=float)
        raw = raw - raw.mean()
        o = raw - (raw @ a) * a
        o = o / np.sqrt((o**2).sum())

        def with_r(r):
            return r * a + np.sqrt(1 - r**2) * o

        just_under, at, just_over = with_r(0.799), with_r(0.8), with_r(0.801)
        assert np.corrcoef(a, just_under)[0, 1] == pytest.approx(0.799, abs=1e-9)
        assert np.corrcoef(a, at)[0, 1] == pytest.approx(0.8, abs=1e-9)
        frame = pd.DataFrame({"a": a, "just_under": just_under, "at": at, "just_over": just_over})

        pairs = {(p["a"], p["b"]) for p in run(frame, config).metrics["correlation"]["pairs"]}

        assert ("a", "just_under") not in pairs
        assert ("a", "at") in pairs
        assert ("a", "just_over") in pairs


class TestColumnGuardrail:
    def test_the_matrix_is_skipped_above_the_column_threshold_and_the_guardrail_says_so(
        self, figures, tmp_path
    ):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({f"n{i}": rs.normal(size=30) for i in range(5)})
        config = AnalysisConfig(output_dir=tmp_path, column_threshold=3)

        result = run(frame, config)

        assert result.metrics["correlation"] == {
            "computed": False,
            "columns": [],
            "matrix": None,
            "pairs": [],
        }
        assert "correlation_heatmap" not in result.artifacts
        assert any(f.category == "guardrail" and "Pairwise" in f.title for f in result.findings)

    def test_fewer_than_two_numeric_columns_skips_without_a_guardrail_finding(
        self, config, figures
    ):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=30), "cat": ["x", "y"] * 15})

        result = run(frame, config)

        assert result.metrics["correlation"] == {
            "computed": False,
            "columns": [],
            "matrix": None,
            "pairs": [],
        }
        assert "correlation_heatmap" not in result.artifacts
        assert not any(f.category == "guardrail" for f in result.findings)


class TestRowGuardrail:
    def test_a_large_dataset_is_analyzed_on_a_sample_and_the_findings_say_so(
        self, figures, tmp_path
    ):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(
            {"a": rs.normal(size=500), "b": rs.normal(size=500), "target": rs.normal(size=500)}
        )
        config = AnalysisConfig(output_dir=tmp_path, row_threshold=100)

        result = run(frame, config, target="target")

        sample_finding = [
            f for f in result.findings if f.title == "Analysis ran on a sample of the rows"
        ]
        assert len(sample_finding) == 1
        assert "sample of 100" in sample_finding[0].evidence

    def test_the_same_seed_gives_the_same_sample_and_the_same_result(self, tmp_path, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(
            {"a": rs.normal(size=500), "b": rs.normal(size=500), "target": rs.normal(size=500)}
        )
        config = AnalysisConfig(output_dir=tmp_path / "a", row_threshold=100, random_seed=7)
        other = AnalysisConfig(output_dir=tmp_path / "b", row_threshold=100, random_seed=7)

        first = run(frame, config, target="target")
        second = run(frame, other, target="target")

        assert first.metrics["correlation"]["matrix"] == second.metrics["correlation"]["matrix"]


class TestNotUsedColumns:
    def test_identifier_text_datetime_and_unknown_columns_are_named_with_reasons(
        self, config, figures
    ):
        rs = np.random.RandomState(0)
        n = 60
        frame = pd.DataFrame(
            {
                "num": rs.normal(size=n),
                "ident": [f"ID-{k:04d}" for k in range(n)],
                "memo": [f"order {k} was shipped and signed for on time" for k in range(n)],
                "when": [f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}" for k in range(n)],
                "nested": [[k] for k in range(n)],
            }
        )

        result = run(frame, config)

        note = finding(result, "Some columns were not compared")
        assert note.severity is Severity.INFO
        assert "identifiers, where every value is different (ident)" in note.evidence
        assert "free text" in note.evidence and "memo" in note.evidence
        assert "dates and times" in note.evidence and "when" in note.evidence
        assert "mixed, nested or no values" in note.evidence and "nested" in note.evidence
        assert set(note.affected_columns) == set()  # not_used_finding names no columns field


class TestTarget:
    def test_a_numeric_target_is_a_regression_task(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=40), "y": rs.normal(size=40)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["task"] == "regression"
        assert result.metrics["target"]["class_balance"] is None
        assert result.metrics["target"]["distribution"] is not None

    def test_a_categorical_target_is_a_classification_task(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=40), "y": rs.choice(["x", "y", "z"], size=40)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["task"] == "classification"
        assert result.metrics["target"]["distribution"] is None
        assert result.metrics["target"]["class_balance"] is not None

    def test_a_boolean_target_is_also_a_classification_task(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=40), "y": rs.choice([True, False], size=40)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["task"] == "classification"

    def test_no_target_set_gives_no_target_section(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=40), "b": rs.normal(size=40)})

        result = run(frame, config)

        assert result.metrics["target"] is None
        assert "target_distribution" not in result.artifacts

    @pytest.mark.parametrize(
        ("column", "kind"),
        [
            # An identifier is not in this list: the schema never types a target column as an
            # identifier (that exemption is S10's, and this module inherits it, see
            # profile_schema's "the target column never is [an identifier]").
            ("memo", "text"),
            ("when", "datetime"),
            ("nested", "unknown"),
        ],
    )
    def test_an_unusable_target_type_is_named_and_excluded(self, config, figures, column, kind):
        rs = np.random.RandomState(0)
        n = 60
        frame = pd.DataFrame(
            {
                "a": rs.normal(size=n),
                "memo": [f"order {k} was shipped and signed for on time" for k in range(n)],
                "when": [f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}" for k in range(n)],
                "nested": [[k] for k in range(n)],
            }
        )

        result = run(frame, config, target=column)

        assert result.metrics["target"] == {"name": column, "usable": False, "reason": kind}
        assert "target_distribution" not in result.artifacts
        note = finding(result, "The target could not be compared against the other columns")
        assert note.affected_columns == (column,)
        # the target itself does not also appear in the "not compared" note for other columns
        assert column not in finding(result, "Some columns were not compared").evidence


class TestAssociations:
    def test_a_numeric_feature_against_a_numeric_target_is_pearson(self, config, figures):
        base = np.linspace(0, 1, 40) ** 1.3
        frame = pd.DataFrame({"x": base, "y": base * 2 - 3})

        result = run(frame, config, target="y")

        a = association(result, "x")
        assert a["kind"] == "pearson"
        assert a["effect"] == pytest.approx(1.0)

    def test_pearson_still_runs_against_a_discrete_zero_one_numeric_target(self, config, figures):
        # A 0/1-coded target gets the class-balance display (see TestImbalanceThresholds), but a
        # point-biserial correlation against it is still a legitimate use of Pearson, and it
        # stays the association used: this is deliberately independent of the display choice.
        rs = np.random.RandomState(0)
        y = rs.randint(0, 2, size=200).astype(float)
        x = y * 3 + rs.normal(size=200)
        frame = pd.DataFrame({"y": y, "x": x})

        result = run(frame, config, target="y")

        a = association(result, "x")
        assert a["kind"] == "pearson"

    def test_a_categorical_feature_against_a_numeric_target_is_the_correlation_ratio(
        self, config, figures
    ):
        # Two groups, [1, 2, 3] and [7, 8, 9]: SS between 54, SS within 4, eta2 = 54 / 58.
        values = [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
        groups = ["low", "low", "low", "high", "high", "high"]
        frame = pd.DataFrame({"y": values, "g": groups})

        result = run(frame, config, target="y")

        a = association(result, "g")
        assert a["kind"] == "eta_squared"
        assert a["effect"] == pytest.approx((54 / 58) ** 0.5)
        f_expected, p_expected = scipy_stats.f_oneway([1.0, 2.0, 3.0], [7.0, 8.0, 9.0])
        assert a["p_adjusted"] == pytest.approx(p_expected)

    def test_the_correlation_ratio_weighs_groups_by_their_size_not_by_count(self, config, figures):
        # A small group of 2 and a big group of 8, each with some spread of its own, so a grand
        # mean weighted by size (8.4) differs from one that treats both groups equally (6.0), and
        # the two give different eta-squared values, not just different intermediate means.
        small, big = [1.0, 3.0], [9.0, 11.0] * 4
        frame = pd.DataFrame({"y": small + big, "g": ["small"] * 2 + ["big"] * 8})

        result = run(frame, config, target="y")

        weighted_mean = (sum(small) + sum(big)) / 10
        ss_between = (
            len(small) * (2.0 - weighted_mean) ** 2 + len(big) * (10.0 - weighted_mean) ** 2
        )
        ss_within = sum((v - 2.0) ** 2 for v in small) + sum((v - 10.0) ** 2 for v in big)
        eta2 = ss_between / (ss_between + ss_within)
        unweighted_mean = (2.0 + 10.0) / 2
        wrong_ss_between = (2.0 - unweighted_mean) ** 2 + (10.0 - unweighted_mean) ** 2
        wrong_eta2 = wrong_ss_between / (wrong_ss_between + ss_within)
        assert eta2 != pytest.approx(wrong_eta2)  # the two formulas really do disagree here

        a = association(result, "g")
        assert a["effect"] == pytest.approx(eta2**0.5)

    def test_no_variance_anywhere_gives_no_association(self, config, figures):
        frame = pd.DataFrame({"y": [5.0] * 10, "g": ["a"] * 5 + ["b"] * 5})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["not_tested"] == [
            {"name": "g", "reason": "not_enough_data"}
        ]

    def test_a_group_of_exactly_one_row_is_dropped_and_can_collapse_to_a_single_group(
        self, config, figures
    ):
        # "rare" has one row, "common" has the rest: MIN_GROUP_SIZE (2) drops "rare", leaving one
        # group, which is below MIN_GROUPS (2).
        rs = np.random.RandomState(0)
        y = rs.normal(size=30)
        g = ["common"] * 29 + ["rare"]
        frame = pd.DataFrame({"y": y, "g": g})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["not_tested"] == [
            {"name": "g", "reason": "not_enough_data"}
        ]

    def test_two_categorical_columns_use_cramers_v(self, config, figures):
        # A perfectly aligned 2x2 table: V = 1.0 exactly.
        a_values = ["a1"] * 10 + ["a2"] * 10
        b_values = ["b1"] * 10 + ["b2"] * 10
        frame = pd.DataFrame({"f": a_values, "y": b_values})

        result = run(frame, config, target="y")

        a = association(result, "f")
        assert a["kind"] == "cramers_v"
        assert a["effect"] == pytest.approx(1.0)

    def test_cramers_v_and_its_p_value_match_an_independent_chi_square_test(self, config, figures):
        # A 3x3 table, so min(rows, columns) - 1 = 2: for a 2x2 table this factor is always 1,
        # which cannot tell a normalized V from an unnormalized one. The counts are uneven and
        # partial, so chi2 is neither 0 nor the degenerate n*(k-1) of a perfect alignment either.
        table = np.array([[12, 3, 1], [4, 10, 2], [2, 3, 9]]) * 3  # scaled so every expected
        # cell count clears MIN_EXPECTED_COUNT
        f_values, y_values = [], []
        for row, category in enumerate("abc"):
            for column, target_label in enumerate("xyz"):
                f_values += [category] * table[row, column]
                y_values += [target_label] * table[row, column]
        frame = pd.DataFrame({"f": f_values, "y": y_values})

        result = run(frame, config, target="y")

        chi2, p_expected, _, _ = scipy_stats.chi2_contingency(table, correction=False)
        n = table.sum()
        v_expected = (chi2 / (n * 2)) ** 0.5  # min(3, 3) - 1 == 2
        a = association(result, "f")
        assert a["effect"] == pytest.approx(v_expected)
        assert a["p_adjusted"] == pytest.approx(p_expected)

    def test_a_balanced_independent_table_gives_zero_association(self, config, figures):
        a_values = (["a1"] * 5 + ["a2"] * 5) * 2
        b_values = ["b1"] * 10 + ["b2"] * 10
        frame = pd.DataFrame({"f": a_values, "y": b_values})

        result = run(frame, config, target="y")

        a = association(result, "f")
        assert a["effect"] == pytest.approx(0.0, abs=1e-9)
        assert a["p_adjusted"] == pytest.approx(1.0)

    def test_a_single_category_feature_cannot_be_compared_by_cramers_v_either(
        self, config, figures
    ):
        y = ["b1"] * 15 + ["b2"] * 15
        frame = pd.DataFrame({"f": ["same"] * 30, "y": y})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["not_tested"] == [
            {"name": "f", "reason": "not_enough_data"}
        ]

    def test_the_expected_count_boundary_is_exactly_five_inclusive(self, config, figures):
        # A 2x2 table, n=20, evenly split: every expected count is exactly 5.0, at the limit.
        # n=18 with the same proportions gives an expected count of 4.5, just under it.
        def table(n):
            half = n // 2
            f_values = (["f1"] * (half // 2) + ["f2"] * (half - half // 2)) * 2
            y_values = ["y1"] * half + ["y2"] * (n - half)
            return pd.DataFrame({"f": f_values, "y": y_values})

        at_limit = run(table(20), config, target="y")
        under_limit = run(table(18), config, target="y")

        assert association(at_limit, "f")["kind"] == "cramers_v"
        assert under_limit.metrics["target"]["not_tested"] == [
            {"name": "f", "reason": "not_enough_data"}
        ]

    def categories_frame(self, n_categories):
        rs = np.random.RandomState(0)
        n = n_categories * 4
        codes = np.arange(n) % n_categories
        return pd.DataFrame({"y": rs.normal(size=n), "many": [f"c{c}" for c in codes]})

    def test_the_category_limit_is_exactly_fifty_inclusive(self, config, figures):
        at_limit = run(self.categories_frame(50), config, target="y")
        over_limit = run(self.categories_frame(51), config, target="y")

        assert at_limit.metrics["target"]["not_tested"] == []
        assert association(at_limit, "many")["kind"] == "eta_squared"
        assert over_limit.metrics["target"]["not_tested"] == [
            {"name": "many", "reason": "too_many_categories"}
        ]

    def test_a_feature_with_too_little_overlapping_data_is_not_tested(self, config, figures):
        rs = np.random.RandomState(0)
        y = rs.normal(size=40)
        x = np.full(40, np.nan)
        x[:5] = rs.normal(size=5)  # far fewer than MIN_PAIRS present rows
        frame = pd.DataFrame({"y": y, "x": x})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["not_tested"] == [
            {"name": "x", "reason": "not_enough_data"}
        ]

    def test_a_column_with_a_single_category_cannot_be_a_group_predictor(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"y": rs.normal(size=30), "flat": ["same"] * 30})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["not_tested"] == [
            {"name": "flat", "reason": "not_enough_data"}
        ]

    def test_p_values_are_bonferroni_adjusted_over_the_features_that_were_tested(
        self, config, figures
    ):
        rs = np.random.RandomState(0)
        n = 200
        frame = pd.DataFrame(
            {
                "y": rs.normal(size=n),
                "a": rs.normal(size=n),
                "b": rs.normal(size=n),
                "c": rs.normal(size=n),
            }
        )

        result = run(frame, config, target="y")

        assert len(result.metrics["target"]["associations"]) == 3  # tests_run == 3
        for a in result.metrics["target"]["associations"]:
            _, raw_p = scipy_stats.pearsonr(frame["y"], frame[a["name"]])
            assert a["p_adjusted"] == pytest.approx(min(1.0, raw_p * 3))

    def test_the_target_never_appears_among_its_own_associations(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"y": rs.normal(size=30), "x": rs.normal(size=30)})

        result = run(frame, config, target="y")

        names = {a["name"] for a in result.metrics["target"]["associations"]}
        names |= {n["name"] for n in result.metrics["target"]["not_tested"]}
        assert "y" not in names

    def test_a_numeric_target_is_excluded_from_the_correlation_matrix_too(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"y": rs.normal(size=30), "x": rs.normal(size=30)})

        result = run(frame, config, target="y")

        assert "y" not in result.metrics["correlation"]["columns"]


class TestTopPredictorsFinding:
    def test_only_associations_past_the_effect_and_significance_cutoffs_are_listed(
        self, config, figures
    ):
        rs = np.random.RandomState(0)
        n = 500
        real = rs.normal(size=n)
        y = real * 3 + rs.normal(size=n)
        frame = pd.DataFrame(
            {"y": y, "real": real, "noise1": rs.normal(size=n), "noise2": rs.normal(size=n)}
        )

        result = run(frame, config, target="y")

        found = finding(result, "Columns most associated with the target")
        assert found.affected_columns == ("real",)
        assert len(result.metrics["target"]["associations"]) == 3  # all three still in metrics

    def test_no_finding_when_nothing_clears_the_cutoffs(self, config, figures):
        rs = np.random.RandomState(0)
        n = 60
        frame = pd.DataFrame(
            {"y": rs.normal(size=n), "a": rs.normal(size=n), "b": rs.normal(size=n)}
        )

        result = run(frame, config, target="y")

        assert "Columns most associated with the target" not in findings(result)

    def test_the_effect_cutoff_is_exactly_zero_point_one_even_when_significant(
        self, config, figures
    ):
        # A large enough sample makes even a weak correlation statistically significant, so this
        # isolates the MIN_EFFECT cutoff from the ALPHA one: both candidates clear ALPHA easily,
        # only the one at or above 0.1 should be reported.
        n = 3000
        a = np.linspace(-1, 1, n)
        a = (a - a.mean()) / np.sqrt(((a - a.mean()) ** 2).sum())
        raw = np.sin(np.arange(n, dtype=float))
        raw = raw - raw.mean()
        o = raw - (raw @ a) * a
        o = o / np.sqrt((o**2).sum())

        def with_r(r):
            return r * a + np.sqrt(1 - r**2) * o

        just_under, just_over = with_r(0.099), with_r(0.101)
        assert np.corrcoef(a, just_under)[0, 1] == pytest.approx(0.099, abs=1e-9)
        assert np.corrcoef(a, just_over)[0, 1] == pytest.approx(0.101, abs=1e-9)
        frame = pd.DataFrame({"y": a, "just_under": just_under, "just_over": just_over})

        result = run(frame, config, target="y")

        found = finding(result, "Columns most associated with the target")
        assert found.affected_columns == ("just_over",)


class TestClassBalance:
    def test_class_counts_and_shares_are_exact_and_ranked_most_frequent_first(
        self, config, figures
    ):
        y = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        frame = pd.DataFrame({"y": y, "x": range(100)})

        result = run(frame, config, target="y")

        balance = result.metrics["target"]["class_balance"]
        assert balance["classes"] == [
            {"label": "a", "count": 50, "share": 0.5},
            {"label": "b", "count": 30, "share": 0.3},
            {"label": "c", "count": 20, "share": 0.2},
        ]
        assert (balance["other_count"], balance["minority_label"]) == (0, "c")
        assert balance["minority_share"] == 0.2

    def test_equal_counts_are_ordered_by_label(self, config, figures):
        y = ["b"] * 20 + ["a"] * 20
        frame = pd.DataFrame({"y": y, "x": range(40)})

        classes = run(frame, config, target="y").metrics["target"]["class_balance"]["classes"]

        assert [c["label"] for c in classes] == ["a", "b"]

    def test_more_than_ten_classes_are_pooled_into_other(self, config, figures):
        counts = list(range(20, 8, -1))  # 12 classes, 20 down to 9, all distinct counts
        y = sum(([f"c{i}"] * count for i, count in enumerate(counts)), [])
        frame = pd.DataFrame({"y": y, "x": range(len(y))})

        result = run(frame, config, target="y")

        balance = result.metrics["target"]["class_balance"]
        assert len(balance["classes"]) == 10
        assert balance["other_count"] == counts[10] + counts[11]

    def test_the_plotted_bars_match_the_computed_counts(self, config, figures):
        y = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        frame = pd.DataFrame({"y": y, "x": range(100)})

        run(frame, config, target="y")

        axis = visible(figures["target_distribution.png"])[0]
        assert sorted(p.get_width() for p in axis.patches) == [20, 30, 50]


class TestImbalanceThresholds:
    def frame(self, minority):
        y = ["common"] * (100 - minority) + ["rare"] * minority
        return pd.DataFrame({"y": y, "x": range(100)})

    def test_just_below_five_percent_is_medium(self, config, figures):
        result = run(self.frame(4), config, target="y")

        found = finding(result, "The target classes are imbalanced")
        assert found.severity is Severity.MEDIUM

    def test_exactly_five_percent_is_low_not_medium(self, config, figures):
        result = run(self.frame(5), config, target="y")

        found = finding(result, "The target classes are imbalanced")
        assert found.severity is Severity.LOW

    def test_just_below_ten_percent_is_still_low(self, config, figures):
        result = run(self.frame(9), config, target="y")

        found = finding(result, "The target classes are imbalanced")
        assert found.severity is Severity.LOW

    def test_exactly_ten_percent_is_not_reported(self, config, figures):
        result = run(self.frame(10), config, target="y")

        assert "The target classes are imbalanced" not in findings(result)

    def test_the_interpretation_says_imbalance_is_often_expected_and_recommends_stratifying(
        self, config, figures
    ):
        result = run(self.frame(4), config, target="y")

        found = finding(result, "The target classes are imbalanced")
        assert "expected for many real classification problems" in found.interpretation
        assert "not a defect by itself" in found.interpretation
        assert "stratified" in found.recommendation

    def test_a_zero_one_numeric_target_is_also_checked_for_imbalance(self, config, figures):
        # 0/1 stored as numbers, not strings or booleans: schema types this column "numeric", so
        # reaching the imbalance check at all depends on _is_discrete_numeric, not on `kind`.
        y = np.array([0] * 195 + [1] * 5, dtype=float)
        frame = pd.DataFrame({"y": y, "x": np.random.RandomState(0).normal(size=200)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["task"] == "regression"
        found = finding(result, "The target classes are imbalanced")
        assert found.severity is Severity.MEDIUM


class TestPrivacy:
    def test_a_flagged_target_hides_its_class_labels_but_keeps_counts(self, config, figures):
        names = ["Anna", "Peter", "Maria", "John"] * 20
        frame = pd.DataFrame({"first_name": names, "x": range(80)})

        result = run(frame, config, target="first_name")

        balance = result.metrics["target"]["class_balance"]
        assert all("label" not in c for c in balance["classes"])
        assert balance["minority_label"] is None
        assert not any(name in result.to_json() for name in set(names))

    def test_the_plot_hides_the_labels_too(self, config, figures):
        names = ["Anna", "Peter", "Maria", "John"] * 20
        frame = pd.DataFrame({"first_name": names, "x": range(80)})

        run(frame, config, target="first_name")

        axis = visible(figures["target_distribution.png"])[0]
        assert list(axis.get_yticks()) == []
        assert "labels hidden" in axis.texts[0].get_text()


class TestFigures:
    def test_the_heatmap_cells_match_the_matrix(self, config, figures):
        base = np.linspace(0, 1, 30) ** 1.3
        frame = pd.DataFrame({"a": base, "b": base * 2, "c": -base})

        result = run(frame, config)

        image = visible(figures["correlation_heatmap.png"])[0].images[0]
        np.testing.assert_allclose(image.get_array(), result.metrics["correlation"]["matrix"])

    def test_a_numeric_targets_distribution_bars_sum_to_its_count(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"y": rs.normal(size=200), "x": rs.normal(size=200)})

        result = run(frame, config, target="y")

        axis = visible(figures["target_distribution.png"])[0]
        assert (
            sum(p.get_height() for p in axis.patches)
            == result.metrics["target"]["distribution"]["count"]
        )

    def test_a_discrete_numeric_target_gets_the_class_balance_view(self, config, figures):
        # A 0/1-coded target, such as churned stored as numbers rather than booleans: it stays a
        # "regression" task (Pearson is still valid against it), but its own display is the
        # classification-style class-balance chart, not a numeric histogram.
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"y": rs.randint(0, 2, size=200), "x": rs.normal(size=200)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["task"] == "regression"
        assert result.metrics["target"]["distribution"] is None
        assert result.metrics["target"]["class_balance"] is not None
        axis = visible(figures["target_distribution.png"])[0]
        assert len(axis.patches) == 2

    def test_a_few_valued_but_non_whole_numeric_target_stays_a_histogram(self, config, figures):
        # Few distinct values alone is not enough: they must be whole numbers too, or this would
        # wrongly catch a continuous target that happens to take few distinct readings.
        rs = np.random.RandomState(0)
        y = np.tile([0.5, 1.5, 2.5, 3.5, 4.5], 40)
        frame = pd.DataFrame({"y": y, "x": rs.normal(size=200)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["class_balance"] is None
        assert result.metrics["target"]["distribution"] is not None

    def test_the_discrete_boundary_is_exactly_fifteen_distinct_whole_values(self, config, figures):
        rs = np.random.RandomState(0)

        frame15 = pd.DataFrame({"y": np.arange(200) % 15, "x": rs.normal(size=200)})
        result15 = run(frame15, config, target="y")
        assert result15.metrics["target"]["class_balance"] is not None
        # 15 distinct classes, but TOP_CLASSES pools past 10: 10 named bars plus one "other" bar,
        # the same pooling every other classification target gets past that count.
        assert len(visible(figures["target_distribution.png"])[0].patches) == 11

        frame16 = pd.DataFrame({"y": np.arange(200) % 16, "x": rs.normal(size=200)})
        result16 = run(frame16, config, target="y")
        # 16 distinct values is over the discrete boundary: back to the numeric distribution.
        assert result16.metrics["target"]["class_balance"] is None
        assert result16.metrics["target"]["distribution"] is not None

    def test_file_names_are_pngs_under_the_configured_directory(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame(
            {"a": rs.normal(size=30), "b": rs.normal(size=30), "y": rs.normal(size=30)}
        )

        result = run(frame, config, target="y")

        assert set(result.artifacts) == {"correlation_heatmap", "target_distribution"}
        for path in result.artifacts.values():
            assert path.parent == config.output_dir
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_the_same_data_gives_the_same_files(self, tmp_path):
        rs = np.random.RandomState(0)

        def frame():
            r = np.random.RandomState(0)
            return pd.DataFrame(
                {"a": r.normal(size=30), "b": r.normal(size=30), "y": r.normal(size=30)}
            )

        first = run(frame(), AnalysisConfig(output_dir=tmp_path / "one"), target="y")
        second = run(frame(), AnalysisConfig(output_dir=tmp_path / "two"), target="y")

        for key in first.artifacts:
            assert first.artifacts[key].read_bytes() == second.artifacts[key].read_bytes()
        del rs


class TestNumericTargetRange:
    """The wide-fence treatment of extreme values for the target's own histogram."""

    def test_a_value_far_above_the_rest_is_kept_but_shown_as_one_edge_bar(self, config, figures):
        # 1 to 99 and 1000: quartiles 25.75 and 75.25, so the fences are 1 and 223.75.
        y = list(range(1, 100)) + [1000]
        frame = pd.DataFrame({"y": y, "x": np.random.RandomState(0).normal(size=100)})

        result = run(frame, config, target="y")

        distribution = result.metrics["target"]["distribution"]
        assert distribution["range_shown"] == {"low": 1.0, "high": 223.75, "below": 0, "above": 1}
        assert distribution["max"] == 1000  # the statistics still use every value
        assert distribution["count"] == 100

    def test_a_column_with_nothing_beyond_its_fences_is_drawn_whole(self, config, figures):
        y = np.arange(1, 101) * 2.0  # no non-integer step, but avoid identifier typing below
        y = y + 0.5
        frame = pd.DataFrame({"y": y, "x": np.random.RandomState(0).normal(size=100)})

        result = run(frame, config, target="y")

        axis = visible(figures["target_distribution.png"])[0]
        assert len(axis.texts) == 0  # no overflow label was drawn
        assert result.metrics["target"]["distribution"]["range_shown"] is None

    def test_the_high_fence_is_cut_back_to_the_datas_own_maximum(self, config, figures):
        # 1 to 20 plus one deep low outlier: q1=5, q3=15, so the high fence (3 IQR above q3)
        # would sit at 45, far past the data's own maximum of 20, which the range is cut to.
        y = [-1000.0] + list(range(1, 21))
        frame = pd.DataFrame({"y": y, "x": np.random.RandomState(0).normal(size=21)})

        result = run(frame, config, target="y")

        assert result.metrics["target"]["distribution"]["range_shown"] == {
            "low": -25.0,  # q1 - 3 * spread = 5 - 30, well above the data's actual minimum
            "high": 20.0,  # cut to the data's own maximum, not the uncapped fence at 45
            "below": 1,
            "above": 0,
        }


class TestResult:
    def test_the_dataset_is_left_as_it_was(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=30), "y": rs.normal(size=30)})
        before = frame.copy(deep=True)

        run(frame, config, target="y")

        pd.testing.assert_frame_equal(frame, before)

    def test_config_is_recorded_and_the_result_survives_json(self, config, figures):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({"a": rs.normal(size=30), "y": rs.normal(size=30)})

        result = run(frame, config, target="y")

        assert result.config == config
        assert AnalysisResult.from_json(result.to_json()) == result

    def test_a_dataset_without_rows_is_rejected(self, config):
        with pytest.raises(
            DatasetError, match="cannot look for relationships in a dataset with no rows"
        ):
            run(pd.DataFrame({"a": []}), config)

    def test_a_perfectly_correlated_pair_still_survives_json(self, config, figures):
        # r = 1.0 exactly can make a t-statistic infinite; the p-value must still be finite JSON.
        base = np.linspace(0, 1, 30) ** 1.3
        frame = pd.DataFrame({"a": base, "b": base, "y": base})

        result = run(frame, config, target="y")

        assert AnalysisResult.from_json(result.to_json()) == result


class TestTheExampleDataset:
    def test_the_telecom_file_produces_both_figures_without_crashing(self, config):
        assert REAL_FILE.exists(), f"the committed example dataset is missing: {REAL_FILE}"

        result = check_relationships(load_dataset(REAL_FILE, target="churned"), config)

        # churned is stored as 0/1 integers, and the schema never reinterprets integers as
        # booleans, so its associations still run as Pearson correlations, like profile_schema
        # and the univariate histograms already treat it. Its own display is nonetheless the
        # class-balance view, not a numeric histogram, since it is discrete with only 2 values.
        assert result.metrics["target"]["task"] == "regression"
        assert result.metrics["target"]["distribution"] is None
        class_balance = result.metrics["target"]["class_balance"]
        assert class_balance is not None
        assert sorted(c["count"] for c in class_balance["classes"]) == [178, 461]
        assert set(result.artifacts) == {"correlation_heatmap", "target_distribution"}
        assert not any(f.category == "privacy" for f in result.findings)
