import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity, load_dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.quality.missingness import check_missingness

N = 2000
CONFIG = AnalysisConfig()


def planted_frame() -> pd.DataFrame:
    """2,000 rows with missingness planted at known rates and with known structure."""
    rs = np.random.RandomState(1)
    x = rs.normal(size=N)
    region = np.array(["n", "s", "e", "w"] * (N // 4))
    frame = pd.DataFrame({"x": x, "region": region})

    def holes(name, positions):
        column = rs.normal(size=N)
        column[positions] = np.nan
        frame[name] = column

    shared = np.random.RandomState(3).choice(N, 200, replace=False)
    holes("mcar30", np.random.RandomState(2).choice(N, 600, replace=False))  # 30%, at random
    holes("pair_a", shared)  # 10%, missing together with pair_b
    holes("pair_b", shared)
    holes("depends", np.nonzero(x > np.quantile(x, 0.75))[0])  # 25%, exactly where x is high
    holes("by_region", np.nonzero(region == "n")[0])  # 25%, exactly where region is n
    holes("rare2", np.random.RandomState(4).choice(N, 40, replace=False))  # 2%
    holes("mostly", np.random.RandomState(5).choice(N, 1700, replace=False))  # 85%
    frame["all_present"] = rs.normal(size=N)
    label = np.array(["yes", "no"] * (N // 2), dtype=object)
    label[np.random.RandomState(6).choice(N, 60, replace=False)] = None  # 3%
    frame["label"] = label
    return frame


@pytest.fixture(scope="module")
def result():
    return check_missingness(Dataset(data=planted_frame(), name="planted"), CONFIG)


@pytest.fixture(scope="module")
def columns(result):
    return {column["name"]: column for column in result.metrics["columns"]}


def findings_for(result, column):
    return [f for f in result.findings if column in f.affected_columns]


def rate_severity(result, column):
    for finding in result.findings:
        if finding.title.endswith(f"missingness in {column}"):
            return finding.severity
    return None


class TestRates:
    def test_rates_are_exact_counts_over_every_row(self, result, columns):
        expected = {
            "mcar30": 600,
            "pair_a": 200,
            "pair_b": 200,
            "depends": 500,
            "by_region": 500,
            "rare2": 40,
            "mostly": 1700,
            "label": 60,
            "x": 0,
            "all_present": 0,
        }
        assert {name: columns[name]["n_missing"] for name in expected} == expected
        assert columns["mcar30"]["rate"] == 0.30
        assert columns["mostly"]["rate"] == 0.85

    def test_overall_counts_cells_and_incomplete_rows(self, result):
        frame = planted_frame()
        overall = result.metrics["overall"]

        assert overall["cells"] == frame.size
        assert overall["missing_cells"] == int(frame.isna().sum().sum())
        assert overall["rows_with_missing"] == int(frame.isna().any(axis=1).sum())
        assert overall["row_rate"] == overall["rows_with_missing"] / N

    def test_severity_follows_the_grades_with_inclusive_lower_bounds(self):
        counts = {"c49": 49, "c50": 50, "c199": 199, "c200": 200, "c399": 399, "c400": 400}
        counts |= {"c799": 799, "c800": 800}
        frame = pd.DataFrame(
            {name: [np.nan] * k + [1.0] * (1000 - k) for name, k in counts.items()}
        )

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        assert {name: rate_severity(result, name) for name in counts} == {
            "c49": None,
            "c50": Severity.LOW,
            "c199": Severity.LOW,
            "c200": Severity.MEDIUM,
            "c399": Severity.MEDIUM,
            "c400": Severity.HIGH,
            "c799": Severity.HIGH,
            "c800": Severity.CRITICAL,
        }

    def test_incomplete_rows_are_graded_like_a_rate(self, result):
        finding = next(f for f in result.findings if f.title.startswith("Many rows"))

        assert result.metrics["overall"]["row_rate"] >= 0.85
        assert finding.severity is Severity.CRITICAL

    def test_a_little_missing_gets_no_finding_at_all(self):
        frame = pd.DataFrame({"a": [np.nan] * 2 + [1.0] * 98, "b": [1.0] * 100})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        # Two missing values are too few to test, but the column is below the 5% rate that gets
        # any finding, so it is not named either. Its label is still in the metrics.
        assert result.findings == ()
        assert result.metrics["columns"][0]["dependence"] == "not_tested"

    def test_any_missing_target_value_is_at_least_high(self):
        dataset = Dataset(data=planted_frame(), name="t", target="label")

        result = check_missingness(dataset, CONFIG)

        (finding,) = [f for f in result.findings if f.title == "Target column has missing values"]
        assert finding.severity is Severity.HIGH  # 3% would otherwise be no finding at all
        assert "3.0%" in finding.evidence

    def test_a_target_missing_more_than_high_keeps_its_own_grade(self):
        frame = pd.DataFrame({"y": [np.nan] * 85 + [1.0] * 15, "a": range(100)})

        result = check_missingness(Dataset(data=frame, name="t", target="y"), CONFIG)

        (finding,) = [f for f in result.findings if f.title == "Target column has missing values"]
        assert finding.severity is Severity.CRITICAL

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            check_missingness(Dataset(data=pd.DataFrame({"a": []}), name="t"), CONFIG)


class TestCorrelation:
    def test_phi_matches_the_analytic_value_for_a_known_table(self):
        # 30 rows missing in both, 10 only in a, 10 only in b, 50 in neither: phi = 1400 / 2400.
        a = [np.nan] * 40 + [1.0] * 60
        b = [np.nan] * 30 + [1.0] * 10 + [np.nan] * 10 + [1.0] * 50
        frame = pd.DataFrame({"a": a, "b": b})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        (pair,) = result.metrics["pairs"]["listed"]
        assert pair["phi"] == pytest.approx(1400 / 2400, abs=1e-12)
        assert pair["n_both"] == 30
        (finding,) = [f for f in result.findings if f.title.startswith("Missingness is corr")]
        assert (finding.severity, finding.confidence) == (Severity.LOW, 0.7)

    def test_columns_missing_together_are_reported_as_one_finding(self, result):
        (finding,) = [f for f in result.findings if f.title.startswith("Missingness is corr")]
        (pair,) = result.metrics["pairs"]["listed"]

        assert result.metrics["pairs"]["above_threshold"] == 1
        assert (pair["a"], pair["b"], pair["phi"]) == ("pair_a", "pair_b", 1.0)
        assert finding.affected_columns == ("pair_a", "pair_b")
        assert (finding.severity, finding.confidence) == (Severity.MEDIUM, 0.9)

    def test_a_weak_but_significant_correlation_is_not_reported(self):
        # phi is about 0.10 with 20,000 rows: hugely significant, but far below the 0.5 cutoff.
        rs = np.random.RandomState(0)
        n = 20000
        a = rs.rand(n) < 0.30
        b = np.where(a, rs.rand(n) < 0.36, rs.rand(n) < 0.26)
        frame = pd.DataFrame({"a": np.where(a, np.nan, 1.0), "b": np.where(b, np.nan, 1.0)})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        assert result.metrics["pairs"]["tested"] == 1
        assert result.metrics["pairs"]["above_threshold"] == 0

    def test_pairs_are_corrected_for_how_many_were_tested(self):
        # a and b overlap in 15 of 20 rows each (n=40): phi = 0.5, raw p = 0.0016. Among the 36
        # pairs of 9 columns that is not significant after the correction (0.0016 * 36 > 0.05).
        n = 40
        rs = np.random.RandomState(1)
        masks = {"a": np.arange(n) < 20, "b": (np.arange(n) >= 5) & (np.arange(n) < 25)}
        for k in range(7):
            masks[f"n{k}"] = np.isin(np.arange(n), rs.choice(n, 20, replace=False))
        frame = pd.DataFrame({name: np.where(mask, np.nan, 1.0) for name, mask in masks.items()})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        assert result.metrics["pairs"]["tested"] == 36
        assert result.metrics["pairs"]["above_threshold"] == 0

    def test_pairs_with_too_few_expected_counts_are_skipped_not_reported(self):
        a = [np.nan] * 3 + [1.0] * 97
        frame = pd.DataFrame({"a": a, "b": a})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        assert result.metrics["pairs"]["tested"] == 0
        assert result.metrics["pairs"]["skipped_small_counts"] == 1
        assert result.metrics["pairs"]["above_threshold"] == 0


class TestDependence:
    def test_missingness_that_depends_on_a_numeric_column_is_detected(self, result, columns):
        (predictor, *_) = columns["depends"]["predictors"]
        (finding,) = [f for f in findings_for(result, "depends") if "depends on" in f.title]

        assert columns["depends"]["dependence"] == "detected"
        assert (predictor["name"], predictor["kind"]) == ("x", "numeric")
        assert predictor["effect"] > 2  # the missing rows are the top quarter of x
        assert (finding.severity, finding.confidence) == (Severity.MEDIUM, 0.8)

    def test_missingness_that_depends_on_a_category_is_detected_with_v_of_one(self, columns):
        (predictor, *_) = columns["by_region"]["predictors"]

        assert columns["by_region"]["dependence"] == "detected"
        assert (predictor["name"], predictor["kind"]) == ("region", "categorical")
        assert predictor["effect"] == pytest.approx(1.0)

    def test_missingness_placed_at_random_is_not_detected(self, columns):
        assert columns["mcar30"]["dependence"] == "not_detected"
        assert columns["mcar30"]["comparisons_run"] > 0
        assert columns["mcar30"]["predictors"] == []

    def test_a_significant_but_tiny_difference_is_not_reported(self):
        rs = np.random.RandomState(0)
        n = 20000
        x = rs.normal(size=n)
        missing = rs.rand(n) < 0.30
        x[missing] += 0.06  # about 0.06 standard deviations: significant at this size, tiny
        m = np.where(missing, np.nan, 1.0)
        frame = pd.DataFrame({"x": x, "m": m})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        column = next(c for c in result.metrics["columns"] if c["name"] == "m")
        assert column["dependence"] == "not_detected"

    def test_a_significant_but_tiny_category_association_is_not_reported(self):
        # Missing 26% to 38% by region: Cramer's V is about 0.06, yet chi-square is overwhelming.
        rs = np.random.RandomState(0)
        n = 20000
        region = rs.choice(["n", "s", "e", "w"], size=n)
        chance = pd.Series({"n": 0.26, "s": 0.30, "e": 0.34, "w": 0.38})[region].to_numpy()
        missing = rs.rand(n) < chance
        frame = pd.DataFrame({"region": region, "m": np.where(missing, np.nan, 1.0)})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        column = next(c for c in result.metrics["columns"] if c["name"] == "m")
        assert column["dependence"] == "not_detected"

    def test_a_difference_that_is_only_significant_before_correction_is_not_reported(self):
        # d is exactly 0.8 with 25 rows per group: raw p is about 0.007, but 15 comparisons run.
        rs = np.random.RandomState(0)
        z = rs.normal(size=25)
        z = (z - z.mean()) / z.std(ddof=1)
        frame = pd.DataFrame({f"noise{k}": rs.normal(size=50) for k in range(14)})
        frame["signal"] = np.concatenate([z + 0.8, z])
        frame["m"] = [np.nan] * 25 + [1.0] * 25

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        column = next(c for c in result.metrics["columns"] if c["name"] == "m")
        assert column["comparisons_run"] == 15
        assert column["dependence"] == "not_detected"

    def test_the_effect_size_does_not_depend_on_where_the_values_sit(self):
        rs = np.random.RandomState(0)
        x = rs.normal(size=N)
        m = np.where(x > np.quantile(x, 0.75), np.nan, 1.0)

        def effect(offset):
            frame = pd.DataFrame({"x": x + offset, "m": m})
            result = check_missingness(Dataset(data=frame, name="t"), CONFIG)
            column = next(c for c in result.metrics["columns"] if c["name"] == "m")
            return column["predictors"][0]["effect"]

        assert effect(1e9) == pytest.approx(effect(0.0), rel=1e-6)


class TestColumnsThatCouldNotBeTested:
    def test_they_are_reported_with_their_reasons(self):
        frame = pd.DataFrame(
            {
                "x": np.random.RandomState(0).normal(size=100),
                "few": [np.nan] * 5 + [1.0] * 95,  # exactly 5%, the lowest rate that is reported
                "nothing_observed": [np.nan] * 100,
            }
        )

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        (finding,) = [f for f in result.findings if f.severity is Severity.INFO]
        assert finding.title == "Some columns could not be tested for dependence"
        assert "2 of 2 columns with a missing rate of 5% or more" in finding.evidence
        assert "under 20 in a group" in finding.evidence
        assert "every value is missing" in finding.evidence
        assert finding.affected_columns == ("few", "nothing_observed")
        labels = {c["name"]: c["dependence_reason"] for c in result.metrics["columns"]}
        assert labels["few"] == "small_group"
        assert labels["nothing_observed"] == "all_missing"

    def test_columns_below_the_rate_threshold_are_neither_named_nor_counted(self):
        frame = pd.DataFrame(
            {
                "x": np.random.RandomState(0).normal(size=100),
                "tiny": [np.nan] * 2 + [1.0] * 98,  # 2%: untestable, but too little to report
                "few": [np.nan] * 5 + [1.0] * 95,
            }
        )

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        (finding,) = [f for f in result.findings if f.severity is Severity.INFO]
        assert "1 of 1 columns" in finding.evidence
        assert finding.affected_columns == ("few",)
        labels = {c["name"]: c["dependence"] for c in result.metrics["columns"]}
        assert labels["tiny"] == "not_tested"  # still visible in the metrics

    def test_a_lone_numeric_column_has_nothing_to_compare_against(self):
        frame = pd.DataFrame({"only": [np.nan, 1.0] * 30})

        result = check_missingness(Dataset(data=frame, name="t"), CONFIG)

        column = result.metrics["columns"][0]
        assert (column["dependence"], column["dependence_reason"]) == (
            "not_tested",
            "no_predictors",
        )

    def test_nothing_is_reported_when_every_column_was_tested(self, result):
        assert not [f for f in result.findings if f.severity is Severity.INFO]


class TestLiteralTokens:
    def write(self, tmp_path, rows):
        path = tmp_path / "tokens.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
        return load_dataset(path)

    def test_na_in_a_column_of_short_uppercase_codes_is_flagged(self, tmp_path):
        codes = ["US", "FR", "DE", "JP"]
        rows = ["country,n"] + [
            f"{'NA' if k % 3 == 0 and k < 42 else codes[k % 4]},{k}" for k in range(60)
        ]

        result = check_missingness(self.write(tmp_path, rows), CONFIG)

        (finding,) = [f for f in result.findings if f.title == "Literal token read as missing"]
        assert (finding.severity, finding.confidence) == (Severity.MEDIUM, 0.6)
        assert finding.affected_columns == ("country",)
        assert 'contained "NA" in 14 rows' in finding.evidence

    def test_other_columns_and_other_tokens_are_left_alone(self, tmp_path):
        rows = ["city,score,status,lower,n"]
        for k in range(40):
            city = "NA" if k % 4 == 0 else ["Paris", "Rome", "Oslo"][k % 3]
            score = "NA" if k % 5 == 0 else str(k)
            status = ["null", "None", "n/a"][k % 3] if k % 2 == 0 else ["OK", "NO"][k % 2 - 1]
            lower = "NA" if k % 4 == 0 else ["us", "fr"][k % 2]
            rows.append(f"{city},{score},{status},{lower},{k}")

        result = check_missingness(self.write(tmp_path, rows), CONFIG)

        assert not [f for f in result.findings if f.title == "Literal token read as missing"]


class TestGuardrails:
    def frame(self, n_rows=300, n_columns=5):
        rs = np.random.RandomState(0)
        frame = pd.DataFrame({f"c{i}": rs.normal(size=n_rows) for i in range(n_columns)})
        frame.loc[::7, "c0"] = np.nan
        return frame

    def test_rates_use_every_row_and_the_pairwise_parts_use_the_sample(self):
        frame = self.frame()
        config = AnalysisConfig(row_threshold=100)

        result = check_missingness(Dataset(data=frame, name="t"), config)

        assert result.metrics["columns"][0]["n_missing"] == int(frame["c0"].isna().sum())
        assert result.metrics["n_rows"] == 300
        assert result.metrics["pairwise"] == {"performed": True, "rows_used": 100, "sampled": True}
        assert [f.category for f in result.findings if f.category == "guardrail"] == ["guardrail"]

    def test_too_many_columns_skip_the_pairwise_parts_and_say_so_without_a_sampling_note(self):
        frame = self.frame()
        config = AnalysisConfig(row_threshold=100, column_threshold=3)

        result = check_missingness(Dataset(data=frame, name="t"), config)

        guardrail = [f for f in result.findings if f.category == "guardrail"]
        assert [f.title for f in guardrail] == ["Pairwise analyses were skipped"]
        assert result.metrics["pairwise"]["performed"] is False
        assert result.metrics["pairs"]["above_threshold"] == 0
        column = result.metrics["columns"][0]
        assert (column["dependence"], column["dependence_reason"]) == (
            "not_tested",
            "pairwise_skipped",
        )
        assert not [
            f for f in result.findings if f.severity is Severity.INFO and f not in guardrail
        ]

    def test_no_limit_reached_means_no_guardrail_finding(self, result):
        assert not [f for f in result.findings if f.category == "guardrail"]


class TestResult:
    def test_the_config_is_recorded_and_the_result_round_trips(self):
        config = AnalysisConfig(random_seed=7, row_threshold=1500)
        result = check_missingness(Dataset(data=planted_frame(), name="t"), config)

        assert result.config == config
        assert AnalysisResult.from_json(result.to_json()) == result

    def test_the_same_input_gives_the_same_result(self, result):
        again = check_missingness(Dataset(data=planted_frame(), name="planted"), CONFIG)

        assert again == result
