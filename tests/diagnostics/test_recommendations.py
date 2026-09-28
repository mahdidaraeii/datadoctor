import pytest

from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.diagnostics.recommendations import Recommendation, build_recommendations

EDA_UNUSABLE_TITLE = "The target could not be compared against the other columns"
READINESS_UNUSABLE_TITLE = "Target column's type cannot be modeled"
EDA_IMBALANCE_TITLE = "The target classes are imbalanced"
STRATIFIED_TITLE = "Stratified splitting is recommended for this target"


def f(category, title, severity, confidence=1.0, columns=()):
    return Finding(
        category=category,
        severity=severity,
        confidence=confidence,
        title=title,
        evidence="evidence",
        interpretation="interpretation",
        limitations="limitations",
        affected_columns=columns,
    )


def build(*, quality=(), eda=(), splits=(), leakage=(), readiness=()):
    return build_recommendations(
        quality=AnalysisResult(findings=quality),
        eda=AnalysisResult(findings=eda),
        splits=AnalysisResult(findings=splits),
        leakage=AnalysisResult(findings=leakage),
        readiness=AnalysisResult(findings=readiness),
    )


class TestTargetUnusableMerge:
    def test_eda_and_readiness_target_unusable_merge_keeping_the_higher_severity(self):
        eda_finding = f("eda", EDA_UNUSABLE_TITLE, Severity.INFO, columns=("target",))
        readiness_finding = f(
            "readiness", READINESS_UNUSABLE_TITLE, Severity.CRITICAL, columns=("target",)
        )

        (rec,) = build(eda=(eda_finding,), readiness=(readiness_finding,))

        assert rec.severity == Severity.CRITICAL
        assert rec.columns == ("target",)
        assert set(rec.findings) == {eda_finding, readiness_finding}


class TestImbalanceMerge:
    def test_eda_imbalance_and_splits_stratification_merge(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("quality",))
        splits_finding = f(
            "split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("quality",)
        )

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        assert rec.severity == Severity.MEDIUM
        assert set(rec.findings) == {eda_finding, splits_finding}

    def test_same_titles_but_different_columns_do_not_merge(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("target_a",))
        splits_finding = f(
            "split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("target_b",)
        )

        recs = build(eda=(eda_finding,), splits=(splits_finding,))

        assert len(recs) == 2
        assert {f.title for r in recs for f in r.findings} == {
            EDA_IMBALANCE_TITLE,
            STRATIFIED_TITLE,
        }


class TestUnrelatedFindingsStaySeparate:
    def test_a_missingness_finding_and_an_outlier_finding_on_the_same_column_stay_two_entries(
        self,
    ):
        # Real titles from quality.missingness_findings and quality.outliers_findings, both
        # naming the same column, neither in any registered group: sharing a column is not
        # sharing a root cause.
        missingness_finding = f(
            "missingness", "High missingness in income", Severity.HIGH, columns=("income",)
        )
        outlier_finding = f(
            "outliers",
            "Outlier candidates in numeric columns",
            Severity.LOW,
            columns=("income",),
        )

        recs = build(quality=(missingness_finding, outlier_finding))

        assert len(recs) == 2


class TestHeadlineExclusion:
    def test_the_readiness_headline_never_appears_regardless_of_its_dynamic_text(self):
        for headline_title in (
            "No blocking issues were found by these checks",
            "Readiness was not assessed",
            "1 blocking issue found",
            "3 blocking issues found",
        ):
            headline = f("readiness", headline_title, Severity.CRITICAL)

            recs = build(readiness=(headline,))

            assert recs == ()

    def test_readiness_target_unusable_is_kept_when_alone(self):
        unusable = f("readiness", READINESS_UNUSABLE_TITLE, Severity.CRITICAL, columns=("y",))

        (rec,) = build(readiness=(unusable,))

        assert rec.findings == (unusable,)


class TestExactDuplicateCollapse:
    def test_a_byte_identical_finding_from_two_sources_collapses_to_one(self):
        guardrail = f("guardrail", "Analysis ran on a sample of the rows", Severity.INFO)

        recs = build(splits=(guardrail,), leakage=(guardrail,))

        assert len(recs) == 1
        assert recs[0].findings == (guardrail,)


class TestConfidencePairing:
    def test_the_higher_severity_findings_confidence_wins_even_if_lower(self):
        low_severity_high_confidence = f(
            "eda", EDA_UNUSABLE_TITLE, Severity.INFO, confidence=1.0, columns=("target",)
        )
        high_severity_low_confidence = f(
            "readiness",
            READINESS_UNUSABLE_TITLE,
            Severity.CRITICAL,
            confidence=0.4,
            columns=("target",),
        )

        (rec,) = build(
            eda=(low_severity_high_confidence,), readiness=(high_severity_low_confidence,)
        )

        assert rec.severity == Severity.CRITICAL
        assert rec.confidence == 0.4

    def test_a_tie_in_severity_takes_the_higher_confidence(self):
        lower_confidence = f(
            "eda", EDA_IMBALANCE_TITLE, Severity.MEDIUM, confidence=0.3, columns=("y",)
        )
        higher_confidence = f(
            "split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, confidence=0.9, columns=("y",)
        )

        (rec,) = build(eda=(lower_confidence,), splits=(higher_confidence,))

        assert rec.severity == Severity.MEDIUM
        assert rec.confidence == 0.9


class TestRanking:
    def test_exact_positions_for_a_mix_of_severities_and_a_confidence_tie_break(self):
        critical = f("leakage", "duplicate", Severity.CRITICAL, confidence=0.9)
        high_low_confidence = f("splits_x", "temporal", Severity.HIGH, confidence=0.4)
        high_high_confidence = f("splits_y", "grouped", Severity.HIGH, confidence=0.8)
        medium = f("quality", "outliers", Severity.MEDIUM, confidence=1.0)

        recs = build(
            leakage=(critical,),
            splits=(high_low_confidence, high_high_confidence),
            quality=(medium,),
        )

        assert [r.findings[0] for r in recs] == [
            critical,
            high_high_confidence,
            high_low_confidence,
            medium,
        ]


class TestSerialization:
    def test_a_singleton_recommendation_round_trips_through_strict_json(self):
        finding = f("outliers", "Outlier candidates in numeric columns", Severity.LOW)

        (rec,) = build(quality=(finding,))

        assert Recommendation.from_json(rec.to_json()) == rec

    def test_a_merged_recommendation_round_trips_through_strict_json(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",))
        splits_finding = f("split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("y",))

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        assert Recommendation.from_json(rec.to_json()) == rec
        assert len(Recommendation.from_json(rec.to_json()).findings) == 2


class TestValidation:
    def test_a_recommendation_needs_at_least_one_finding(self):
        with pytest.raises(ValueError, match="at least one finding"):
            Recommendation(severity=Severity.INFO, confidence=1.0, columns=(), findings=())


class TestNoInput:
    def test_no_findings_anywhere_gives_no_recommendations(self):
        assert build() == ()
