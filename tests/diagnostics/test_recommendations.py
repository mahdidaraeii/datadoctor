import pytest

from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.diagnostics.recommendations import (
    Recommendation,
    build_recommendations,
    drop_findings_superseded_by_recommendations,
)
from datadoctor.diagnostics.splits_findings import stratified_split_finding
from datadoctor.eda.relationships_findings import class_imbalance_finding

EDA_UNUSABLE_TITLE = "The target could not be compared against the other columns"
READINESS_UNUSABLE_TITLE = "Target column's type cannot be modeled"
EDA_IMBALANCE_TITLE = "The target classes are imbalanced"
STRATIFIED_TITLE = "Stratified splitting is recommended for this target"


def f(
    category,
    title,
    severity,
    confidence=1.0,
    columns=(),
    evidence="evidence",
    limitations="limitations",
    recommendation=None,
):
    return Finding(
        category=category,
        severity=severity,
        confidence=confidence,
        title=title,
        evidence=evidence,
        interpretation="interpretation",
        limitations=limitations,
        affected_columns=columns,
        recommendation=recommendation,
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


class TestToFinding:
    def test_a_singleton_flattens_with_its_own_fields_unchanged(self):
        finding = f(
            "outliers",
            "Outlier candidates in numeric columns",
            Severity.LOW,
            confidence=0.4,
            columns=("age",),
        )
        (rec,) = build(quality=(finding,))

        flattened = rec.to_finding()

        assert flattened.category == "recommendation"
        assert flattened.severity == Severity.LOW
        assert flattened.confidence == 0.4
        assert flattened.title == "Outlier candidates in numeric columns"
        assert flattened.evidence == "evidence"
        assert flattened.affected_columns == ("age",)

    def test_a_merge_joins_evidence_from_both_sources_and_takes_the_first_title(self):
        eda_finding = f(
            "eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",), evidence="eda evidence"
        )
        splits_finding = f(
            "split_strategy",
            STRATIFIED_TITLE,
            Severity.MEDIUM,
            columns=("y",),
            evidence="splits evidence",
        )

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        flattened = rec.to_finding()

        assert flattened.title == EDA_IMBALANCE_TITLE  # eda is read before splits
        assert flattened.evidence == "eda evidence; splits evidence"
        assert flattened.severity == Severity.MEDIUM  # the winning (higher) severity
        assert flattened.affected_columns == ("y",)

    def test_a_merge_keeps_duplicate_text_only_once(self):
        eda_finding = f(
            "eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",), limitations="same text"
        )
        splits_finding = f(
            "split_strategy",
            STRATIFIED_TITLE,
            Severity.MEDIUM,
            columns=("y",),
            limitations="same text",
        )

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        assert rec.to_finding().limitations == "same text"

    def test_no_recommendation_text_from_either_source_gives_none(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",))
        splits_finding = f("split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("y",))

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        assert rec.to_finding().recommendation is None


class TestJoinCollapsesNearRestates:
    """The wine_quality.csv fixture: a classification target whose smallest class ('9') holds
    0.1% of the rows. eda's and split_strategy's own finding builders, not hand-typed strings,
    so this stays true to what those analyzers actually say."""

    def eda_finding(self):
        return class_imbalance_finding("quality", 9, 0.0010208248264597796, Severity.MEDIUM)

    def splits_finding(self):
        return stratified_split_finding("quality", 0.0010208248264597796, Severity.MEDIUM)

    def test_recommendation_text_collapses_to_the_more_informative_sentence(self):
        (rec,) = build(eda=(self.eda_finding(),), splits=(self.splits_finding(),))

        assert rec.to_finding().recommendation == (
            "Use stratified sampling for train/test splits and cross-validation, and choose an "
            "evaluation metric that accounts for the imbalance."
        )

    def test_evidence_does_not_collapse_because_it_is_not_a_literal_substring(self):
        # eda names the minority label ('9') in the middle of the sentence, splits does not, so
        # neither text is a literal substring of the other even though both report the same
        # 0.1% fact. This is the one sentence in this fixture the substring rule cannot join.
        (rec,) = build(eda=(self.eda_finding(),), splits=(self.splits_finding(),))

        assert rec.to_finding().evidence == (
            "The smallest class '9' holds 0.1% of the rows.; "
            "The smallest class holds 0.1% of the rows."
        )

    def test_limitations_do_not_collapse_because_they_say_different_things(self):
        (rec,) = build(eda=(self.eda_finding(),), splits=(self.splits_finding(),))

        limitations = rec.to_finding().limitations
        assert "The severity thresholds here are conventions, not a judgment on this data." in (
            limitations
        )
        assert "The severity thresholds here are the same conventions used elsewhere." in (
            limitations
        )


class TestJoinSubstringRule:
    def test_the_shorter_text_is_dropped_regardless_of_which_source_gives_it(self):
        short_first = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",), evidence="Short.")
        long_second = f(
            "split_strategy",
            STRATIFIED_TITLE,
            Severity.MEDIUM,
            columns=("y",),
            evidence="Short, with more detail.",
        )
        (short_first_rec,) = build(eda=(short_first,), splits=(long_second,))

        long_first = f(
            "eda",
            EDA_IMBALANCE_TITLE,
            Severity.LOW,
            columns=("y",),
            evidence="Short, with more detail.",
        )
        short_second = f(
            "split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("y",), evidence="Short."
        )
        (long_first_rec,) = build(eda=(long_first,), splits=(short_second,))

        assert short_first_rec.to_finding().evidence == "Short, with more detail."
        assert long_first_rec.to_finding().evidence == "Short, with more detail."

    def test_a_trailing_period_only_difference_still_collapses(self):
        with_period = f(
            "eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",), evidence="Same fact."
        )
        without_period = f(
            "split_strategy",
            STRATIFIED_TITLE,
            Severity.MEDIUM,
            columns=("y",),
            evidence="Same fact",
        )

        (rec,) = build(eda=(with_period,), splits=(without_period,))

        assert rec.to_finding().evidence == "Same fact."

    def test_genuinely_different_text_is_never_touched(self):
        eda_finding = f(
            "eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",), evidence="Fact one."
        )
        splits_finding = f(
            "split_strategy",
            STRATIFIED_TITLE,
            Severity.MEDIUM,
            columns=("y",),
            evidence="A completely unrelated fact.",
        )

        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))

        assert rec.to_finding().evidence == "Fact one.; A completely unrelated fact."


class TestDropSupersededByRecommendations:
    def test_the_wine_imbalance_finding_appears_exactly_once_as_the_recommendation_version(self):
        eda_finding = class_imbalance_finding("quality", 9, 0.0010208248264597796, Severity.MEDIUM)
        splits_finding = stratified_split_finding("quality", 0.0010208248264597796, Severity.MEDIUM)
        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))
        flattened = rec.to_finding()

        # report's own combine step: profile/quality/eda's raw findings alongside diagnose's
        # already-merged recommendation views, in that order.
        kept = drop_findings_superseded_by_recommendations([eda_finding, flattened])

        assert kept == (flattened,)
        assert kept[0].category == "recommendation"

    def test_the_target_unusable_finding_also_collapses_to_its_recommendation_version(self):
        eda_finding = f("eda", EDA_UNUSABLE_TITLE, Severity.INFO, columns=("target",))
        readiness_finding = f(
            "readiness", READINESS_UNUSABLE_TITLE, Severity.CRITICAL, columns=("target",)
        )
        (rec,) = build(eda=(eda_finding,), readiness=(readiness_finding,))
        flattened = rec.to_finding()

        kept = drop_findings_superseded_by_recommendations([eda_finding, flattened])

        assert kept == (flattened,)

    def test_the_recommendation_itself_is_never_dropped_when_it_appears_alone(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",))
        splits_finding = f("split_strategy", STRATIFIED_TITLE, Severity.MEDIUM, columns=("y",))
        (rec,) = build(eda=(eda_finding,), splits=(splits_finding,))
        flattened = rec.to_finding()

        kept = drop_findings_superseded_by_recommendations([flattened])

        assert kept == (flattened,)

    def test_the_raw_finding_alone_is_untouched_when_no_recommendation_is_present(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("y",))

        kept = drop_findings_superseded_by_recommendations([eda_finding])

        assert kept == (eda_finding,)

    def test_a_column_mismatch_keeps_both(self):
        eda_finding = f("eda", EDA_IMBALANCE_TITLE, Severity.LOW, columns=("a",))
        # A recommendation-category finding for the same title but a different column: not the
        # same instance of the fact, so the raw finding must not be dropped.
        unrelated_recommendation = f(
            "recommendation", EDA_IMBALANCE_TITLE, Severity.MEDIUM, columns=("b",)
        )

        kept = drop_findings_superseded_by_recommendations([eda_finding, unrelated_recommendation])

        assert set(kept) == {eda_finding, unrelated_recommendation}

    def test_findings_outside_any_registered_group_are_never_touched(self):
        quality_finding = f(
            "outliers", "Outlier candidates in numeric columns", Severity.LOW, columns=("x",)
        )
        singleton_recommendation = f(
            "recommendation", "Outlier candidates in numeric columns", Severity.LOW, columns=("x",)
        )

        kept = drop_findings_superseded_by_recommendations(
            [quality_finding, singleton_recommendation]
        )

        assert set(kept) == {quality_finding, singleton_recommendation}


class TestValidation:
    def test_a_recommendation_needs_at_least_one_finding(self):
        with pytest.raises(ValueError, match="at least one finding"):
            Recommendation(severity=Severity.INFO, confidence=1.0, columns=(), findings=())


class TestNoInput:
    def test_no_findings_anywhere_gives_no_recommendations(self):
        assert build() == ()
