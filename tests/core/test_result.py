import dataclasses
import math

import pytest

from datadoctor.core.result import Finding, Severity, sort_findings


def make_finding(**overrides) -> Finding:
    fields = {
        "category": "data_quality",
        "severity": Severity.HIGH,
        "confidence": 0.94,
        "title": "High missingness detected",
        "evidence": "Column income contains 38.4% missing values.",
        "interpretation": "Imputation may bias estimates if missingness is not random.",
        "limitations": "Missingness mechanism was not tested against external data.",
        "affected_columns": ["income"],
        "recommendation": "Investigate the missingness mechanism before modeling.",
    }
    fields.update(overrides)
    return Finding(**fields)


class TestSeverity:
    def test_members_and_order(self):
        assert [s.name for s in Severity] == ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    def test_is_a_string_enum(self):
        assert Severity.HIGH == "high"
        assert Severity("info") is Severity.INFO

    def test_rank_is_strictly_decreasing_from_critical_to_info(self):
        ranks = [s.rank for s in Severity]
        assert ranks == sorted(ranks, reverse=True)
        assert len(set(ranks)) == len(ranks)

    def test_rank_does_not_follow_alphabetical_order(self):
        # "info" < "low" as strings, but INFO is the less severe of the two.
        assert Severity.INFO.rank < Severity.LOW.rank
        assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.MEDIUM.rank


class TestFindingConstruction:
    def test_full_example_from_the_result_contract(self):
        finding = make_finding()

        assert finding.category == "data_quality"
        assert finding.severity is Severity.HIGH
        assert finding.confidence == 0.94
        assert finding.affected_columns == ("income",)
        assert finding.recommendation == "Investigate the missingness mechanism before modeling."

    def test_optional_fields_default_to_empty(self):
        finding = Finding(
            category="size",
            severity=Severity.INFO,
            confidence=1.0,
            title="Rows sampled",
            evidence="Dataset has 2,000,000 rows; sampled 100,000.",
            interpretation="Statistics are computed on a sample.",
            limitations="Rare categories may be under-represented.",
        )

        assert finding.affected_columns == ()
        assert finding.recommendation is None

    def test_affected_columns_list_is_stored_as_tuple(self):
        finding = make_finding(affected_columns=["a", "b"])

        assert finding.affected_columns == ("a", "b")

    def test_severity_and_confidence_are_separate_fields(self):
        names = {f.name for f in dataclasses.fields(Finding)}

        assert {"severity", "confidence"} <= names
        assert not {"score", "priority", "risk"} & names

    @pytest.mark.parametrize("missing", ["evidence", "interpretation", "limitations"])
    def test_evidence_interpretation_and_limitations_are_required(self, missing):
        fields = dataclasses.asdict(make_finding())
        del fields[missing]

        with pytest.raises(TypeError):
            Finding(**fields)

    def test_positional_construction_is_rejected(self):
        with pytest.raises(TypeError):
            Finding("data_quality", Severity.HIGH, 0.9, "t", "e", "i", "l")


class TestFindingValidation:
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_inside_range_is_accepted(self, confidence):
        assert make_finding(confidence=confidence).confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf])
    def test_confidence_outside_range_is_rejected(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            make_finding(confidence=confidence)

    @pytest.mark.parametrize("severity", ["high", 3, None])
    def test_severity_must_be_a_severity_member(self, severity):
        with pytest.raises(TypeError, match="severity"):
            make_finding(severity=severity)

    def test_single_string_is_not_split_into_characters(self):
        with pytest.raises(TypeError, match="affected_columns"):
            make_finding(affected_columns="income")

    def test_finding_is_immutable(self):
        finding = make_finding()

        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.severity = Severity.LOW


class TestSortFindings:
    def test_all_five_severities_come_out_most_severe_first(self):
        shuffled = [
            make_finding(severity=Severity.LOW, title="low"),
            make_finding(severity=Severity.CRITICAL, title="critical"),
            make_finding(severity=Severity.INFO, title="info"),
            make_finding(severity=Severity.HIGH, title="high"),
            make_finding(severity=Severity.MEDIUM, title="medium"),
        ]

        result = sort_findings(shuffled)

        assert [f.title for f in result] == ["critical", "high", "medium", "low", "info"]

    def test_equal_severity_keeps_input_order(self):
        findings = [make_finding(severity=Severity.MEDIUM, title=str(i)) for i in range(5)]

        assert [f.title for f in sort_findings(findings)] == ["0", "1", "2", "3", "4"]

    def test_confidence_does_not_influence_order(self):
        sure_but_minor = make_finding(severity=Severity.LOW, confidence=1.0, title="minor")
        unsure_but_major = make_finding(severity=Severity.HIGH, confidence=0.1, title="major")
        low_conf_first = make_finding(severity=Severity.MEDIUM, confidence=0.1, title="first")
        high_conf_second = make_finding(severity=Severity.MEDIUM, confidence=0.9, title="second")

        result = sort_findings([sure_but_minor, unsure_but_major, low_conf_first, high_conf_second])

        assert [f.title for f in result] == ["major", "first", "second", "minor"]

    def test_input_is_not_mutated_and_a_new_list_is_returned(self):
        findings = [
            make_finding(severity=Severity.INFO, title="info"),
            make_finding(severity=Severity.HIGH, title="high"),
        ]
        snapshot = list(findings)

        result = sort_findings(findings)

        assert findings == snapshot
        assert result is not findings

    def test_empty_input(self):
        assert sort_findings([]) == []
