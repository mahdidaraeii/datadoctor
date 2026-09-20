import math
from pathlib import Path

import numpy as np
import pytest

from datadoctor.core.exceptions import SerializationError
from datadoctor.core.result import AnalysisResult, Severity


@pytest.fixture
def full_result(make_finding) -> AnalysisResult:
    return AnalysisResult(
        findings=[
            make_finding(severity=Severity.CRITICAL, title="Target leakage suspected"),
            make_finding(
                severity=Severity.INFO,
                confidence=1.0,
                title="Colonne « âge » échantillonnée",
                affected_columns=(),
                recommendation=None,
            ),
        ],
        metrics={
            "n_rows": 1000,
            "missing_rate": 0.384,
            "undefined_correlation": None,
            "per_column": {"income": {"missing": 384, "type": "numeric"}, "tags": ["a", "b"]},
            "flag": True,
        },
        artifacts={"missingness_bar": Path("outputs/plots/missingness.png")},
    )


class TestConstruction:
    def test_defaults_are_empty(self):
        result = AnalysisResult()

        assert result.findings == ()
        assert result.metrics == {}
        assert result.artifacts == {}

    def test_findings_list_is_stored_as_tuple_in_order(self, make_finding):
        first = make_finding(title="first", severity=Severity.LOW)
        second = make_finding(title="second", severity=Severity.CRITICAL)

        result = AnalysisResult(findings=[first, second])

        assert result.findings == (first, second)

    def test_artifact_strings_become_paths(self):
        result = AnalysisResult(artifacts={"plot": "outputs/a.png"})

        assert result.artifacts == {"plot": Path("outputs/a.png")}

    def test_metrics_dict_is_copied(self):
        metrics = {"n_rows": 10}

        result = AnalysisResult(metrics=metrics)
        metrics["n_rows"] = 99
        metrics["extra"] = 1

        assert result.metrics == {"n_rows": 10}

    def test_is_immutable(self):
        result = AnalysisResult()

        with pytest.raises(AttributeError):
            result.findings = ()


class TestValidation:
    def test_findings_must_be_finding_objects(self):
        with pytest.raises(TypeError, match="Finding"):
            AnalysisResult(findings=[{"title": "not a finding"}])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_metric_is_rejected_and_named(self, bad):
        with pytest.raises(SerializationError, match="'correlation'"):
            AnalysisResult(metrics={"correlation": bad})

    def test_nan_nested_inside_a_metric_is_rejected(self):
        with pytest.raises(SerializationError, match="'stats'"):
            AnalysisResult(metrics={"stats": {"mean": [1.0, math.nan]}})

    def test_numpy_scalar_is_rejected(self):
        with pytest.raises(SerializationError, match="'n_rows'"):
            AnalysisResult(metrics={"n_rows": np.int64(5)})

    def test_tuple_is_rejected_because_it_returns_as_a_list(self):
        with pytest.raises(SerializationError, match="round trip"):
            AnalysisResult(metrics={"pair": (1, 2)})

    def test_integer_dict_keys_are_rejected_because_they_return_as_strings(self):
        with pytest.raises(SerializationError, match="round trip"):
            AnalysisResult(metrics={"counts": {1: "a"}})

    def test_metric_names_must_be_strings(self):
        with pytest.raises(TypeError, match="metric names"):
            AnalysisResult(metrics={1: 2.0})

    def test_artifact_names_must_be_strings(self):
        with pytest.raises(TypeError, match="artifact names"):
            AnalysisResult(artifacts={1: "a.png"})


class TestSerialization:
    def test_dict_round_trip(self, full_result):
        assert AnalysisResult.from_dict(full_result.to_dict()) == full_result

    def test_json_round_trip(self, full_result):
        assert AnalysisResult.from_json(full_result.to_json()) == full_result

    def test_empty_result_round_trips(self):
        assert AnalysisResult.from_json(AnalysisResult().to_json()) == AnalysisResult()

    def test_artifact_paths_are_stored_with_forward_slashes(self, full_result):
        assert full_result.to_dict()["artifacts"] == {
            "missingness_bar": "outputs/plots/missingness.png"
        }

    def test_finding_order_survives_the_round_trip(self, full_result):
        restored = AnalysisResult.from_json(full_result.to_json())

        assert [f.severity for f in restored.findings] == [Severity.CRITICAL, Severity.INFO]

    def test_to_dict_does_not_alias_the_metrics(self, full_result):
        exported = full_result.to_dict()
        exported["metrics"]["per_column"]["income"]["missing"] = -1

        assert full_result.metrics["per_column"]["income"]["missing"] == 384

    def test_nan_added_after_construction_is_caught_at_serialization(self):
        result = AnalysisResult(metrics={"rate": 0.5})
        result.metrics["rate"] = math.nan

        with pytest.raises(SerializationError, match="cannot serialize AnalysisResult"):
            result.to_json()

    def test_unknown_key_is_rejected(self):
        with pytest.raises(SerializationError, match="unknown keys: warnings"):
            AnalysisResult.from_dict({"warnings": []})

    def test_finding_with_a_missing_key_is_rejected(self):
        with pytest.raises(SerializationError, match="missing keys: limitations"):
            AnalysisResult.from_dict(
                {
                    "findings": [
                        {
                            "category": "c",
                            "severity": "low",
                            "confidence": 0.5,
                            "title": "t",
                            "evidence": "e",
                            "interpretation": "i",
                        }
                    ]
                }
            )
