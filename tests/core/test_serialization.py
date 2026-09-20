import json
from pathlib import Path

import pytest

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.exceptions import SerializationError
from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.core.serialization import ensure_json_roundtrip, require_keys


def strict_loads(text: str):
    """Parse like a standards-compliant parser: NaN and Infinity tokens are errors."""

    def reject(token: str):
        raise AssertionError(f"non-standard JSON constant in output: {token}")

    return json.loads(text, parse_constant=reject)


@pytest.fixture
def instances(make_finding):
    finding = make_finding(title="Colonne « âge » incohérente")
    return [
        finding,
        AnalysisConfig(random_seed=3, row_threshold=10, column_threshold=5),
        AnalysisResult(
            findings=[finding],
            metrics={"rate": 0.5, "undefined": None, "nested": {"xs": [1, 2.5, None]}},
            artifacts={"plot": Path("outputs/plot.png")},
        ),
    ]


class TestStrictJsonOutput:
    def test_output_parses_with_a_strict_parser(self, instances):
        for instance in instances:
            assert strict_loads(instance.to_json()) == instance.to_dict()

    def test_compact_output_parses_with_a_strict_parser(self, instances):
        for instance in instances:
            text = instance.to_json(indent=None)
            assert "\n" not in text
            assert strict_loads(text) == instance.to_dict()

    def test_round_trip_is_lossless_for_every_type(self, instances):
        for instance in instances:
            assert type(instance).from_json(instance.to_json()) == instance

    def test_output_is_deterministic(self, instances):
        for instance in instances:
            assert instance.to_json() == instance.to_json()
            assert instance.to_json() == type(instance).from_json(instance.to_json()).to_json()

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_constants_are_rejected_on_input(self, token):
        with pytest.raises(SerializationError, match="not valid JSON"):
            AnalysisResult.from_json(f'{{"metrics": {{"x": {token}}}}}')

    @pytest.mark.parametrize("text", ["", "{", "not json", '{"metrics": }'])
    def test_malformed_text_is_rejected(self, text):
        with pytest.raises(SerializationError, match="invalid JSON"):
            AnalysisConfig.from_json(text)

    @pytest.mark.parametrize("text", ["[]", "3", '"text"', "null"])
    def test_top_level_must_be_an_object(self, text):
        with pytest.raises(SerializationError, match="JSON object"):
            AnalysisConfig.from_json(text)


class TestFindingSerialization:
    def test_dict_round_trip(self, make_finding):
        finding = make_finding()

        assert Finding.from_dict(finding.to_dict()) == finding

    def test_severity_is_stored_as_its_string_value(self, make_finding):
        assert make_finding(severity=Severity.MEDIUM).to_dict()["severity"] == "medium"

    def test_optional_fields_may_be_omitted(self):
        finding = Finding.from_dict(
            {
                "category": "size",
                "severity": "info",
                "confidence": 1.0,
                "title": "t",
                "evidence": "e",
                "interpretation": "i",
                "limitations": "l",
            }
        )

        assert finding.affected_columns == ()
        assert finding.recommendation is None

    def test_unknown_severity_is_rejected(self, make_finding):
        data = make_finding().to_dict() | {"severity": "catastrophic"}

        with pytest.raises(SerializationError, match="catastrophic"):
            Finding.from_dict(data)

    def test_severity_is_case_sensitive(self, make_finding):
        data = make_finding().to_dict() | {"severity": "HIGH"}

        with pytest.raises(SerializationError):
            Finding.from_dict(data)

    def test_missing_required_keys_are_all_named(self, make_finding):
        data = make_finding().to_dict()
        del data["evidence"]
        del data["limitations"]

        with pytest.raises(SerializationError, match="missing keys: evidence, limitations"):
            Finding.from_dict(data)

    def test_unknown_key_is_rejected(self, make_finding):
        data = make_finding().to_dict() | {"score": 0.9}

        with pytest.raises(SerializationError, match="unknown keys: score"):
            Finding.from_dict(data)

    def test_out_of_range_confidence_is_still_rejected(self, make_finding):
        data = make_finding().to_dict() | {"confidence": 1.5}

        with pytest.raises(ValueError, match="confidence"):
            Finding.from_dict(data)


class TestHelpers:
    def test_require_keys_accepts_a_valid_mapping(self):
        require_keys("Thing", {"a": 1, "b": 2}, required=["a"], optional=["b"])

    def test_require_keys_rejects_non_mappings(self):
        with pytest.raises(SerializationError, match="Thing must be a JSON object"):
            require_keys("Thing", [("a", 1)], required=["a"])

    @pytest.mark.parametrize(
        "value", [1, 2.5, "text", None, True, [1, [2]], {"a": {"b": [None, 1.5]}}]
    )
    def test_ensure_json_roundtrip_accepts_plain_json_values(self, value):
        ensure_json_roundtrip("value", value)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), (1, 2), {1: "a"}, {"a": {2, 3}}, Path("x"), object()]
    )
    def test_ensure_json_roundtrip_rejects_everything_else(self, value):
        with pytest.raises(SerializationError):
            ensure_json_roundtrip("value", value)
