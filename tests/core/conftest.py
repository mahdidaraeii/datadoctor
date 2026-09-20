import pytest

from datadoctor.core.result import Finding, Severity


@pytest.fixture
def make_finding():
    def factory(**overrides) -> Finding:
        fields = {
            "category": "data_quality",
            "severity": Severity.HIGH,
            "confidence": 0.94,
            "title": "High missingness detected",
            "evidence": "Column income contains 38.4% missing values.",
            "interpretation": "Imputation may bias estimates if missingness is not random.",
            "limitations": "Missingness mechanism was not tested against external data.",
            "affected_columns": ("income",),
            "recommendation": "Investigate the missingness mechanism before modeling.",
        }
        fields.update(overrides)
        return Finding(**fields)

    return factory
