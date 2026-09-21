import datetime

import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisResult, Dataset, Severity
from datadoctor.profiling.schema import SemanticType, profile_schema

N = 60


def planted_frame() -> pd.DataFrame:
    """60 rows with one planted column per case. Expected results are listed below."""
    rows = range(N)
    start = datetime.date(2024, 1, 1)
    return pd.DataFrame(
        {
            # identifier candidates
            "customer_id": [1000 + k for k in rows],
            "account_id": [(k * 7919 + 13) % 100003 for k in rows],
            "legacy_id": [np.nan if k in (3, 10) else float(k + 1) for k in rows],
            "reading_key": [k * 1.37 + 0.5 for k in rows],
            "units_sold": [(k * 104729 + 7) % 999983 for k in rows],
            "row_number": [k + 1 for k in rows],
            "parent_id": [k % 5 for k in rows],
            "ticket_id": [k % 58 for k in rows],  # 58 of 60 distinct: near-unique, not unique
            "order_ref": [f"ORD-{k:05d}" for k in rows],
            # numbers
            "age": [20 + k % 30 for k in rows],
            "income": [30000 + k * 137.31 for k in rows],
            "zero_one": [k % 2 for k in rows],
            "score": [None if k % 15 == 0 else (k % 10) + 0.5 for k in rows],
            # strings
            "city": [["Oslo", "Paris", "Rome", "Lima", "Kyiv"][k % 5] for k in rows],
            "sku": [f"SKU-{k % 21:02d}" for k in rows],
            "comment": [f"note {k} about the order that shipped late" for k in rows],
            "zip_code": [["01234", "02139", "10001", "90210", "60614"][k % 5] for k in rows],
            "year_text": [["1999", "2024", "2010"][k % 3] for k in rows],
            "us_date": [["03/04/2024", "04/03/2024"][k % 2] for k in rows],
            "optin": [["yes", "no"][k % 2] for k in rows],
            # dates
            "signup_date": [(start + datetime.timedelta(days=k)).isoformat() for k in rows],
            "event_time": ["2024-01-05 10:30:00" if k % 2 else "2024-01-05T10:30:59" for k in rows],
            "last_login": pd.date_range("2024-01-01", periods=N),
            # other dtypes
            "active": [k % 3 != 0 for k in rows],
            "tier": pd.Series(
                [["gold", "silver", "bronze"][k % 3] for k in rows], dtype="category"
            ),
            # unclassifiable
            "empty": [np.nan] * N,
            "nested": [[k] for k in rows],
            "mixed": [k if k % 2 else str(k) for k in rows],
        }
    )


EXPECTED_TYPES = {
    "customer_id": "identifier",
    "account_id": "identifier",
    "legacy_id": "identifier",
    "reading_key": "numeric",
    "units_sold": "numeric",
    "row_number": "identifier",
    "parent_id": "numeric",
    "ticket_id": "identifier",
    "order_ref": "identifier",
    "age": "numeric",
    "income": "numeric",
    "zero_one": "numeric",
    "score": "numeric",
    "city": "categorical",
    "sku": "categorical",
    "comment": "text",
    "zip_code": "categorical",
    "year_text": "categorical",
    "us_date": "categorical",
    "optin": "categorical",
    "signup_date": "datetime",
    "event_time": "datetime",
    "last_login": "datetime",
    "active": "boolean",
    "tier": "categorical",
    "empty": "unknown",
    "nested": "unknown",
    "mixed": "unknown",
}

EXPECTED_IDENTIFIER_CONFIDENCE = {
    "customer_id": 0.95,
    "account_id": 0.95,
    "legacy_id": 0.95,
    "ticket_id": 0.95,
    "row_number": 0.6,
    "order_ref": 0.6,
    "parent_id": 0.4,
}


# Whether the name matched, for the identifier candidates. Only row_number and order_ref are
# candidates on their values alone. parent_id is a candidate because of its name alone.
EXPECTED_IDENTIFIER_NAMED = {
    "customer_id": True,
    "account_id": True,
    "legacy_id": True,
    "ticket_id": True,
    "row_number": False,
    "order_ref": False,
    "parent_id": True,
}


@pytest.fixture(scope="module")
def result() -> AnalysisResult:
    return profile_schema(Dataset(data=planted_frame(), name="planted"))


@pytest.fixture(scope="module")
def columns(result) -> dict:
    return {column["name"]: column for column in result.metrics["columns"]}


class TestSemanticTypes:
    def test_every_planted_column_gets_its_expected_type(self, columns):
        assert {name: c["semantic_type"] for name, c in columns.items()} == EXPECTED_TYPES

    def test_columns_are_reported_in_file_order(self, result):
        assert [c["name"] for c in result.metrics["columns"]] == list(planted_frame().columns)

    def test_counts(self, result, columns):
        assert result.metrics["n_rows"] == N
        assert result.metrics["n_columns"] == len(EXPECTED_TYPES)
        assert (columns["score"]["null_count"], columns["score"]["cardinality"]) == (4, 10)
        assert (columns["legacy_id"]["null_count"], columns["legacy_id"]["cardinality"]) == (2, 58)
        assert (columns["empty"]["null_count"], columns["empty"]["cardinality"]) == (N, 0)
        assert columns["nested"]["cardinality"] is None
        assert columns["city"]["cardinality"] == 5


class TestIdentifiers:
    def test_confidence_follows_the_evidence_tiers(self, columns):
        found = {
            name: c["identifier_confidence"]
            for name, c in columns.items()
            if c["identifier_confidence"] is not None
        }

        assert found == EXPECTED_IDENTIFIER_CONFIDENCE

    def test_the_column_name_is_reported_as_evidence_for_the_candidates_only(self, columns):
        named = {name: c["identifier_named"] for name, c in columns.items()}

        assert {n: v for n, v in named.items() if v is not None} == EXPECTED_IDENTIFIER_NAMED
        assert all(
            (c["identifier_named"] is None) == (c["identifier_confidence"] is None)
            for c in columns.values()
        )

    def test_every_candidate_gets_a_finding_with_the_same_confidence(self, result):
        by_column = {f.affected_columns[0]: f for f in result.findings}

        assert {name: f.confidence for name, f in by_column.items()} == (
            EXPECTED_IDENTIFIER_CONFIDENCE
        )
        assert all(f.severity is Severity.LOW and f.category == "schema" for f in result.findings)
        assert all(f.confidence < 1.0 for f in result.findings)

    def test_finding_keeps_evidence_interpretation_and_limitations_apart(self, result):
        finding = next(f for f in result.findings if f.affected_columns == ("customer_id",))

        assert "60 distinct values in 60 non-null rows" in finding.evidence
        assert "heuristic" in finding.interpretation
        assert "English-centric" in finding.limitations

    def test_unique_non_sequential_integers_are_flagged_only_when_the_name_says_identifier(self):
        values = [(k * 7919 + 13) % 100003 for k in range(N)]
        frame = pd.DataFrame({"account_id": values, "units_sold": values})

        columns = {
            c["name"]: c for c in profile_schema(Dataset(data=frame, name="t")).metrics["columns"]
        }

        assert columns["account_id"]["identifier_confidence"] == 0.95
        assert columns["units_sold"]["identifier_confidence"] is None
        assert columns["units_sold"]["semantic_type"] == "numeric"

    def test_name_patterns_match_identifier_words_but_not_look_alikes(self):
        values = [(k * 7919 + 13) % 100003 for k in range(N)]
        matching = ["ID", "user_id", "user-id", "user id", "customerId", "customerID", "uuid"]
        matching += ["guid", "api_key"]
        look_alikes = ["paid", "monkey", "valid", "grid", "identity", "Idle"]
        frame = pd.DataFrame({name: values for name in matching + look_alikes})

        columns = {
            c["name"]: c for c in profile_schema(Dataset(data=frame, name="t")).metrics["columns"]
        }
        flagged = {n for n, c in columns.items() if c["identifier_confidence"] is not None}

        assert flagged == set(matching)

    def test_too_few_rows_give_no_uniqueness_signal(self):
        frame = pd.DataFrame({"id": range(10), "code": [f"c{k}" for k in range(10)]})

        columns = {
            c["name"]: c for c in profile_schema(Dataset(data=frame, name="t")).metrics["columns"]
        }

        assert columns["id"]["identifier_confidence"] == 0.4
        assert columns["id"]["semantic_type"] == "numeric"
        assert columns["code"]["identifier_confidence"] is None
        assert columns["code"]["semantic_type"] == "categorical"

    def test_target_is_never_an_identifier(self):
        dataset = Dataset(data=planted_frame(), name="planted", target="customer_id")

        result = profile_schema(dataset)

        column = next(c for c in result.metrics["columns"] if c["name"] == "customer_id")
        assert column["semantic_type"] == "numeric"
        assert column["identifier_confidence"] is None
        assert all(f.affected_columns != ("customer_id",) for f in result.findings)


class TestResultShape:
    def test_result_is_deterministic_and_survives_json(self, result):
        again = profile_schema(Dataset(data=planted_frame(), name="planted"))

        assert again.to_json() == result.to_json()
        assert AnalysisResult.from_json(result.to_json()) == result

    def test_non_string_column_labels_are_reported_as_text(self):
        frame = pd.DataFrame({1: ["a", "b"], 2: [1, 2]})

        names = [
            c["name"] for c in profile_schema(Dataset(data=frame, name="t")).metrics["columns"]
        ]

        assert names == ["1", "2"]


def type_of(values):
    frame = pd.DataFrame({"d": values})
    return profile_schema(Dataset(data=frame, name="t")).metrics["columns"][0]["semantic_type"]


class TestIsoTextDates:
    def test_far_future_dates_are_a_datetime_on_every_pandas_version(self):
        # pandas 2.3 cannot hold these as datetime64[ns] and turns them into NaT, while pandas 3.0
        # can. The type must not depend on which one is installed.
        values = (["9999-12-31", "3000-01-01", "2262-04-12", "2024-01-01"] * 8)[:30]

        assert type_of(values) == "datetime"

    def test_far_future_dates_with_times_and_offsets_are_a_datetime_too(self):
        values = [
            "9999-12-31T23:59:59Z",
            "3000-01-01 00:00:00+02:00",
            "2262-04-12T10:30:00.123456",
            "0001-06-30T12:00:00.123456789",  # nanosecond digits outside pandas' nanosecond range
            "2024-01-01T00:00",
        ] * 8

        assert type_of(values) == "datetime"

    def test_the_largest_valid_clock_and_offset_values_are_accepted(self):
        values = ["2024-01-05T23:59:59+23:59", "2024-01-05T23:59-2359", "2024-01-05 00:00:00Z"] * 10

        assert type_of(values) == "datetime"

    @pytest.mark.parametrize(
        "bad",
        [
            "2024-02-30",  # not a calendar date
            "2024-13-01",  # month 13
            "2024-01-05T24:00",  # hour 24, the shortest form with a time
            "2024-01-05T24:00:00",  # hour 24
            "2024-01-05T10:60:00",  # minute 60
            "2024-01-05T10:30:60",  # second 60
            "2024-01-05T10:30+24:00",  # offset hour 24
            "2024-01-05T10:30+02:60",  # offset minute 60
            "2024-01-05T10:30:00 +02:00",  # a space before the offset
        ],
    )
    def test_one_value_that_is_not_a_real_date_or_time_rules_the_column_out(self, bad):
        values = ["2024-01-05"] * 29 + [bad]

        assert type_of(values) == "categorical"

    def test_a_value_that_is_not_shaped_like_a_date_is_found_after_the_first_thousand(self):
        # numpy reads a bare year as a date, so only the pattern gate can reject this value.
        values = ["2024-01-05"] * 1500 + ["1234"]

        assert type_of(values) == "categorical"

    @pytest.mark.parametrize("bad", ["2024-02-30", "2024-01-05T24:00:00"])
    def test_a_bad_value_is_found_behind_many_distinct_good_ones_that_repeat(self, bad):
        good = [f"2024-01-{1 + k % 28:02d}T{k % 24:02d}:30:00" for k in range(60)]
        values = good * 2 + [bad]

        assert type_of(good * 2) == "datetime"
        assert type_of(values) != "datetime"


def test_semantic_type_values_are_the_documented_words():
    assert [t.value for t in SemanticType] == [
        "numeric",
        "categorical",
        "datetime",
        "text",
        "identifier",
        "boolean",
        "unknown",
    ]
