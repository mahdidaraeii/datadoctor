from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity, load_dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.privacy import check_pii

N = 200
CONFIG = AnalysisConfig()
REAL_FILE = Path(__file__).parents[2] / "data" / "telecom_customers.csv"

FIRST_NAMES = ["Anna", "Peter", "Maria", "John", "Sofia", "Lars", "Chloe", "Omar"]
SURNAMES = ["Müller", "O'Brien", "García", "Smith", "Van Dijk", "Nguyen", "Kowalski"]
# Checksum-valid IBANs from the public documentation examples, some written with spaces.
IBANS = [
    "DE89370400440532013000",
    "GB82 WEST 1234 5698 7654 32",
    "FR1420041010050500013M02606",
    "NL91 ABNA 0417 1643 00",
    "ES9121000418450200051332",
]
CALL_ME = "Please call me back on +44 7700 900123 tomorrow about the invoice"
FINE = "Everything was fine with the service and the staff today"


def planted_frame() -> pd.DataFrame:
    """200 rows of fake personal data, one column per case, and decoys that are not personal."""
    rows = range(N)
    return pd.DataFrame(
        {
            # personal data, found by the values
            "email": [f"user{k}@example.com" for k in rows],
            "contact_number": [f"+44 7700 900{k:03d}" for k in rows],
            "mobile": [f"+1 555 010 {k:04d}" for k in rows],
            "tax_ref": [f"{100 + k % 500:03d}-{10 + k % 80:02d}-{1000 + k:04d}" for k in rows],
            "insurance_no": [f"AB{100000 + k}C" for k in rows],
            "bank_account": [IBANS[k % len(IBANS)] for k in rows],
            "partial_email": [f"user{k}@example.com" if k % 10 == 0 else "unknown" for k in rows],
            "phone_alt": [f"+1 555 010 {k:04d}" if k % 3 == 0 else "none" for k in rows],
            # names, found by the column name and confirmed by the values
            "first_name": [FIRST_NAMES[k % len(FIRST_NAMES)] for k in rows],
            "surname": [SURNAMES[k % len(SURNAMES)] for k in rows],
            "name": [f"{FIRST_NAMES[k % 8]} {SURNAMES[k % 7]}" for k in rows],
            # free text
            "comments": [CALL_ME if k % 10 == 0 else FINE for k in rows],
            "feedback": [FINE] * N,
            # numbers stored as numbers, found by the column name alone
            "phone_number": [5550100000 + k for k in rows],
            "ssn": [100010000 + k for k in rows],
            # decoys: not personal data, and some are shaped like it
            "postal_code": [10000 + k * 37 % 89999 for k in rows],
            "zip": [f"{k * 37 % 100000:05d}" for k in rows],
            "customer_id": [f"CUS{100000 + k}" for k in rows],
            "signup_date": [f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}" for k in rows],
            "amount": [f"{k * 3}.{k % 100:02d}" for k in rows],
            "big_amount": [f"{1 + k}.{234 + k}.567.890" for k in rows],
            "ip_address": [f"192.168.{k % 250}.{1 + k % 200}" for k in rows],
            "account": [f"{100000000 + k * 7}" for k in rows],
            "version": [f"{k % 5}.{k % 9}.{k % 3}" for k in rows],
            "product_name": [["Premium Plan", "Basic Plan", "Family Pack"][k % 3] for k in rows],
            "plan_type": [["Business", "Premium", "Basic"][k % 3] for k in rows],
            "product_description": [
                f"A durable device with a {k % 7} year warranty included" for k in rows
            ],
            "phone_model": [["iPhone 12", "Pixel 8", "Galaxy S23"][k % 3] for k in rows],
            "phone_calls": [k % 40 for k in rows],
        }
    )


EXPECTED = {
    ("email", "email", "column", None),
    ("contact_number", "phone", "column", None),
    ("mobile", "phone", "column", None),
    ("tax_ref", "national_id", "column", "us_ssn"),
    ("insurance_no", "national_id", "column", "uk_nino"),
    ("bank_account", "national_id", "column", "iban"),
    ("partial_email", "email", "partial", None),
    ("phone_alt", "phone", "partial", None),
    ("first_name", "name", "corroborated", None),
    ("surname", "name", "corroborated", None),
    ("name", "name", "bare", None),
    ("comments", "free_text", "embedded", None),
    ("feedback", "free_text", "hint", None),
    ("phone_number", "phone", "name", None),
    ("ssn", "national_id", "name", None),
}

# Severity is the impact if it is real, and confidence depends only on the kind of evidence.
EXPECTED_GRADES = {
    "National identifiers found in columns": (Severity.HIGH, 0.85),
    "Columns named like national identifiers": (Severity.MEDIUM, 0.4),
    "Email addresses found in columns": (Severity.MEDIUM, 0.9),
    "Email addresses found in some values": (Severity.MEDIUM, 0.7),
    "Phone numbers found in columns": (Severity.MEDIUM, 0.8),
    "Phone numbers found in some values": (Severity.MEDIUM, 0.7),
    "Columns named like phone numbers": (Severity.MEDIUM, 0.4),
    "Columns that appear to hold personal names": (Severity.MEDIUM, 0.7),
    "Columns named 'name' whose values look like personal names": (Severity.LOW, 0.5),
    "Free-text columns with contact details in them": (Severity.MEDIUM, 0.6),
    "Free-text columns that may hold personal data": (Severity.LOW, 0.4),
}

# Values that must never appear in a result, taken from the planted rows.
SECRETS = [
    "user5@example.com",
    "+1 555 010 0005",
    "+44 7700 900005",
    "105-15-1005",
    "AB100005C",
    "DE89370400440532013000",
    "GB82 WEST 1234 5698 7654 32",
    "O'Brien",
    "Müller",
    "900123",
    "5550100005",
]


@pytest.fixture(scope="module")
def result() -> AnalysisResult:
    return check_pii(Dataset(data=planted_frame(), name="planted"), CONFIG)


def check(frame: pd.DataFrame, config: AnalysisConfig = CONFIG) -> AnalysisResult:
    return check_pii(Dataset(data=frame, name="t"), config)


def found(result: AnalysisResult) -> set:
    return {(d["name"], d["kind"], d["evidence"], d["format"]) for d in result.metrics["columns"]}


def in_column(values, name="col") -> set:
    return found(check(pd.DataFrame({name: values})))


class TestPlantedPersonalData:
    def test_every_planted_column_is_found_by_the_right_evidence_and_no_decoy_is(self, result):
        assert found(result) == EXPECTED

    def test_severity_and_confidence_follow_the_kind_of_evidence(self, result):
        graded = {f.title: (f.severity, f.confidence) for f in result.findings}

        assert graded == EXPECTED_GRADES

    def test_every_finding_is_below_full_confidence_and_says_what_would_confirm_it(self, result):
        for f in result.findings:
            assert f.category == "privacy"
            assert f.confidence < 1.0
            assert "To confirm" in f.recommendation
            assert "Nothing was changed here" in f.recommendation
            assert f.limitations

    def test_columns_are_listed_by_how_many_values_matched(self, result):
        phones = next(f for f in result.findings if f.title == "Phone numbers found in columns")

        assert set(phones.affected_columns) == {"contact_number", "mobile"}
        assert "200 of 200 values, 100%" in phones.evidence

    def test_the_metrics_hold_counts_and_shares(self, result):
        entry = next(d for d in result.metrics["columns"] if d["name"] == "partial_email")

        assert (entry["matched"], entry["checked"], entry["share"]) == (20, 200, 0.1)
        assert "postal_code" not in result.metrics["scanned_columns"]  # numbers are not scanned
        assert "zip" in result.metrics["scanned_columns"]


class TestValuesAreNeverShown:
    def test_no_planted_value_appears_anywhere_in_the_result(self, result):
        text = result.to_json()

        cells = planted_frame().astype(str).to_numpy().ravel()
        for secret in SECRETS:
            assert any(secret in cell for cell in cells), f"{secret!r} is not planted"
            assert secret not in text

    def test_the_findings_never_quote_a_value_from_the_free_text(self, result):
        text = " ".join(f.evidence + f.interpretation + f.limitations for f in result.findings)

        assert "call me back" not in text
        assert "invoice" not in text


class TestWhatCountsAsEvidence:
    @pytest.mark.parametrize(
        ("value", "is_phone"),
        [
            ("+1 555 010 0001", True),  # a plus, in groups
            ("(555) 123-4567", True),  # separators, no plus
            ("555-123-4567", True),  # exactly two separators, no plus
            ("+15551234567", True),  # a plus and no separators
            ("5551234567", False),  # plain digits: a postal code, an id, an amount
            ("555-123456789", False),  # one separator and no plus
            ("+1 234 5678", False),  # 8 digits, too few
            ("+1 2345 6789 0123 4567", False),  # 17 digits, too many
            ("2024-05-14", False),  # a date
            ("192.168.100.100", False),  # an IP address
            ("1.234.567.890", False),  # a formatted amount, shaped like an IP address
        ],
    )
    def test_phone_numbers_need_a_plus_or_two_separators_and_the_right_number_of_digits(
        self, value, is_phone
    ):
        assert bool(in_column([value] * 30)) == is_phone

    @pytest.mark.parametrize(
        "value",
        [
            "000-12-3456",  # area 000 was never issued
            "666-12-3456",  # nor was 666
            "912-12-3456",  # nor the 900s
            "123-00-3456",  # group 00
            "123-45-0000",  # serial 0000
        ],
    )
    def test_a_social_security_number_shape_with_a_number_never_issued_is_not_one(self, value):
        assert in_column([value] * 30) == set()

    def test_a_valid_social_security_number_is_found(self):
        assert in_column(["123-45-6789"] * 30) == {("col", "national_id", "column", "us_ssn")}

    @pytest.mark.parametrize("value", ["BG123456A", "ZZ123456A", "DA123456A", "AB123456E"])
    def test_a_national_insurance_number_needs_an_allocated_prefix_and_a_suffix_a_to_d(self, value):
        assert in_column([value] * 30) == set()

    def test_an_iban_needs_a_correct_checksum(self):
        good = "DE89370400440532013000"
        broken = "DE88370400440532013000"

        assert in_column([good] * 30) == {("col", "national_id", "column", "iban")}
        assert in_column([broken] * 30) == set()

    def test_the_thresholds_are_inclusive_and_a_few_matches_are_ignored(self):
        def emails(matching, total):
            return [f"u{k}@example.com" if k < matching else "unknown" for k in range(total)]

        assert in_column(emails(80, 100)) == {("col", "email", "column", None)}  # 80%
        assert in_column(emails(79, 100)) == {("col", "email", "partial", None)}  # just below
        assert in_column(emails(10, 1000)) == {("col", "email", "partial", None)}  # 1%
        assert in_column(emails(9, 1000)) == set()  # 0.9%
        assert in_column(emails(2, 20)) == set()  # 10%, but only two matches
        assert in_column(emails(3, 100)) == {("col", "email", "partial", None)}  # three is enough

    def test_a_few_phone_numbers_count_only_in_a_column_named_like_a_phone(self):
        values = [f"+1 555 010 {k:04d}" if k < 30 else "none" for k in range(100)]

        assert in_column(values) == set()
        assert in_column(values, "phone_alt") == {("phone_alt", "phone", "partial", None)}


class TestFreeText:
    SENTENCE = "Everything was fine today thanks"  # five words
    SHORT = "all good"

    def test_the_word_and_share_cutoffs_are_inclusive(self):
        def hint(long, total, sentence):
            values = [sentence] * long + [self.SHORT] * (total - long)
            return in_column(values, "notes")

        assert hint(5, 10, self.SENTENCE) == {("notes", "free_text", "hint", None)}  # half
        assert hint(4, 10, self.SENTENCE) == set()  # under half
        assert hint(10, 10, "Everything was fine today") == set()  # four words

    def test_contact_details_need_three_matches_and_one_percent(self):
        def with_details(matching, total):
            values = [
                "Please write to me at anna@example.org about it" if k < matching else self.SENTENCE
                for k in range(total)
            ]
            return in_column(values, "text")

        assert with_details(3, 300) == {("text", "free_text", "embedded", None)}  # 1%
        assert with_details(3, 301) == set()  # just under 1%
        assert with_details(2, 300) == set()  # only two

    def test_an_email_alone_in_the_text_counts(self):
        values = ["Please write to me at anna@example.org about it"] * 3 + [self.SENTENCE] * 20

        assert in_column(values, "text") == {("text", "free_text", "embedded", None)}

    def test_only_the_first_thousand_values_decide_whether_a_column_is_free_text(self):
        values = [self.SENTENCE] * 400 + [self.SHORT] * 600 + [self.SENTENCE] * 1000

        # 40% of the first 1,000 are long, although 70% of all are.
        assert in_column(values, "notes") == set()

    @pytest.mark.parametrize(
        "name",
        ["comment", "notes", "message", "Review", "bio", "complaints", "narrative", "transcript"]
        + ["remarks", "feedback", "messages", "reviews", "comments"],
    )
    def test_the_names_that_say_free_text(self, name):
        assert in_column([self.SENTENCE] * 20, name) == {(name, "free_text", "hint", None)}

    def test_a_column_not_named_for_free_text_is_left_alone_without_contact_details(self):
        assert in_column([self.SENTENCE] * 20, "product_description") == set()


class TestNameEvidenceVocabulary:
    @pytest.mark.parametrize("name", ["email", "e_mail", "Emails", "mail_address"])
    def test_email_names(self, name):
        assert in_column(["not an address"] * 20, name) == {(name, "email", "name", None)}

    @pytest.mark.parametrize("name", ["telephone", "tel", "fax", "msisdn", "Phones", "mobile"])
    def test_phone_names_on_numbers_stored_as_numbers(self, name):
        frame = pd.DataFrame({name: np.arange(20) + 5550100000})

        assert found(check(frame)) == {(name, "phone", "name", None)}

    @pytest.mark.parametrize(
        "name",
        ["ssn", "nino", "iban", "passport_no", "social_security_number", "national_id"]
        + ["national_insurance_no", "tax_id"],
    )
    def test_national_identifier_names_on_numbers_stored_as_numbers(self, name):
        frame = pd.DataFrame({name: np.arange(20) + 100010000})

        assert found(check(frame)) == {(name, "national_id", "name", None)}

    @pytest.mark.parametrize(
        "name",
        ["last_name", "middle_name", "given_name", "family_name", "full_name", "maiden_name"]
        + ["forename", "lastname", "fname", "lname", "surname", "firstName"],
    )
    def test_person_name_columns(self, name):
        assert in_column(FIRST_NAMES * 5, name) == {(name, "name", "corroborated", None)}

    def test_a_plain_name_column_needs_most_of_its_values_to_look_like_names(self):
        def share(names, others):
            return in_column(["Omar Ali"] * names + ["basic pack"] * others, "name")

        assert share(8, 2) == {("name", "name", "bare", None)}  # 80%, inclusive
        assert share(7, 3) == set()  # 70%


class TestMoreEvidenceRules:
    def test_a_column_of_ibans_written_with_spaces_is_not_free_text(self):
        # Six space-separated groups, but no words.
        values = ["GB82 WEST 1234 5698 7654 32"] * 30

        assert in_column(values) == {("col", "national_id", "column", "iban")}

    @pytest.mark.parametrize("name", ["ssn_status", "passport_valid", "tax_id_type", "nino_count"])
    def test_a_national_identifier_word_next_to_a_word_that_changes_the_meaning_is_not_evidence(
        self, name
    ):
        frame = pd.DataFrame({name: np.arange(20) + 100010000})

        assert found(check(frame)) == set()

    def test_the_weaker_kinds_of_evidence_have_their_own_grades(self):
        ssns = ["123-45-6789" if k < 10 else "unknown" for k in range(100)]
        cases = {
            "National identifiers found in some values": pd.DataFrame({"col": ssns}),
            "Columns named like email addresses": pd.DataFrame({"email": ["not an address"] * 20}),
            "Columns named like personal names": pd.DataFrame({"first_name": ["a1", "b2"] * 10}),
        }

        graded = {
            f.title: (f.severity, f.confidence)
            for frame in cases.values()
            for f in check(frame).findings
        }

        assert graded == {
            "National identifiers found in some values": (Severity.HIGH, 0.7),
            "Columns named like email addresses": (Severity.MEDIUM, 0.4),
            "Columns named like personal names": (Severity.LOW, 0.5),
        }

    def test_columns_are_ranked_by_how_many_values_matched(self):
        frame = pd.DataFrame(
            {
                "a_col": [f"u{k}@example.com" if k < 85 else "unknown" for k in range(100)],
                "b_col": [f"u{k}@example.com" for k in range(100)],
            }
        )

        (finding,) = check(frame).findings

        assert finding.affected_columns == ("b_col", "a_col")


class TestWhatTheFindingsSay:
    def test_findings_run_from_national_identifiers_to_free_text_and_strongest_evidence_first(
        self, result
    ):
        assert [f.title for f in result.findings] == [
            "National identifiers found in columns",
            "Columns named like national identifiers",
            "Email addresses found in columns",
            "Email addresses found in some values",
            "Phone numbers found in columns",
            "Phone numbers found in some values",
            "Columns named like phone numbers",
            "Columns that appear to hold personal names",
            "Columns named 'name' whose values look like personal names",
            "Free-text columns with contact details in them",
            "Free-text columns that may hold personal data",
        ]

    def test_the_columns_that_matched_most_are_listed_first_and_ties_by_name(self, result):
        phones = next(f for f in result.findings if f.title == "Phone numbers found in columns")

        assert phones.affected_columns == ("contact_number", "mobile")

    def test_each_kind_of_evidence_is_described_with_counts_only(self, result):
        evidence = {f.title: f.evidence for f in result.findings}

        assert (
            "tax_ref (US social security number: 200 of 200 values, 100%)"
            in (evidence["National identifiers found in columns"])
        )
        assert "bank_account (IBAN: 200 of 200" in evidence["National identifiers found in columns"]
        assert (
            "ssn (the name suggests it, and no value matched)"
            in (evidence["Columns named like national identifiers"])
        )
        assert (
            "surname (200 of 200 values, 100% look like names)"
            in (evidence["Columns that appear to hold personal names"])
        )
        assert (
            "comments (20 of 200 values, 10% contain an email address or a phone number)"
            in (evidence["Free-text columns with contact details in them"])
        )
        assert (
            "feedback (named like free text)"
            in (evidence["Free-text columns that may hold personal data"])
        )

    def test_the_limits_of_weak_evidence_are_said_only_where_they_apply(self, result):
        limits = {f.title: f.limitations for f in result.findings}

        assert "the name is the only evidence" in limits["Columns named like phone numbers"]
        assert "the name is the only evidence" not in limits["Phone numbers found in columns"]
        assert "may mix personal data" in limits["Phone numbers found in some values"]
        assert "may mix personal data" not in limits["Phone numbers found in columns"]

    def test_only_five_columns_are_spelled_out_and_all_are_listed_as_affected(self):
        frame = pd.DataFrame(
            {f"mail_{k}": [f"u{j}@example.com" for j in range(30)] for k in range(7)}
        )

        (finding,) = check(frame).findings

        assert len(finding.affected_columns) == 7
        assert "7 columns:" in finding.evidence
        assert "and 2 more" in finding.evidence


class TestColumnNames:
    @pytest.mark.parametrize(
        "name",
        ["phone_model", "phone_calls", "email_opt_in", "company_name", "user_name", "product_name"],
    )
    def test_a_name_that_says_it_is_something_else_is_not_evidence(self, name):
        frame = pd.DataFrame({name: ["Anna Schmidt", "Peter Jones"] * 20})

        assert found(check(frame)) == set()

    def test_a_numeric_column_named_for_usage_is_not_a_phone_number(self):
        frame = pd.DataFrame({"mobile_data_usage": np.arange(50) + 1000})

        assert found(check(frame)) == set()

    def test_a_person_name_column_is_reported_by_its_name_even_when_the_values_are_not_names(self):
        frame = pd.DataFrame({"first_name": ["a1", "b2", "c3"] * 10})

        assert found(check(frame)) == {("first_name", "name", "name", None)}

    def test_a_plain_name_column_needs_values_that_look_like_names(self):
        products = pd.DataFrame({"name": ["Premium plan", "basic pack", "family pack"] * 10})

        assert found(check(products)) == set()

    def test_a_short_value_column_named_comment_is_not_free_text(self):
        frame = pd.DataFrame({"comment": ["ok", "fine", "late"] * 20})

        assert found(check(frame)) == set()

    def test_a_column_of_full_names_is_not_mistaken_for_free_text(self):
        frame = pd.DataFrame({"notes": ["Anna Maria van Dijk", "Peter van der Berg"] * 20})

        assert found(check(frame)) == set()

    def test_camel_case_names_are_split_into_words(self):
        frame = pd.DataFrame({"firstName": FIRST_NAMES * 5, "PhoneNumber": np.arange(40)})

        assert found(check(frame)) == {
            ("firstName", "name", "corroborated", None),
            ("PhoneNumber", "phone", "name", None),
        }


class TestGuardrails:
    def test_a_large_dataset_is_examined_on_a_sample_and_the_findings_say_so(self):
        result = check(planted_frame(), AnalysisConfig(row_threshold=50))

        assert result.metrics["rows_examined"] == 50
        assert found(result) == EXPECTED  # a match in a sample shows the column has them
        sample = [f for f in result.findings if f.category == "guardrail"]
        assert [(f.severity, f.title) for f in sample] == [
            (Severity.INFO, "Analysis ran on a sample of the rows")
        ]
        assert "sample of 50 of 200 rows" in result.findings[0].evidence

    def test_a_dataset_within_the_threshold_has_no_sample_note(self, result):
        assert not [f for f in result.findings if f.category == "guardrail"]
        assert "sample of" not in result.findings[0].evidence


class TestResult:
    def test_config_is_recorded_and_the_result_is_deterministic(self):
        config = AnalysisConfig(random_seed=3, row_threshold=100)

        first = check(planted_frame(), config)

        assert first.config == config
        assert check(planted_frame(), config).to_json() == first.to_json()

    def test_the_result_survives_json(self, result):
        assert AnalysisResult.from_json(result.to_json()) == result

    def test_a_dataset_without_rows_is_rejected(self):
        with pytest.raises(DatasetError, match="no rows"):
            check(pd.DataFrame({"a": []}))


class TestTheExampleDataset:
    """The committed example file holds no personal data. It must never be flagged."""

    def test_no_column_is_flagged_as_personal_data_as_loaded(self):
        assert REAL_FILE.exists(), f"the committed example dataset is missing: {REAL_FILE}"

        result = check_pii(load_dataset(REAL_FILE), CONFIG)

        assert result.findings == ()
        assert result.metrics["columns"] == []
        assert "customer_id" in result.metrics["scanned_columns"]

    def test_no_column_is_flagged_when_the_postal_code_is_kept_as_text(self):
        assert REAL_FILE.exists(), f"the committed example dataset is missing: {REAL_FILE}"

        dataset = load_dataset(REAL_FILE, text_columns=["postal_code"])
        result = check_pii(dataset, CONFIG)

        assert result.findings == ()
        assert {"customer_id", "postal_code"} <= set(result.metrics["scanned_columns"])
