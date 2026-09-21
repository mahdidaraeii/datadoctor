"""The findings that ``check_pii`` reports, and the wording of each.

Only column names, counts and shares appear here, never a value, because a value could be the
personal data itself. Every finding's confidence is below 1 and says what would confirm it.
"""

from datadoctor.core.result import Finding, Severity

# How many columns a finding spells out. The rest are counted, and all are in the metrics.
LISTED = 5

# The severity is the impact if the finding is real, and the confidence is how sure the evidence
# makes us. Both depend only on what kind of personal data it is and how it was found.
GRADES = {
    ("national_id", "column"): (Severity.HIGH, 0.85),
    ("national_id", "partial"): (Severity.HIGH, 0.7),
    ("national_id", "name"): (Severity.MEDIUM, 0.4),
    ("email", "column"): (Severity.MEDIUM, 0.9),
    ("email", "partial"): (Severity.MEDIUM, 0.7),
    ("email", "name"): (Severity.MEDIUM, 0.4),
    ("phone", "column"): (Severity.MEDIUM, 0.8),
    ("phone", "partial"): (Severity.MEDIUM, 0.7),
    ("phone", "name"): (Severity.MEDIUM, 0.4),
    ("name", "corroborated"): (Severity.MEDIUM, 0.7),
    ("name", "name"): (Severity.LOW, 0.5),
    ("name", "bare"): (Severity.LOW, 0.5),
    ("free_text", "embedded"): (Severity.MEDIUM, 0.6),
    ("free_text", "hint"): (Severity.LOW, 0.4),
}

TITLES = {
    ("national_id", "column"): "National identifiers found in columns",
    ("national_id", "partial"): "National identifiers found in some values",
    ("national_id", "name"): "Columns named like national identifiers",
    ("email", "column"): "Email addresses found in columns",
    ("email", "partial"): "Email addresses found in some values",
    ("email", "name"): "Columns named like email addresses",
    ("phone", "column"): "Phone numbers found in columns",
    ("phone", "partial"): "Phone numbers found in some values",
    ("phone", "name"): "Columns named like phone numbers",
    ("name", "corroborated"): "Columns that appear to hold personal names",
    ("name", "name"): "Columns named like personal names",
    ("name", "bare"): "Columns named 'name' whose values look like personal names",
    ("free_text", "embedded"): "Free-text columns with contact details in them",
    ("free_text", "hint"): "Free-text columns that may hold personal data",
}

_ORDER = ["national_id", "email", "phone", "name", "free_text"]
_ORDER_OF_EVIDENCE = ["column", "corroborated", "embedded", "partial", "bare", "name", "hint"]

_FORMATS = {
    "us_ssn": "US social security number",
    "uk_nino": "UK national insurance number",
    "iban": "IBAN",
}

_INTERPRETATION = {
    "national_id": (
        "National identifiers identify one person and are heavily regulated. Keeping them in a "
        "modeling dataset is rarely necessary, and sharing the dataset would share them. This is "
        "a heuristic based on the format of the values or the name of the column."
    ),
    "email": (
        "Email addresses identify people, often directly. A dataset that holds them can usually "
        "be tied back to individuals, so it may fall under privacy rules. This is a heuristic "
        "based on the format of the values or the name of the column."
    ),
    "phone": (
        "Phone numbers identify people, often directly, and may fall under privacy rules. This "
        "is a heuristic based on the format of the values or the name of the column."
    ),
    "name": (
        "Personal names identify people, alone or together with other columns, and may fall "
        "under privacy rules. This is a heuristic based mostly on the name of the column."
    ),
    "free_text": (
        "People write personal details into free text, such as names, contact details and "
        "health or account information, and nothing in the column marks where. This is a "
        "heuristic: it is a warning about what the column could hold, not proof that it holds it."
    ),
}

# What would confirm the finding, and what to do then. Nothing is changed by this analysis.
_CONFIRM = {
    "national_id": (
        "To confirm, have someone who is allowed to see the values check a few against the "
        "issuing scheme's rules. For IBANs a correct checksum does not show that the account "
        "exists."
    ),
    "email": (
        "To confirm, have someone who is allowed to see the values check that they are real "
        "addresses of individuals and not role or system mailboxes such as support@ or noreply@."
    ),
    "phone": (
        "To confirm, have someone who is allowed to see the values check that they are real "
        "numbers of individuals and not extensions, order numbers or other reference numbers."
    ),
    "name": (
        "To confirm, have someone who is allowed to see the values check that they are the names "
        "of people and not of organisations, products or places."
    ),
    "free_text": (
        "To confirm, have someone who is allowed to read personal data look at a small sample "
        "of the values."
    ),
}
_THEN = (
    " If they are personal data, decide whether the column is needed, and remove, mask or hash "
    "it before the data is shared or used for modeling. Nothing was changed here."
)

_LIMITATIONS = {
    "national_id": (
        "Only US social security numbers, UK national insurance numbers and IBANs are "
        "recognized, so identifiers of other countries are not found. A value that matches can "
        "still be something else, and a real identifier written in another format is missed."
    ),
    "email": (
        "The pattern accepts any text shaped like an address and does not check that the domain "
        "or mailbox exists. Addresses written in unusual ways are missed."
    ),
    "phone": (
        "A phone number needs 9 to 15 digits with a plus or at least two separators, since "
        "plain digit runs are too easily ids, postal codes or amounts. Numbers written as plain "
        "digits, with one separator or with local formats are missed unless the column is "
        "named like a phone."
    ),
    "name": (
        "Values alone cannot tell a name from a product or a place, so a column is examined "
        "only if its name says it holds personal names. Names in columns with other names are "
        "missed, and names of organisations in such a column are false alarms."
    ),
    "free_text": (
        "Only email addresses and phone numbers are looked for inside the text. Names, "
        "addresses and other details written in the text are not found, and a column can hold "
        "personal data without any of the recognized details."
    ),
}
_BY_NAME = (
    " For columns reported only by their name, no value matched the pattern, or the values "
    "were not compared with it because they are numbers, so the name is the only evidence."
)
_PARTIAL = (
    " For columns reported for some values, most values did not match, so the column may mix "
    "personal data with other content."
)


def _more(total: int, shown: int) -> str:
    return f" and {total - shown} more" if total > shown else ""


def _describe(detection: dict) -> str:
    name, evidence = detection["name"], detection["evidence"]
    if evidence == "name":
        return f"{name} (the name suggests it, and no value matched)"
    matched, checked = detection["matched"], detection["checked"]
    counts = f"{matched:,} of {checked:,} values, {detection['share']:.0%}"
    if detection["kind"] == "national_id":
        return f"{name} ({_FORMATS[detection['format']]}: {counts})"
    if detection["kind"] == "name":
        return f"{name} ({counts} look like names)"
    if detection["kind"] == "free_text":
        if evidence == "hint":
            return f"{name} (named like free text)"
        return f"{name} ({counts} contain an email address or a phone number)"
    return f"{name} ({counts})"


def build(detections: list[dict], *, rows_examined: int, n_rows: int) -> list[Finding]:
    """One finding for each kind of personal data and way of finding it, most serious first."""
    findings = []
    keys = sorted(
        {(d["kind"], d["evidence"]) for d in detections},
        key=lambda key: (_ORDER.index(key[0]), _ORDER_OF_EVIDENCE.index(key[1])),
    )
    sample_note = (
        f" Counts are from a sample of {rows_examined:,} of {n_rows:,} rows."
        if rows_examined < n_rows
        else ""
    )
    for kind, evidence in keys:
        columns = [d for d in detections if (d["kind"], d["evidence"]) == (kind, evidence)]
        ranked = sorted(columns, key=lambda d: (-(d["matched"] or 0), d["name"]))
        shown = [_describe(d) for d in ranked[:LISTED]]
        severity, confidence = GRADES[(kind, evidence)]
        limitations = _LIMITATIONS[kind]
        if evidence == "name":
            limitations += _BY_NAME
        if evidence == "partial":
            limitations += _PARTIAL
        findings.append(
            Finding(
                category="privacy",
                severity=severity,
                confidence=confidence,
                title=TITLES[(kind, evidence)],
                evidence=(
                    f"{len(columns)} {'column' if len(columns) == 1 else 'columns'}: "
                    + "; ".join(shown)
                    + _more(len(columns), len(shown))
                    + "."
                    + sample_note
                    + " The values are not shown."
                ),
                interpretation=_INTERPRETATION[kind],
                limitations=limitations,
                affected_columns=tuple(d["name"] for d in ranked),
                recommendation=_CONFIRM[kind] + _THEN,
            )
        )
    return findings
