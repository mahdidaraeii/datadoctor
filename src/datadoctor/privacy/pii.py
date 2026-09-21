"""Personal data diagnostics: emails, phone numbers, national identifiers, names and free text.

Detection is by pattern, plus the column name as weaker evidence, and every finding is a
heuristic that says so and says what would confirm it. Values are never returned: the results hold
column names, counts and shares only.
"""

from collections.abc import Callable
from typing import NamedTuple

from pandas.api import types as pt

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.core.guardrails import apply_guardrails
from datadoctor.core.result import AnalysisResult
from datadoctor.privacy import patterns
from datadoctor.privacy import pii_findings as report

# A column "holds" a kind of personal data when at least this share of its non-null values match.
FULL_SHARE = 0.8
# Below that, a column is reported as having some values that match if at least this share do.
PARTIAL_SHARE = 0.01
# Fewer matches than this are never reported, however high their share.
MIN_MATCHES = 3
# A text column is free text when at least half of its first FREE_TEXT_LOOKED_AT values have this
# many words. A personal name has at most four, so a column of names is not free text.
FREE_TEXT_WORDS = 5
FREE_TEXT_SHARE = 0.5
FREE_TEXT_LOOKED_AT = 1000

_NUMERIC_KINDS = {"integer", "floating", "mixed-integer-float"}


class Detection(NamedTuple):
    """One column that shows one kind of personal data, and the evidence for it.

    ``kind`` is email, phone, national_id, name or free_text. ``evidence`` is how it was found:
    ``column`` (most values match), ``partial`` (some do), ``name`` (only the column name
    suggests it), ``corroborated`` and ``bare`` (a name column whose values look like names),
    ``embedded`` (contact details inside free text) or ``hint`` (a free-text column by its name).
    ``matched`` and ``checked`` are ``None`` when no value was matched against a pattern.
    """

    name: str
    kind: str
    evidence: str
    format: str | None
    matched: int | None
    checked: int | None
    share: float | None


def check_pii(dataset: Dataset, config: AnalysisConfig) -> AnalysisResult:
    """Look for columns that hold personal data, without ever showing a value.

    **Emails, phone numbers, national identifiers.** A text column is reported when at least
    ``FULL_SHARE`` of its non-null values match, or, for emails and national identifiers, when at
    least ``PARTIAL_SHARE`` of them do (and at least ``MIN_MATCHES``). A phone number needs 9 to
    15 digits in groups with a plus or two separators, so plain digit runs such as postal codes and
    ids never count. National identifiers are US social security numbers, UK national insurance
    numbers and IBANs, all validated beyond their shape (IBANs by the mod-97 checksum). A column
    named like one of these but with no matching value is reported at low confidence, and so is a
    numeric column named like a phone number or an identifier.

    **Names.** A column is reported only if its name says it holds personal names, such as
    ``first_name`` or ``surname``, since values alone cannot tell a person from a product or a city.
    A plain ``name`` column needs values that look like names as well.

    **Free text.** A text column where most values have several words is reported if some values
    contain an email address or phone number, or if its name says it holds comments, notes or the
    like.

    The severity is the impact if the finding is real. The confidence is below 1 for every finding
    and depends only on the kind of evidence. The size guardrails of ``config`` apply: a large
    dataset is examined on a seeded sample of its rows, which is said in the findings.

    Matching uses Python's ``re`` on each value, so the result does not depend on the pandas
    version or on which regex engine its string dtype uses.

    ``metrics`` holds ``n_rows``, ``rows_examined``, ``scanned_columns`` (the text columns whose
    values were matched) and ``columns`` (one entry per detection, with counts and shares only).
    Values are never printed, stored or logged.

    Args:
        dataset: The dataset to examine. It must have at least one row.
        config: The settings to run under. They are recorded in the result.

    Returns:
        The metrics and findings above, with ``config`` recorded.
    """
    frame = dataset.data
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        raise DatasetError("cannot look for personal data in a dataset with no rows")
    guard = apply_guardrails(dataset, config)
    sample = guard.frame

    detections: list[Detection] = []
    scanned: list[str] = []
    for position in range(n_columns):
        name = str(frame.columns[position])
        words = patterns.tokens(name)
        values = sample.iloc[:, position].dropna()
        if values.empty:
            continue
        kind = pt.infer_dtype(values, skipna=True)
        if kind == "string":
            scanned.append(name)
            detections.extend(_in_text_column(name, words, values.tolist()))
        elif kind in _NUMERIC_KINDS:
            detections.extend(_by_name_only(name, words))

    findings = report.build(
        [detection._asdict() for detection in detections],
        rows_examined=len(sample),
        n_rows=n_rows,
    )
    if sample is not frame:
        findings.append(guard.findings[0])  # the row sample comes first, per apply_guardrails
    metrics = {
        "n_rows": n_rows,
        "rows_examined": len(sample),
        "scanned_columns": scanned,
        "columns": [detection._asdict() for detection in detections],
    }
    return AnalysisResult(findings=findings, metrics=metrics, config=config)


def _in_text_column(name: str, words: frozenset[str], texts: list[str]) -> list[Detection]:
    if _is_free_text(texts):
        found = _free_text(name, words, texts)
        return [] if found is None else [found]
    detections = _contact_details_and_ids(name, words, texts)
    person = _person_name(name, words, texts)
    if person is not None:
        detections.append(person)
    return detections


def _count(texts: list[str], matches: Callable[[str], bool]) -> int:
    return sum(1 for text in texts if matches(text))


def _level(matched: int, checked: int, *, partial_allowed: bool) -> str | None:
    if matched < MIN_MATCHES:
        return None
    share = matched / checked
    if share >= FULL_SHARE:
        return "column"
    if partial_allowed and share >= PARTIAL_SHARE:
        return "partial"
    return None


def _contact_details_and_ids(name: str, words: frozenset[str], texts: list[str]) -> list[Detection]:
    checked = len(texts)
    found = _detection(
        name,
        "email",
        _count(texts, patterns.is_email),
        checked,
        partial=True,
        named=patterns.names_email(words),
    )
    # A phone number is hard to tell from other digit groups, so a few matches count only when the
    # column is also named like a phone.
    named_phone = patterns.names_phone(words)
    found += _detection(
        name,
        "phone",
        _count(texts, patterns.is_phone),
        checked,
        partial=named_phone,
        named=named_phone,
    )

    for format_name, is_valid in patterns.NATIONAL_IDS.items():
        matched = _count(texts, is_valid)
        level = _level(matched, checked, partial_allowed=True)
        if level is not None:
            share = matched / checked
            found.append(
                Detection(name, "national_id", level, format_name, matched, checked, share)
            )
    if not any(d.kind == "national_id" for d in found) and patterns.names_national_id(words):
        found.append(Detection(name, "national_id", "name", None, None, None, None))
    return found


def _detection(
    name: str, kind: str, matched: int, checked: int, *, partial: bool, named: bool
) -> list[Detection]:
    level = _level(matched, checked, partial_allowed=partial)
    if level is not None:
        return [Detection(name, kind, level, None, matched, checked, matched / checked)]
    if named:
        return [Detection(name, kind, "name", None, None, None, None)]
    return []


def _by_name_only(name: str, words: frozenset[str]) -> list[Detection]:
    """Numeric columns: a phone number or an identifier stored as a number has lost its format."""
    found = []
    if patterns.names_phone(words):
        found.append(Detection(name, "phone", "name", None, None, None, None))
    if patterns.names_national_id(words):
        found.append(Detection(name, "national_id", "name", None, None, None, None))
    return found


def _person_name(name: str, words: frozenset[str], texts: list[str]) -> Detection | None:
    kind = patterns.names_a_person(words)
    if kind is None:
        return None
    checked = len(texts)
    matched = _count(texts, patterns.is_name)
    looks_like_names = matched >= MIN_MATCHES and matched / checked >= FULL_SHARE
    if kind == "person":
        evidence = "corroborated" if looks_like_names else "name"
    elif looks_like_names:
        evidence = "bare"
    else:
        return None
    return Detection(name, "name", evidence, None, matched, checked, matched / checked)


def _is_free_text(texts: list[str]) -> bool:
    looked_at = texts[:FREE_TEXT_LOOKED_AT]
    long_enough = sum(1 for text in looked_at if patterns.word_count(text) >= FREE_TEXT_WORDS)
    return long_enough / len(looked_at) >= FREE_TEXT_SHARE


def _free_text(name: str, words: frozenset[str], texts: list[str]) -> Detection | None:
    checked = len(texts)
    matched = _count(texts, patterns.has_contact_detail)
    if matched >= MIN_MATCHES and matched / checked >= PARTIAL_SHARE:
        return Detection(name, "free_text", "embedded", None, matched, checked, matched / checked)
    if patterns.names_free_text(words):
        return Detection(name, "free_text", "hint", None, matched, checked, matched / checked)
    return None
