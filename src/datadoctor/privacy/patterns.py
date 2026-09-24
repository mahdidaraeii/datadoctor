"""Patterns and validators for personal data, and the column-name evidence that goes with them.

Everything here looks at one value or one column name and answers yes or no. Nothing returns,
stores or logs a value, so a caller that keeps only the answers can never leak what was matched.

The patterns are deliberately narrow. A wrong match on a postal code or an order number costs more
trust than a missed personal number, and every match is a heuristic that the findings say so about.

Matching uses Python's ``re`` on one value at a time, never the ``str`` methods of a pandas column.
On pandas 3 those can run on a different regex engine, where ``\\w`` matches only ASCII letters, so
``Müller`` would match on one pandas version and not on another.
"""

import re

_DIGITS = frozenset("0123456789")

# --- Values -----------------------------------------------------------------------------------

# An address as a whole value, and inside longer text with ``search``.
EMAIL = re.compile(r"[\w.+%-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")

PHONE_SHAPE = re.compile(r"\+?\(?\d{1,4}\)?(?:[\s.-]?\(?\d{1,4}\)?){2,5}", re.ASCII)
# The same shape found inside text, not glued to other characters.
PHONE_EMBEDDED = re.compile(
    r"(?<![\w.])\+?\(?\d{1,4}\)?(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?!\w)", re.ASCII
)
_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}", re.ASCII)

SSN_SHAPE = re.compile(r"\d{3}-\d{2}-\d{4}", re.ASCII)
_NINO = re.compile(r"[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]", re.ASCII)
_NINO_UNUSED_PREFIXES = frozenset({"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"})
_IBAN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", re.ASCII)

_LETTER_RUN = re.compile(r"[^\W\d_]+")


def is_email(value: str) -> bool:
    """Whether a whole value is a single email address."""
    return "@" in value and EMAIL.fullmatch(value) is not None


def is_phone(value: str) -> bool:
    """Whether a value has the shape of a phone number.

    It needs 9 to 15 digits in groups, and either a leading plus or at least two separators. A run
    of plain digits never counts, because postal codes, account numbers and ids look like that. Nor
    does an IPv4 address, a date (too few digits), or a social security number, which is written
    with dashes in groups of 3, 2 and 4.
    """
    if not 9 <= len(value) <= 30 or not PHONE_SHAPE.fullmatch(value):
        return False
    if _IPV4.fullmatch(value) or SSN_SHAPE.fullmatch(value):
        return False
    digits = sum(character in _DIGITS for character in value)
    separators = sum(character in " .-()" for character in value)
    return 9 <= digits <= 15 and (value.startswith("+") or separators >= 2)


def has_embedded_phone(text: str) -> bool:
    """Whether a longer text contains a phone-shaped run of digits, tested by ``is_phone``."""
    return any(is_phone(found.group()) for found in PHONE_EMBEDDED.finditer(text))


def has_contact_detail(text: str) -> bool:
    """Whether a longer text contains an email address or a phone number."""
    return ("@" in text and EMAIL.search(text) is not None) or has_embedded_phone(text)


def is_us_ssn(value: str) -> bool:
    """A US social security number in ``123-45-6789`` form, without the numbers never issued."""
    if len(value) != 11 or not SSN_SHAPE.fullmatch(value):
        return False
    area, group, serial = value.split("-")
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def is_uk_nino(value: str) -> bool:
    """A UK national insurance number, without the prefixes that are not allocated."""
    compact = value.replace(" ", "")
    return _NINO.fullmatch(compact) is not None and compact[:2] not in _NINO_UNUSED_PREFIXES


def is_iban(value: str) -> bool:
    """An IBAN whose mod-97 checksum is correct.

    The country-specific length is not checked, only the general shape and the checksum, so a
    value can pass without being a real account.
    """
    compact = value.replace(" ", "")
    if not 15 <= len(compact) <= 34 or _IBAN.fullmatch(compact) is None:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(character, 36)) for character in rearranged)) % 97 == 1


NATIONAL_IDS = {"us_ssn": is_us_ssn, "uk_nino": is_uk_nino, "iban": is_iban}

# A personal name: one to four words, each starting with a capital letter and made of letters,
# with apostrophes, hyphens and a final full stop allowed, and short lowercase particles between.
_WORD = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:['’-][^\W\d_]+)*\.?"
_NAME = re.compile(rf"{_WORD}(?: (?:van|von|de|der|da|di|del|la|le|bin|al|{_WORD})){{0,3}}")


def is_name(value: str) -> bool:
    """Whether a whole value has the shape of a personal name: capitalized words, no digits."""
    return len(value) <= 60 and _NAME.fullmatch(value) is not None


def word_count(text: str) -> int:
    """Words as runs of letters, so a phone number written with spaces has none."""
    return len(_LETTER_RUN.findall(text))


# --- Column names -----------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

EMAIL_TOKENS = frozenset({"email", "emails", "mail"})
PHONE_TOKENS = frozenset({"phone", "phones", "telephone", "tel", "mobile", "fax", "msisdn"})
NATIONAL_ID_TOKENS = frozenset({"ssn", "nino", "iban", "passport"})
NATIONAL_ID_PAIRS = (
    ("social", "security"),
    ("national", "id"),
    ("national", "insurance"),
    ("tax", "id"),
)
# Words that turn a column named like personal data into something else: how many phone calls, the
# phone's model, an e-mail opt-in flag.
NOT_THE_VALUE_TOKENS = frozenset(
    {"model", "type", "brand", "plan", "os", "make", "count", "usage", "service", "provider"}
    | {"carrier", "calls", "minutes", "data", "price", "cost", "charge", "charges", "flag"}
    | {"has", "is", "status", "verified", "valid", "optin", "opt", "sent", "domain", "template"}
    | {"length", "rate"}
)

NAME_QUALIFIERS = frozenset({"first", "last", "middle", "given", "family", "sur", "full", "maiden"})
NAME_SINGLE_TOKENS = frozenset(
    {"surname", "firstname", "lastname", "forename", "fullname", "givenname", "familyname"}
    | {"middlename", "fname", "lname"}
)
FREE_TEXT_TOKENS = frozenset(
    {"comment", "comments", "note", "notes", "feedback", "message", "messages", "remark"}
    | {"remarks", "review", "reviews", "bio", "complaint", "complaints", "narrative"}
    | {"transcript"}
)


def tokens(name: str) -> frozenset[str]:
    """The lowercase words of a column name, split on non-letters and on camelCase."""
    words = re.split(r"[^a-z0-9]+", _CAMEL.sub(" ", name).lower())
    return frozenset(word for word in words if word)


def names_email(words: frozenset[str]) -> bool:
    """Whether a column name's words say it holds email addresses, such as ``email``."""
    return bool(words & EMAIL_TOKENS) and not words & NOT_THE_VALUE_TOKENS


def names_phone(words: frozenset[str]) -> bool:
    """Whether a column name's words say it holds phone numbers, such as ``mobile``."""
    return bool(words & PHONE_TOKENS) and not words & NOT_THE_VALUE_TOKENS


def names_national_id(words: frozenset[str]) -> bool:
    """Whether a column name's words say it holds a national identifier, such as ``ssn``."""
    named = bool(words & NATIONAL_ID_TOKENS) or any(
        set(pair) <= words for pair in NATIONAL_ID_PAIRS
    )
    return named and not words & NOT_THE_VALUE_TOKENS


def names_free_text(words: frozenset[str]) -> bool:
    """Whether a column name's words say it holds free text, such as ``comments``."""
    return bool(words & FREE_TEXT_TOKENS)


def names_a_person(words: frozenset[str]) -> str | None:
    """``"person"`` for a name that says it is a person's, ``"bare"`` for a plain ``name``.

    A plain ``name`` column could be a person, a product or a place, so it is told apart from a
    ``first_name`` or a ``surname``. ``None`` for anything else, such as ``product_name`` or
    ``user_name``, which have no word that says they are a person's.
    """
    if words & NAME_SINGLE_TOKENS or (words & NAME_QUALIFIERS and words & {"name", "names"}):
        return "person"
    if words <= {"name", "names"} and words:
        return "bare"
    return None
