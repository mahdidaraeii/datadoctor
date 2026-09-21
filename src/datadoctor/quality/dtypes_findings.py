"""The findings that ``check_dtypes`` reports, and the wording of each.

Only placeholder-like tokens are shown, never the values of a column that parses as numbers.
"""

from datadoctor.core.result import Finding, Severity

# How many columns a finding spells out. The rest are counted, and all are in the metrics.
LISTED = 5


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _verb(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _more(total: int, shown: int) -> str:
    return f" and {total - shown} more" if total > shown else ""


def _token(token: str) -> str:
    return f"'{token}'" if token else "(empty)"


def _describe_numbers(column: dict) -> str:
    if not column["non_numeric"]:
        return f"{column['name']} (every value parses as a number)"
    listed = ", ".join(f"{_token(t['token'])} x{t['count']:,}" for t in column["tokens"])
    rest = column["tokens_not_listed"]
    if rest and listed:
        listed += f" and {rest:,} other non-numeric {_plural(rest, 'value')}"
    elif rest:
        listed = f"{rest:,} non-numeric {_plural(rest, 'value')}, none short enough to list"
    return (
        f"{column['name']} ({column['numeric_share']:.1%} parse as numbers; other values: {listed})"
    )


def numbers_finding(columns: list[dict]) -> Finding:
    """Columns of text where the values are numbers, apart from a few other tokens."""
    ranked = sorted(columns, key=lambda c: (-c["non_numeric"], c["name"]))
    shown = [_describe_numbers(c) for c in ranked[:LISTED]]
    all_parse = all(not c["non_numeric"] for c in columns)
    return Finding(
        category="dtypes",
        severity=Severity.MEDIUM,
        confidence=0.9 if all_parse else 0.8,
        title="Numbers stored as text",
        evidence=(
            f"{len(columns)} text {_plural(len(columns), 'column')} "
            f"{_verb(len(columns), 'holds', 'hold')} values that parse as numbers: "
            + "; ".join(shown)
            + _more(len(columns), len(shown))
            + "."
        ),
        interpretation=(
            "A column of numbers stored as text is treated as text by every numeric analysis, "
            "so it is left out of means, correlations and outlier checks. When some values are "
            "not numbers, they are usually placeholders for missing data, and they are what "
            "keeps the column from loading as numbers."
        ),
        limitations=(
            "Recognized: plain numbers, and US-style numbers with thousands commas. Not "
            "recognized: currency symbols, percent signs and European decimal commas, since "
            "1.234,5 is ambiguous and is never guessed. Columns whose numbers have leading zeros "
            "are reported separately. Non-numeric tokens longer than 20 characters are not "
            "listed, because they are probably real values. Whether the column really measures a "
            "quantity depends on what it means."
        ),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Find out what the non-numeric tokens mean and treat them as missing if they are "
            "placeholders, then convert the column to numbers. Nothing was converted here."
        ),
    )


def codes_finding(columns: list[dict]) -> Finding:
    """Numeric-looking columns with leading zeros, which are probably codes."""
    ranked = sorted(columns, key=lambda c: (-c["leading_zero"], c["name"]))
    shown = [
        f"{c['name']} ({c['leading_zero']:,} {_plural(c['leading_zero'], 'value')} with a "
        "leading zero)"
        for c in ranked[:LISTED]
    ]
    return Finding(
        category="dtypes",
        severity=Severity.INFO,
        confidence=0.6,
        title="Numeric-looking columns with leading zeros",
        evidence=(
            f"{len(columns)} text {_plural(len(columns), 'column')} "
            f"{_verb(len(columns), 'looks', 'look')} numeric but "
            f"{_verb(len(columns), 'has', 'have')} values with leading zeros: "
            + "; ".join(shown)
            + _more(len(columns), len(shown))
            + "."
        ),
        interpretation=(
            "Values such as 01234 are usually codes, like postcodes or product and account "
            "numbers, and not quantities. Converting them to numbers would drop the leading "
            "zeros, and that cannot be undone. This is a heuristic based on the leading zeros."
        ),
        limitations=(
            "A leading zero is only a hint. A column of codes without leading zeros, such as "
            "five-digit postcodes that all start with other digits, cannot be told from numbers "
            "this way."
        ),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Keep these columns as text unless the values are truly quantities. Nothing was "
            "converted here."
        ),
    )


def _describe_lost_zeros(column: dict, n_rows: int) -> str:
    checked = column["checked"]
    scope = f"the {checked:,}" if checked == n_rows else f"the first {checked:,}"
    text = f"{column['name']} ({column['leading_zero']:,} of {scope} {_plural(checked, 'row')}"
    if column["width"] is not None:
        text += f"; every value was {column['width']} characters wide"
    return text + ")"


def lost_zeros_finding(columns: list[dict], n_rows: int) -> Finding:
    """Columns that were read as numbers although their values had leading zeros in the file."""
    ranked = sorted(columns, key=lambda c: (-c["leading_zero"], c["name"]))
    shown = [_describe_lost_zeros(c, n_rows) for c in ranked[:LISTED]]
    return Finding(
        category="dtypes",
        severity=Severity.MEDIUM,
        confidence=0.7,
        title="Leading zeros were lost when the file was read",
        evidence=(
            f"{len(columns)} numeric {_plural(len(columns), 'column')} had values with a leading "
            "zero in the file, and the zeros were dropped when the numbers were read: "
            + "; ".join(shown)
            + _more(len(columns), len(shown))
            + "."
        ),
        interpretation=(
            "Values such as 01234 are usually codes, like postcodes or product and account "
            "numbers, and not quantities. The loaded column holds 1234, so the zeros cannot be "
            "recovered from it, and joins or comparisons with data that keeps the zeros will not "
            "match. The file itself is unchanged. This is a heuristic: leading zeros suggest "
            "codes but do not prove them."
        ),
        limitations=(
            "The file's own text was compared with the loaded numbers, but only the first "
            "100,000 rows of each column, so a leading zero further down is not counted. Only "
            "columns that were read as whole, non-negative numbers are examined. Codes with no "
            "leading zeros, such as five-digit postcodes that all start with other digits, "
            "cannot be recognized this way."
        ),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Read these columns as text, for example with pandas.read_csv and dtype=str for the "
            "column, and give the frame to Dataset. If every value had the same width, as "
            "recorded above, padding the loaded numbers to that width with zfill restores the "
            "codes. Nothing was converted here."
        ),
    )


def mixed_finding(columns: list[dict]) -> Finding:
    """Columns whose values are of more than one Python type."""
    ranked = sorted(columns, key=lambda c: (-(c["non_null"] - max(c["types"].values())), c["name"]))
    shown = [
        f"{c['name']} ({', '.join(f'{kind} {count:,}' for kind, count in c['types'].items())})"
        for c in ranked[:LISTED]
    ]
    return Finding(
        category="dtypes",
        severity=Severity.MEDIUM,
        confidence=1.0,
        title="Columns mix value types",
        evidence=(
            f"{len(columns)} {_plural(len(columns), 'column')} "
            f"{_verb(len(columns), 'holds', 'hold')} more than one type of value: "
            + "; ".join(shown)
            + _more(len(columns), len(shown))
            + "."
        ),
        interpretation=(
            "A column that mixes types breaks arithmetic and sorting, and usually comes from "
            "spreadsheet cells typed differently, or from a JSON field that is sometimes a "
            "number and sometimes text."
        ),
        limitations=(
            "Types are counted on the values as loaded. A text column whose values merely look "
            "like numbers is not mixed, and is reported as numbers stored as text instead."
        ),
        affected_columns=tuple(c["name"] for c in columns),
        recommendation=(
            "Decide which type each column should have, then correct the values that do not "
            "fit. Nothing was converted here."
        ),
    )
