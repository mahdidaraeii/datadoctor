"""The findings ``check_leakage`` reports, and the wording of each.

Every builder takes plain values, so this module knows nothing about how they were computed.
"""

from datadoctor.core.result import Finding, Severity

LISTED = 5


def _listed(names: list[str]) -> str:
    text = ", ".join(names[:LISTED])
    return text + (f" and {len(names) - LISTED} more" if len(names) > LISTED else "")


def near_perfect_predictor_finding(columns: list[str], threshold: float) -> Finding:
    """A single feature's association with the target is high enough to warrant a closer look."""
    plural = len(columns) != 1
    return Finding(
        category="leakage",
        severity=Severity.HIGH,
        confidence=0.6,
        title="A feature is a near-perfect single-column predictor",
        evidence=(
            f"{_listed(columns)} {'have' if plural else 'has'} an association with the target "
            f"of at least {threshold:.2f}, on the same 0 to 1 scale as an absolute correlation "
            "regardless of which of Pearson correlation, the correlation ratio or Cramer's V "
            "produced it."
        ),
        interpretation=(
            "An association this strong from a single feature is unusual for a real, "
            "independently measured predictor and is worth checking for leakage: a feature "
            "that is only known, or only takes this value, once the target is already decided."
        ),
        limitations=(
            "This is a candidate, not a confirmed defect. A feature strongly related to the "
            "target is not by itself proof of leakage: a real, unusually strong predictor can "
            "clear this threshold too, and strength alone cannot tell them apart. This is most "
            "acute for a continuous feature against a classification target, where the "
            "correlation ratio has a much lower reachable ceiling than Pearson correlation or "
            "Cramer's V do (see the module notes); a continuous proxy for a binary label can "
            "look identical to a real strong predictor by this measure. This check detects "
            "determinism and suspicious naming, not strength on its own, and the threshold "
            "itself is a stated convention, not a law."
        ),
        affected_columns=tuple(columns),
        recommendation=(
            "Check whether this feature could only be known after the target is determined, "
            "or how it was recorded, before using it in a model."
        ),
    )


def duplicated_target_finding(columns: list[str], threshold: float) -> Finding:
    """A feature's association with the target is close enough to 1.0 to be the target itself."""
    plural = len(columns) != 1
    return Finding(
        category="leakage",
        severity=Severity.CRITICAL,
        confidence=0.9,
        title="A feature appears to be the target under another name or encoding",
        evidence=(
            f"{_listed(columns)} {'have' if plural else 'has'} an association with the target "
            f"of at least {threshold:.3f}, on the same scale as an absolute correlation."
        ),
        interpretation=(
            "An association this close to a perfect one is not plausible for an independently "
            "measured feature. The column is very likely a duplicate, or a deterministic "
            "recoding, of the target itself."
        ),
        limitations=(
            "An extremely strong but genuine relationship, such as a physical unit conversion "
            "or a direct rule the target was defined by, can also reach this level without "
            "being a literal copy of the target; the threshold does not distinguish the two."
        ),
        affected_columns=tuple(columns),
        recommendation=(
            "Remove this column before modeling, or confirm it is not derived from the target."
        ),
    )


def suspicious_naming_finding(columns: list[str]) -> Finding:
    """A column's name suggests it records something decided at or after the outcome."""
    plural = len(columns) != 1
    return Finding(
        category="leakage",
        severity=Severity.MEDIUM,
        confidence=0.3,
        title="A column name suggests a post-outcome event",
        evidence=(
            f"{_listed(columns)} {'have' if plural else 'has'} a name suggesting a value "
            "recorded at or after the outcome, such as a cancellation, closure, discharge or "
            "settlement."
        ),
        interpretation=(
            "A column recorded once the outcome is already known would make it unusable for "
            "prediction even if it turned out to be statistically predictive: it would not be "
            "available at the time a real prediction is needed."
        ),
        limitations=(
            "This is a weaker, name-based signal only: it does not check the column's actual "
            "values or its association with the target, so a matching name is not evidence the "
            "column actually causes a problem, and a column with the same issue under a "
            "different or non-English name is not caught."
        ),
        affected_columns=tuple(columns),
        recommendation=(
            "Confirm whether this column would be available at prediction time before using it."
        ),
    )
