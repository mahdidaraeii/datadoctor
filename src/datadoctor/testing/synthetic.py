"""Synthetic dataset generator for validating diagnostics against known ground truth.

Not part of the CLI surface. The leakage and split-strategy diagnostics (part 3) are tested
against datasets built here, where every planted defect's location and strength is known by
construction, rather than guessed at after the fact from a real dataset.

Determinism relies on ``numpy.random.RandomState``, not ``default_rng``: ``RandomState``'s bit
stream is frozen and numpy guarantees it stable across numpy versions, which is exactly what
``core/guardrails.py`` already relies on for the same reason. ``default_rng``'s ``Generator``
carries no such cross-version guarantee.

Three mechanisms plant an association with the target, and all three use the same trick: draw a
signal, draw independent noise, then remove any incidental sample correlation between the two
with a Gram-Schmidt projection before mixing them. That makes the requested strength exact for a
*regression* target, whose stored value is the continuous latent itself. A *classification*
target is a threshold of that latent, and thresholding has no closed-form inverse, so for
classification the requested strength is a construction parameter only: a high value makes the
planted structure detectable, but the discrete labels will not reproduce it as an exact
statistic. Tests reflect this split: exact-tolerance assertions are for regression, and
classification is checked for detectability instead.

When more than one mechanism is requested together (for example ``temporal=True`` with
``groups=...``), each later mechanism is mixed against the latent the earlier ones produced
rather than fresh noise, so structure composes instead of being overwritten. The exact-strength
guarantee above holds only for the mechanism applied last in the fixed order below; earlier
mechanisms are diluted by later ones. Tests exercise each mechanism in isolation.

Fixed draw order for a single call: base latent, then temporal, then groups, then the base
feature columns, then the ``leak_future`` column, then the target, then label noise, then the
``leak_duplicate`` column (built from the final, noised target). This order never changes with
which optional arguments are given, so identical arguments and seed always reproduce the same
dataset.

Column names are fixed and documented rather than returned as metadata, since they can be read
back from the dataset directly: ``target``, ``feature_0..feature_{n-1}``, ``leak_future``,
``leak_duplicate``, ``event_time``, ``group_id``. Only the label-noise indices cannot be read
back from the data, so those alone are returned alongside the dataset.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from datadoctor.core.dataset import Dataset
from datadoctor.core.exceptions import ConfigError

_LEAKAGE_MECHANISMS = ("future", "duplicate")
_TASKS = ("classification", "regression")
_EFFECT_KEYS = ("leakage", "temporal", "group")

_BASE_FEATURE_STRENGTH = 0.6
_DEFAULT_LEAKAGE_STRENGTH = 0.98
_DEFAULT_TEMPORAL_STRENGTH = 0.5
_DEFAULT_GROUP_STRENGTH = 0.5
_EPOCH = pd.Timestamp("2020-01-01")


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticConfig:
    """The base shape of a synthetic problem, before any defect is planted.

    Attributes:
        n_rows: Number of rows to generate. At least 10, so the mechanisms below have enough
            rows for their planted structure to be meaningful.
        n_features: Number of base numeric features. Each is a real, moderate-strength
            predictor of the target, independent of any planted defect.
        random_seed: Seed for every random draw. The same config, the same keyword arguments to
            ``make_synthetic_dataset`` and the same seed always produce the same dataset.
        n_classes: Number of classes. Used only when ``task="classification"``.
    """

    n_rows: int
    n_features: int
    random_seed: int
    n_classes: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.n_rows, int) or isinstance(self.n_rows, bool) or self.n_rows < 10:
            raise ConfigError(f"n_rows must be an int of at least 10, got {self.n_rows!r}")
        if (
            not isinstance(self.n_features, int)
            or isinstance(self.n_features, bool)
            or self.n_features < 1
        ):
            raise ConfigError(f"n_features must be an int of at least 1, got {self.n_features!r}")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ConfigError(f"random_seed must be an int, got {self.random_seed!r}")
        if self.random_seed < 0:
            raise ConfigError(f"random_seed must be zero or greater, got {self.random_seed}")
        if (
            not isinstance(self.n_classes, int)
            or isinstance(self.n_classes, bool)
            or self.n_classes < 2
        ):
            raise ConfigError(f"n_classes must be an int of at least 2, got {self.n_classes!r}")


class SyntheticDataset(NamedTuple):
    """A generated dataset together with the one piece of ground truth it cannot expose itself.

    Attributes:
        dataset: The generated dataset. Planted structure lives in fixed, documented columns
            (see the module docstring), present only when requested.
        flipped_indices: Row positions, sorted, whose target was replaced by the label noise
            mechanism. Empty when ``label_noise`` is 0.0.
    """

    dataset: Dataset
    flipped_indices: tuple[int, ...]


def make_synthetic_dataset(
    config: SyntheticConfig,
    *,
    leakage: str | tuple[str, ...] | None = None,
    temporal: bool = False,
    groups: int | None = None,
    label_noise: float = 0.0,
    effect_sizes: dict[str, float] | None = None,
    task: Literal["classification", "regression"] = "classification",
) -> SyntheticDataset:
    """Build a dataset with known, planted structure for testing diagnostics against.

    Args:
        config: The base problem shape.
        leakage: Which leakage mechanism to plant, or both as a tuple. ``"future"`` adds
            ``leak_future``, a numeric column built from the pre-threshold latent so it carries
            a near-deterministic association with the target. ``"duplicate"`` adds
            ``leak_duplicate``, the final target re-encoded under different values (a relabeled
            category, or an affine rescale for regression) with no noise added.
        temporal: Plant a datetime column ``event_time`` whose value is associated with the
            target, so a random split would leak future information into training.
        groups: Number of entities to assign rows to, added as ``group_id``. The target is
            associated with group membership, so a non-grouped split would leak an entity's
            label across the split. Must be at least 1 and less than ``config.n_rows``.
        label_noise: Fraction of rows, from 0.0 up to but excluding 1.0, whose target is
            replaced after every other mechanism has run. For classification, each selected
            row's class is changed to a different class chosen uniformly at random. For
            regression, which has no discrete class to flip to, each selected row's value is
            replaced with another row's value, chosen uniformly at random.
        effect_sizes: Strength of each requested mechanism, by name: ``"leakage"`` and
            ``"temporal"`` are a Pearson correlation in ``[-1.0, 1.0]`` against the latent
            (exact, to floating-point tolerance, for a regression target); ``"group"`` is an
            eta-squared in ``[0.0, 1.0]`` (same exactness note). Defaults are 0.98, 0.5 and 0.5.
            Unknown keys are rejected.
        task: ``"classification"`` (the default) thresholds the latent into ``config.n_classes``
            roughly balanced classes. ``"regression"`` stores the latent itself as the target.

    Returns:
        The dataset and the label-noise ground truth. See the module docstring for the fixed
        column names, the draw order and the scope of the exactness guarantee.
    """
    if not isinstance(config, SyntheticConfig):
        raise TypeError(f"config must be a SyntheticConfig, got {type(config).__name__}")
    if task not in _TASKS:
        raise ConfigError(f"task must be one of {_TASKS}, got {task!r}")
    mechanisms = _normalize_leakage(leakage)
    if not 0.0 <= label_noise < 1.0:
        raise ConfigError(f"label_noise must be at least 0.0 and below 1.0, got {label_noise}")
    if groups is not None and not (1 <= groups < config.n_rows):
        raise ConfigError(f"groups must be at least 1 and below n_rows, got {groups}")
    strengths = _validate_effect_sizes(effect_sizes)

    rng = np.random.RandomState(config.random_seed)
    n_rows = config.n_rows
    columns: dict[str, object] = {}

    latent = _standardize(rng.normal(size=n_rows))

    if temporal:
        time_rank = rng.permutation(n_rows).astype(float)
        z_time = _standardize(time_rank)
        strength = strengths.get("temporal", _DEFAULT_TEMPORAL_STRENGTH)
        latent = _exact_correlation(z_time, strength, rng, base=latent)
        columns["event_time"] = _EPOCH + pd.to_timedelta(time_rank.astype("int64"), unit="D")

    if groups is not None:
        group_ids = rng.randint(0, groups, size=n_rows)
        strength = strengths.get("group", _DEFAULT_GROUP_STRENGTH)
        latent = _exact_group_effect(group_ids, groups, strength, rng, base=latent)
        columns["group_id"] = [f"group_{g:03d}" for g in group_ids]

    for i in range(config.n_features):
        columns[f"feature_{i}"] = _exact_correlation(latent, _BASE_FEATURE_STRENGTH, rng)

    if "future" in mechanisms:
        strength = strengths.get("leakage", _DEFAULT_LEAKAGE_STRENGTH)
        columns["leak_future"] = _exact_correlation(latent, strength, rng)

    target = _build_target(latent, task, config.n_classes)
    target, flipped = _apply_label_noise(target, label_noise, rng, task, config.n_classes)

    if "duplicate" in mechanisms:
        columns["leak_duplicate"] = _duplicate_target(target, task, config.n_classes, rng)

    columns["target"] = target
    frame = pd.DataFrame(columns)
    dataset = Dataset(data=frame, name="synthetic", target="target")
    return SyntheticDataset(dataset=dataset, flipped_indices=flipped)


def _normalize_leakage(leakage: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if leakage is None:
        return ()
    mechanisms = (leakage,) if isinstance(leakage, str) else tuple(leakage)
    for mechanism in mechanisms:
        if mechanism not in _LEAKAGE_MECHANISMS:
            raise ConfigError(f"leakage must be one of {_LEAKAGE_MECHANISMS}, got {mechanism!r}")
    return mechanisms


def _validate_effect_sizes(effect_sizes: dict[str, float] | None) -> dict[str, float]:
    if effect_sizes is None:
        return {}
    unknown = sorted(set(effect_sizes) - set(_EFFECT_KEYS))
    if unknown:
        raise ConfigError(f"unknown effect_sizes keys: {unknown}")
    for key in ("leakage", "temporal"):
        if key in effect_sizes and not -1.0 <= effect_sizes[key] <= 1.0:
            raise ConfigError(f"effect_sizes[{key!r}] must be between -1.0 and 1.0")
    if "group" in effect_sizes and not 0.0 <= effect_sizes["group"] <= 1.0:
        raise ConfigError("effect_sizes['group'] must be between 0.0 and 1.0")
    return dict(effect_sizes)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    return centered / centered.std()


def _exact_correlation(
    z: np.ndarray, strength: float, rng: np.random.RandomState, base: np.ndarray | None = None
) -> np.ndarray:
    """A standardized series with sample Pearson correlation ``strength`` to standardized ``z``.

    ``base`` supplies the raw material that is orthogonalized against ``z`` to fill the
    remaining variance: a previous mechanism's latent when composing, or ``None`` for fresh
    noise. The Gram-Schmidt projection removes any incidental sample correlation exactly, so the
    result's correlation with ``z`` matches ``strength`` to floating-point precision.
    """
    raw = rng.normal(size=z.size) if base is None else np.asarray(base, dtype=float)
    centered = raw - raw.mean()
    residual = centered - (np.dot(centered, z) / np.dot(z, z)) * z
    residual = _standardize(residual)
    return strength * z + np.sqrt(1.0 - strength**2) * residual


def _exact_group_effect(
    group_ids: np.ndarray,
    n_groups: int,
    strength: float,
    rng: np.random.RandomState,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """A standardized series whose eta-squared against ``group_ids`` is exactly ``strength``.

    Mirrors ``_exact_correlation`` for a categorical predictor: a per-group effect is constant
    within each group, so all of its variance is between-group; the noise is centered within
    each group so none of its variance is. Mixing the two by ``sqrt(strength)``/
    ``sqrt(1 - strength)`` then gives an exact between-group variance share.
    """
    group_effect = rng.normal(size=n_groups)
    signal = _standardize(group_effect[group_ids])

    raw = rng.normal(size=group_ids.size) if base is None else np.asarray(base, dtype=float)
    group_means = pd.Series(raw).groupby(group_ids).transform("mean").to_numpy()
    noise = _standardize(raw - group_means)

    return np.sqrt(strength) * signal + np.sqrt(1.0 - strength) * noise


def _build_target(latent: np.ndarray, task: str, n_classes: int) -> np.ndarray | pd.Categorical:
    if task == "regression":
        return latent.copy()
    edges = np.quantile(latent, np.linspace(0.0, 1.0, n_classes + 1)[1:-1])
    codes = np.searchsorted(edges, latent, side="right")
    return pd.Categorical.from_codes(codes, categories=[f"class_{c}" for c in range(n_classes)])


def _apply_label_noise(
    target: np.ndarray | pd.Categorical,
    label_noise: float,
    rng: np.random.RandomState,
    task: str,
    n_classes: int,
) -> tuple[np.ndarray | pd.Categorical, tuple[int, ...]]:
    n_rows = len(target)
    k = round(label_noise * n_rows)
    if k == 0:
        return target, ()

    flipped = np.sort(rng.choice(n_rows, size=k, replace=False))
    if task == "classification":
        codes = np.asarray(target.codes).copy()
        offsets = rng.randint(1, n_classes, size=k)
        codes[flipped] = (codes[flipped] + offsets) % n_classes
        new_target = pd.Categorical.from_codes(codes, categories=target.categories)
    else:
        values = np.asarray(target, dtype=float).copy()
        partners = rng.randint(0, n_rows - 1, size=k)
        partners = np.where(partners >= flipped, partners + 1, partners)
        values[flipped] = np.asarray(target, dtype=float)[partners]
        new_target = values
    return new_target, tuple(int(i) for i in flipped)


def _duplicate_target(
    target: np.ndarray | pd.Categorical, task: str, n_classes: int, rng: np.random.RandomState
) -> np.ndarray | pd.Categorical:
    if task == "regression":
        return np.asarray(target, dtype=float) * 1000.0 + 7.0
    permutation = rng.permutation(n_classes)
    relabeled = [f"L{permutation[code]}" for code in np.asarray(target.codes)]
    return pd.Categorical(relabeled)
