"""Deterministic per-car linear fit for static garage ride height.

Static garage ``Chassis.{LeftFront,RightFront,LeftRear,RightRear}.RideHeight``
is a kinematic function of platform inputs given the car's installation
ratios: perches, pushrods, heave + third springs, torsion bar turns + OD,
camber, toe, fuel. It is NOT an empirical telemetry channel and has no
environmental dependence. iRacing computes it deterministically from the
garage YAML.

The previous predict path routed this through the Ridge/Forest surrogate
trained on per-(corner, phase, channel) tuples. That made the static RH
prediction a noisy regressor with near-zero gradient against pushrod and
perch -- a 6 mm pushrod move would shift predicted static RH by 0.1 mm,
which violates the physics. This module replaces the surrogate for the
four static RH channels with a per-car closed-form linear fit gated on
``R^2 >= 0.98`` per channel. See ``docs/accuracy-rebuild-2026-05-24/PLAN.md``
P0.2.

When the fit ships, ``PhysicsModel.predict_setup_readouts`` returns the
kinematic prediction for the four static RH channels. When the fit
fails (insufficient corpus, novel feature combination, etc.) the
surrogate fallback remains so the API still returns a value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Platform-input features the static RH fit consumes. Matches the
# coarse-platform feature list used by the legacy k-NN repair path so
# the upstream "what is a platform parameter" answer is consistent.
# Order is stable -- the coefficient vector indexes by this tuple.
STATIC_RH_FEATURES: tuple[str, ...] = (
    "heave_spring_rate_n_per_mm",
    "third_spring_rate_n_per_mm",
    "heave_perch_offset_front_mm",
    "spring_perch_offset_rear_mm",
    "pushrod_length_offset_front_mm",
    "pushrod_length_offset_rear_mm",
    "torsion_bar_turns_fl",
    "torsion_bar_od_fl_mm",
    "torsion_bar_turns_rl",
    "torsion_bar_od_rl_mm",
    "fuel_level_l",
    "camber_fl_deg",
    "camber_rl_deg",
    "toe_front_mm",
    "toe_rl_mm",
)

STATIC_RH_CHANNELS: tuple[str, ...] = (
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
)

# Fit must reach this R^2 to ship. Below this, the linear approximation
# is missing structure (e.g. unmodelled tyre-pressure compression,
# non-linear pushrod travel) and the surrogate fallback is no worse than
# shipping a partial fit.
_R2_SHIP_THRESHOLD: float = 0.98

# Minimum number of sessions before any fit is attempted. Linear least
# squares with k features needs strictly more than k samples to be
# well-conditioned; we keep a margin for feature variance-pruning.
_MIN_SAMPLES: int = 12

# Per-feature minimum standard deviation (in feature units) for the
# feature to be retained in the fit. Below this, the column is
# effectively constant and the corresponding coefficient is unidentified.
_MIN_FEATURE_STD: float = 1e-6

# Ridge regularisation. Tiny by design -- this is OLS for all practical
# purposes; ridge is here only as a numerical safeguard against
# near-collinear feature pairs (e.g. heave perch + pushrod both moving
# the front platform together in some sessions).
_RIDGE_LAMBDA: float = 1e-6


@dataclass(frozen=True, slots=True)
class StaticRhChannelFit:
    """Linear fit for one ``setup_static_*_ride_height_mm`` channel."""

    channel: str
    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    r2: float
    n_samples: int


@dataclass(frozen=True, slots=True)
class StaticRhKinematic:
    """Per-car bundle of per-channel kinematic fits."""

    car: str
    channels: dict[str, StaticRhChannelFit] = field(default_factory=dict)
    rejected_channels: tuple[str, ...] = ()
    n_sessions: int = 0

    def is_ready(self) -> bool:
        return bool(self.channels)


def _collect_training_matrix(
    sid_to_params: dict[str, dict[str, float]],
    sid_to_readouts: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Build the per-session (X, y_by_channel) arrays.

    A session is included when it has at least one platform feature and
    at least one static RH readout. Missing per-feature values are
    represented as NaN in ``X``; per-channel y arrays are returned with
    NaN for sessions missing that channel. Downstream ``_fit_channel``
    masks rows where the channel's y or any retained feature is NaN.
    """
    sids = sorted(sid_to_params)
    n_sessions = len(sids)
    n_features = len(STATIC_RH_FEATURES)
    x = np.full((n_sessions, n_features), np.nan, dtype=float)
    for i, sid in enumerate(sids):
        params = sid_to_params.get(sid) or {}
        for j, feat in enumerate(STATIC_RH_FEATURES):
            value = params.get(feat)
            if value is not None:
                try:
                    x[i, j] = float(value)
                except (TypeError, ValueError):
                    continue
    y_by_channel: dict[str, np.ndarray] = {}
    for channel in STATIC_RH_CHANNELS:
        col = np.full(n_sessions, np.nan, dtype=float)
        for i, sid in enumerate(sids):
            readouts = sid_to_readouts.get(sid) or {}
            value = readouts.get(channel)
            if value is not None:
                try:
                    col[i] = float(value)
                except (TypeError, ValueError):
                    continue
        y_by_channel[channel] = col
    return sids, x, y_by_channel


def _fit_channel(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float, list[int], int] | None:
    """Closed-form ridge least squares for one static RH channel.

    Returns ``(coefficients_aligned_to_input_columns, intercept, r2,
    retained_feature_indices, n_samples)``. ``None`` when the row mask
    yields fewer than ``_MIN_SAMPLES`` usable sessions or no feature
    survives the variance prune.
    """
    if x.shape[0] != y.shape[0]:
        return None
    row_mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n_samples = int(np.sum(row_mask))
    if n_samples < _MIN_SAMPLES:
        # Try the looser per-row mask: allow rows missing some features
        # by zeroing those features' contributions. We still require the
        # target to be present.
        row_mask = np.isfinite(y)
        n_samples = int(np.sum(row_mask))
        if n_samples < _MIN_SAMPLES:
            return None
    x_use = x[row_mask].copy()
    y_use = y[row_mask].copy()
    # Replace NaN features with the column median so they contribute
    # nothing to the fit (mean-centred design); equivalent to letting
    # the intercept absorb the missing-value baseline.
    for j in range(x_use.shape[1]):
        col = x_use[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            x_use[:, j] = 0.0
        else:
            median = float(np.median(finite))
            col[~np.isfinite(col)] = median
            x_use[:, j] = col
    stds = x_use.std(axis=0, ddof=0)
    retained = [int(j) for j in range(x_use.shape[1]) if stds[j] >= _MIN_FEATURE_STD]
    if not retained:
        return None
    x_kept = x_use[:, retained]
    n_features_kept = x_kept.shape[1]
    if n_samples <= n_features_kept:
        # Strictly under-determined even after pruning. Ridge can still
        # produce a numerical answer, but the resulting fit is not a
        # kinematic statement -- refuse rather than ship a
        # falsely-confident result.
        return None
    x_design = np.hstack([np.ones((n_samples, 1)), x_kept])
    gram = x_design.T @ x_design
    gram += _RIDGE_LAMBDA * np.eye(gram.shape[0])
    try:
        beta = np.linalg.solve(gram, x_design.T @ y_use)
    except np.linalg.LinAlgError:
        return None
    intercept = float(beta[0])
    kept_coefs = beta[1:]
    y_hat = x_design @ beta
    ss_res = float(np.sum((y_use - y_hat) ** 2))
    ss_tot = float(np.sum((y_use - float(np.mean(y_use))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    full_coefs = np.zeros(x_use.shape[1], dtype=float)
    for slot, j in enumerate(retained):
        full_coefs[j] = float(kept_coefs[slot])
    return full_coefs, intercept, float(r2), retained, n_samples


def fit_static_rh_kinematic(
    car: str,
    sid_to_params: dict[str, dict[str, float]],
    sid_to_readouts: dict[str, dict[str, float]],
) -> StaticRhKinematic:
    """Fit the per-car kinematic linear surface for static RH readouts.

    Returns a ``StaticRhKinematic`` carrying every channel whose per-
    channel fit reached ``R^2 >= _R2_SHIP_THRESHOLD``. Channels that
    fail the gate are listed in ``rejected_channels`` so callers can
    surface "kinematic fit refused" notices instead of silently falling
    back to the surrogate.
    """
    _sids, x, y_by_channel = _collect_training_matrix(
        sid_to_params, sid_to_readouts,
    )
    n_sessions = x.shape[0]
    if n_sessions < _MIN_SAMPLES:
        return StaticRhKinematic(
            car=car, channels={}, rejected_channels=STATIC_RH_CHANNELS,
            n_sessions=n_sessions,
        )
    channels: dict[str, StaticRhChannelFit] = {}
    rejected: list[str] = []
    for channel in STATIC_RH_CHANNELS:
        result = _fit_channel(x, y_by_channel[channel])
        if result is None:
            rejected.append(channel)
            continue
        coefs, intercept, r2, _retained, n_samples = result
        if r2 < _R2_SHIP_THRESHOLD:
            rejected.append(channel)
            continue
        channels[channel] = StaticRhChannelFit(
            channel=channel,
            features=STATIC_RH_FEATURES,
            coefficients=tuple(float(c) for c in coefs),
            intercept=float(intercept),
            r2=float(r2),
            n_samples=int(n_samples),
        )
    return StaticRhKinematic(
        car=car,
        channels=channels,
        rejected_channels=tuple(rejected),
        n_sessions=n_sessions,
    )


def predict_static_rh_kinematic(
    kinematic: StaticRhKinematic | None,
    setup: dict[str, float],
) -> dict[str, float]:
    """Evaluate the per-car linear fit at ``setup``.

    Returns one entry per channel the fit shipped (channels in
    ``rejected_channels`` are omitted so the caller can fall back to
    the surrogate for those specific channels). When ``kinematic`` is
    ``None`` or empty, returns an empty dict.
    """
    if kinematic is None or not kinematic.channels:
        return {}
    out: dict[str, float] = {}
    for channel, fit in kinematic.channels.items():
        value = float(fit.intercept)
        for feat, coef in zip(fit.features, fit.coefficients, strict=True):
            raw = setup.get(feat)
            if raw is None:
                continue
            try:
                value += float(coef) * float(raw)
            except (TypeError, ValueError):
                continue
        out[channel] = value
    return out


__all__ = [
    "STATIC_RH_CHANNELS",
    "STATIC_RH_FEATURES",
    "StaticRhChannelFit",
    "StaticRhKinematic",
    "fit_static_rh_kinematic",
    "predict_static_rh_kinematic",
]
