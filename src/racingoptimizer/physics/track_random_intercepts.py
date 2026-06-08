"""Per-track random intercepts (P2.2 of accuracy-rebuild-2026-05-24).

Closed-form empirical-Bayes one-way random-intercept model on the
surrogate's per-track residuals, fit ONCE per output channel at training
time and applied additively at predict time.

Key distinction vs the retired ``per_track_residuals`` (P0.1):

  ``per_track_residuals`` = track_median(actual) - global_median(actual)

That is the difference of two raw observation aggregates -- it does NOT
account for the surrogate's predictions, double-counts track bias the
surrogate has already learned, and (worse) flattens the setup -> output
gradient DE depends on because the offset is applied uniformly on top of
the surrogate's already-track-aware output.

The P2.2 model is properly residual-based:

  residual_{t,i} = actual_{t,i} - surrogate_predict_{t,i}
  residual_{t,i} = alpha_t + epsilon_{t,i}
  alpha_t ~ Normal(0, tau^2)
  epsilon ~ Normal(0, sigma^2)

The prior is centred on ZERO (residuals already have the surrogate's
mean removed). Empirical-Bayes estimates ``tau^2`` and ``sigma^2`` via
method-of-moments, then shrinks each track's residual mean toward zero
in proportion to within-track noise vs between-track signal. Low-sample
tracks shrink hard (alpha_t ~ 0, no correction); high-sample tracks
with systematic residual divergence retain a real intercept.

Applied additively at predict time: the surrogate's setup gradient
``d mu / d setup`` is unchanged (alpha_t doesn't depend on setup). The
predicted level shifts per track; the bracket widens by alpha_t's
posterior uncertainty.

Pre-existing math base: ``physics/bayes_retrofit.py`` (same conjugate-
Gaussian framework, different prior centring -- bayes_retrofit shrinks
to the grand mean; this module shrinks to zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import pvariance


@dataclass(frozen=True, slots=True)
class TrackIntercept:
    """Per-(channel, track) residual correction with shrinkage.

    Attributes:
        channel: Output channel name (e.g. ``"accel_lat_g_max"``).
        track: Catalog track slug (e.g. ``"spa_2024_up"``).
        intercept: Posterior mean ``alpha_t``. Added to the surrogate's
            ``mu`` at predict time. Sign convention: positive means the
            surrogate under-predicts this channel on this track.
        intercept_std: Posterior uncertainty in ``alpha_t``. Widens the
            prediction's CI at predict time (in quadrature with the
            surrogate's ``sigma``). High = track has few residual samples
            or noisy residuals; correction unreliable.
        n_samples: Number of residual samples that fed this track's fit.
        shrinkage: ``1 - lambda_t`` in ``[0, 1]``. 1 = full shrinkage to
            zero (no correction applied because residual mean is
            indistinguishable from noise); 0 = no shrinkage (residual
            mean is far from zero relative to within-track noise).
    """
    channel: str
    track: str
    intercept: float
    intercept_std: float
    n_samples: int
    shrinkage: float


# Numerical floor for variance estimates. Below this the posterior maths
# divides by zero (lambda_t denominator). Well below any realistic
# residual scale on the channels we care about.
_VAR_FLOOR: float = 1e-9

# W6 P2: in-sample residual fits underestimate out-of-fold intercept
# variance (coverage dropped on grip channels post-P2.2). Inflate the
# posterior std used for CI widening so held-out actuals are not
# treated as surprises the model was over-confident about.
_INTERCEPT_STD_OOF_INFLATION: float = 2.0


def fit_per_channel(
    per_track_residuals: dict[str, list[float] | tuple[float, ...]],
    *,
    channel_name: str = "",
    min_tracks: int = 2,
    min_samples_per_track: int = 1,
) -> dict[str, TrackIntercept]:
    """Fit the closed-form random-intercept model for one output channel.

    Args:
        per_track_residuals: ``{track: [residual_1, residual_2, ...]}``.
            Each residual is ``actual - surrogate_prediction``.
        channel_name: Tagged on the returned :class:`TrackIntercept`
            instances; not used in the maths.
        min_tracks: Below this many tracks (after the min-sample filter),
            fall back to the per-track empirical mean without
            cross-track partial pooling -- shrinkage is zero because
            we cannot estimate ``tau^2`` from a single track.
        min_samples_per_track: Drop tracks with fewer residuals than
            this. Singleton tracks contribute no within-track variance
            and would degrade the hyperparameter estimation.

    Returns:
        ``{track: TrackIntercept}``. Tracks filtered out (insufficient
        samples) are absent.

    Determinism: pure-Python ``statistics``; same input -> identical
    output.
    """
    tracks: dict[str, tuple[float, ...]] = {
        t: tuple(float(v) for v in values)
        for t, values in per_track_residuals.items()
        if len(values) >= min_samples_per_track
    }
    if not tracks:
        return {}

    track_means: dict[str, float] = {
        t: sum(v) / len(v) for t, v in tracks.items()
    }
    track_n: dict[str, int] = {t: len(v) for t, v in tracks.items()}

    if len(tracks) < min_tracks:
        # Degraded path: empirical per-track mean, zero shrinkage.
        out: dict[str, TrackIntercept] = {}
        for t, vals in tracks.items():
            n = len(vals)
            mu_t = track_means[t]
            var_t = pvariance(vals) if n > 1 else 0.0
            std_t = sqrt(var_t / n) if n > 0 else 0.0
            out[t] = TrackIntercept(
                channel=channel_name,
                track=t,
                intercept=mu_t,
                intercept_std=std_t,
                n_samples=n,
                shrinkage=0.0,
            )
        return out

    # Pooled within-track variance via classical one-way ANOVA SS_within.
    total_within_ss = 0.0
    total_within_df = 0
    for vals in tracks.values():
        n_t = len(vals)
        if n_t < 2:
            continue
        total_within_ss += pvariance(vals) * n_t
        total_within_df += (n_t - 1)
    if total_within_df == 0:
        # All tracks are singletons (filtered above raises min_samples)
        # OR every track has constant residuals. Use between-track
        # spread as the noise floor so the maths stays well-defined.
        sigma_eps_sq = max(_VAR_FLOOR, pvariance(list(track_means.values())))
    else:
        sigma_eps_sq = max(_VAR_FLOOR, total_within_ss / total_within_df)

    # Between-track variance via method-of-moments. The prior is centred
    # on zero, so S_between is the un-grand-mean-subtracted version --
    # we want the variance of the track means around zero, not around
    # their grand mean. (For mean-zero residuals from a well-trained
    # surrogate the grand mean IS zero, and the two forms coincide; but
    # the zero-centred form is the correct posterior maths regardless of
    # whether the surrogate is well-trained.)
    n_tracks = len(tracks)
    s_between = sum(
        track_n[t] * (track_means[t] ** 2) for t in tracks
    ) / max(1, n_tracks - 1)
    avg_inv_n = sum(1.0 / track_n[t] for t in tracks) / n_tracks
    tau_sq = max(0.0, s_between - sigma_eps_sq * avg_inv_n)

    # Per-track posterior. Prior is N(0, tau^2); likelihood is N(mu_t,
    # sigma_eps^2 / n_t). Conjugate update gives a normal posterior
    # centred on (tau^2 / (tau^2 + sigma^2/n)) * mu_t, shrunk toward 0.
    out: dict[str, TrackIntercept] = {}
    for t in tracks:
        n_t = track_n[t]
        denom = tau_sq + sigma_eps_sq / n_t
        if denom <= 0:
            lambda_t = 0.0
        else:
            lambda_t = tau_sq / denom
        post_mean = lambda_t * track_means[t]
        # mean_var: uncertainty in alpha_t = (1 - lambda_t) * sigma^2 / n
        # (the residual variance after partial pooling).
        mean_var = (1.0 - lambda_t) * sigma_eps_sq / n_t
        out[t] = TrackIntercept(
            channel=channel_name,
            track=t,
            intercept=post_mean,
            intercept_std=sqrt(max(0.0, mean_var)),
            n_samples=n_t,
            shrinkage=1.0 - lambda_t,
        )
    return out


def fit_all_channels(
    per_channel_per_track: dict[str, dict[str, list[float] | tuple[float, ...]]],
    *,
    min_tracks: int = 2,
    min_samples_per_track: int = 1,
) -> dict[tuple[str, str], TrackIntercept]:
    """Fit one random-intercept model per output channel.

    Returns a flat ``(channel, track) -> TrackIntercept`` map suitable
    for the ``PhysicsModel.track_random_intercepts`` slot.
    """
    out: dict[tuple[str, str], TrackIntercept] = {}
    for channel, per_track in per_channel_per_track.items():
        intercepts = fit_per_channel(
            per_track,
            channel_name=channel,
            min_tracks=min_tracks,
            min_samples_per_track=min_samples_per_track,
        )
        for track, intercept in intercepts.items():
            out[(channel, track)] = intercept
    return out


def predict_correction(
    intercepts: dict[tuple[str, str], TrackIntercept] | None,
    channel: str,
    track: str | None,
) -> tuple[float, float]:
    """Return ``(intercept, intercept_std)`` for ``(channel, track)``.

    Returns ``(0.0, 0.0)`` when:
      * ``intercepts`` is ``None`` or empty (legacy pickle / model
        without random-intercept fit).
      * ``track`` is ``None`` (caller didn't thread the target track --
        recommend path may not always have it).
      * ``(channel, track)`` isn't in the dict (track wasn't represented
        in training -- e.g. a cross-car-borrowed schedule).

    Safe to call in every predict path; the (0, 0) fallback preserves
    the legacy behaviour exactly.
    """
    if not intercepts or track is None:
        return (0.0, 0.0)
    hit = intercepts.get((channel, track))
    if hit is None:
        return (0.0, 0.0)
    std = float(hit.intercept_std) * _INTERCEPT_STD_OOF_INFLATION
    return (float(hit.intercept), std)


__all__ = [
    "TrackIntercept",
    "fit_all_channels",
    "fit_per_channel",
    "predict_correction",
]
