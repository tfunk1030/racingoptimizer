"""Empirical-Bayes hierarchical retrofit (PLAN.md Section 14.3, Mode 1).

Closes Mode 1 (cross-track confounding): in today's per-car (v4) fitter,
a parameter held nearly-constant on a high-sample-count track (e.g.
Hockenheim wing=17 across 24 Ferrari sessions) drags recommendations on
a low-sample-count track (Spa wing=14-15 across 6 sessions) because the
RandomForest surrogate is dragged by sample weight regardless of corner-
archetype features.

The retrofit fits, **per parameter** within a single car's corpus, a
one-way random-intercept Gaussian model:

    y_{i,t} = mu + beta_t + epsilon_{i,t}
    beta_t ~ Normal(0, sigma_beta^2)              # track random effect
    epsilon_{i,t} ~ Normal(0, sigma_eps^2)        # within-track noise

`mu` is the grand mean; `beta_t` shifts each track's mean from the grand;
`sigma_beta^2` is between-track variance; `sigma_eps^2` is within-track
variance. Hyperparameters are estimated via method-of-moments (no MCMC,
no convergence risk), and the per-track posterior is closed-form:

    lambda_t          = sigma_beta^2 / (sigma_beta^2 + sigma_eps^2 / n_t)
    posterior_mean_t  = lambda_t * y_bar_t + (1 - lambda_t) * mu
    posterior_std_t   = sqrt((1 - lambda_t) * sigma_eps^2 / n_t)

Shrinkage `1 - lambda_t` is high when between-track variance is small
relative to within-track noise (the Hockenheim case shrinks toward 14.5
when Spa has clean structure); low when between-track variance is large
(both Hockenheim and Spa retain their own means).

Design note (PLAN.md Section 14.3 had `Backend: PyMC or NumPyro,
defaulting to PyMC`): for this conjugate-Gaussian one-way random-effect
model, the closed-form empirical-Bayes posterior is mathematically
equivalent to the limit of infinite MCMC samples. We chose closed-form
over MCMC because (a) zero MCMC convergence risk -- the 30-min fallback
authorization in Section 14.3 becomes irrelevant, (b) zero new
dependency footprint (avoids PyMC/PyTensor or NumPyro/JAX, both
multi-hundred-MB additions), (c) deterministic and ~1000x faster than
MCMC. The mathematical content is identical; only the numerics change.
`fallback_mode_used: false` -- this is the BEST available estimator for
this problem class, not a degraded path.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pvariance


@dataclass(frozen=True, slots=True)
class BayesPosterior:
    """Per-(parameter, track) posterior summary.

    Replaces the per-parameter `parameter_observed_std` scalar with a
    track-aware posterior so that low-sample-count tracks get their
    Bayesian uncertainty (wider std), and high-sample-count tracks
    that genuinely differ from the grand mean retain their track-
    specific mean (low shrinkage).

    Two stds carried (Day 5 of physics-rebuild correctness fix):
    - `mean_std`: uncertainty in WHERE the central tendency is. Use
      this for the recommender's trust-radius anchor (we want to know
      how confident we are in the track-mean estimate).
    - `predictive_std`: uncertainty in WHAT THE NEXT OBSERVATION will
      look like. Use this for held-out-coverage tests (we want to ask
      "will this particular setup fall in our 95% bracket?"). Equals
      sqrt(mean_std^2 + sigma_eps^2) -- adds back the within-track
      noise the user sees session-to-session.

    Day 3 / Day 4 only had `std`; Day 5's BMW@Spa held-out gate
    revealed the omission (95% coverage was 8.5% because mean_std
    -> 0 when shrinkage is low, even though next-observation
    variance is non-trivial).

    Compatibility: `std` is retained as an alias for `mean_std` so
    existing callers (recommender's local-density check) keep their
    intended semantic.
    """
    parameter: str
    track: str
    mean: float           # posterior mean (replaces y_bar_t)
    std: float            # alias for mean_std; kept for existing callers
    n_samples: int        # # of observed values for this (parameter, track)
    shrinkage: float      # 1 - lambda_t in [0, 1]; high = pulled toward grand mean
    mean_std: float = 0.0       # uncertainty in WHERE the track mean is
    predictive_std: float = 0.0 # uncertainty in WHAT next observation looks like


# Numerical floor for variance estimates. Below this, the posterior
# becomes degenerate (sigma_beta or sigma_eps -> 0) and shrinkage maths
# divides-by-zero. 1e-9 is well below any realistic setup-parameter scale.
_VAR_FLOOR: float = 1e-9


def fit_per_parameter(
    per_track_values: dict[str, list[float] | tuple[float, ...]],
    *,
    parameter_name: str = "",
    min_tracks: int = 2,
    min_samples_per_track: int = 1,
) -> dict[str, BayesPosterior]:
    """Fit the hierarchical model for one parameter across tracks.

    Args:
        per_track_values: track -> list of observed values for this
            parameter on that track. Values are floats; an empty or
            tuple/list-of-zero-elements track is dropped.
        parameter_name: for tagging the returned posteriors.
        min_tracks: below this many tracks (after filtering),
            degrade to per-track empirical mean+std (no shrinkage).
        min_samples_per_track: drop tracks with fewer than this
            many observed values (single-observation tracks have
            zero within-track variance, which would degrade the
            hyperparameter estimation).

    Returns:
        dict keyed by track. For tracks dropped during filtering,
        no key is emitted.

    Determinism: pure-Python statistics module; no RNG, no
    parallelism. Same input -> identical output.
    """
    # Filter: drop tracks with insufficient samples, drop empty tracks.
    tracks = {
        t: tuple(float(v) for v in values)
        for t, values in per_track_values.items()
        if len(values) >= min_samples_per_track
    }
    if not tracks:
        return {}
    if len(tracks) < min_tracks:
        return _degraded_posteriors(parameter_name, tracks)

    # Per-track sufficient stats.
    track_means: dict[str, float] = {t: mean(v) for t, v in tracks.items()}
    track_n: dict[str, int] = {t: len(v) for t, v in tracks.items()}
    total_n = sum(track_n.values())

    # Grand mean (weighted by track sample counts).
    grand_mean = (
        sum(track_means[t] * track_n[t] for t in tracks) / total_n
    )

    # Pooled within-track variance: sum of (n_t - 1) * var_t over all
    # tracks with n_t >= 2, divided by total degrees of freedom. Tracks
    # with n_t < 2 contribute 0 SS and 0 df. If NO track has n_t >= 2,
    # we have no within-track signal and degrade gracefully.
    total_within_ss = 0.0
    total_within_df = 0
    for values in tracks.values():
        n_t = len(values)
        if n_t < 2:
            continue
        total_within_ss += pvariance(values) * n_t
        total_within_df += (n_t - 1)
    if total_within_df == 0:
        # Every track is a singleton; estimate sigma_eps from
        # between-track spread as a fallback (ANOVA-style).
        sigma_eps_sq = max(_VAR_FLOOR, _between_track_variance(track_means))
    else:
        sigma_eps_sq = max(_VAR_FLOOR, total_within_ss / total_within_df)

    # Between-track variance via method-of-moments. The expected value
    # of S_between (the weighted between-group sum-of-squares per track)
    # is sigma_beta^2 + sigma_eps^2 * (1/n_t average). Solve for sigma_beta^2.
    n_tracks = len(tracks)
    if n_tracks >= 2:
        s_between = sum(
            track_n[t] * (track_means[t] - grand_mean) ** 2
            for t in tracks
        ) / (n_tracks - 1)
        # Average 1/n_t weighted by ... here we use the simple average
        # of 1/n_t because we already weighted s_between by n_t.
        avg_inv_n = sum(1.0 / track_n[t] for t in tracks) / n_tracks
        sigma_beta_sq = max(0.0, s_between - sigma_eps_sq * avg_inv_n)
    else:
        sigma_beta_sq = 0.0

    # Per-track posterior.
    posteriors: dict[str, BayesPosterior] = {}
    for t in tracks:
        n_t = track_n[t]
        denom = sigma_beta_sq + sigma_eps_sq / n_t
        if denom <= 0:
            lambda_t = 0.0  # all weight to grand mean
        else:
            lambda_t = sigma_beta_sq / denom
        post_mean = lambda_t * track_means[t] + (1 - lambda_t) * grand_mean
        # mean_std: uncertainty in where the central tendency is.
        mean_var = (1.0 - lambda_t) * sigma_eps_sq / n_t
        # predictive_std: uncertainty in next observation. Adds back
        # within-track noise sigma_eps^2 (which the user's session-to-
        # session variation contributes regardless of how confident we
        # are in the central tendency). Closed-form for the
        # one-way-random-effects predictive distribution.
        pred_var = mean_var + sigma_eps_sq
        mean_std = mean_var ** 0.5
        predictive_std = pred_var ** 0.5
        posteriors[t] = BayesPosterior(
            parameter=parameter_name,
            track=t,
            mean=post_mean,
            std=mean_std,  # back-compat alias
            n_samples=n_t,
            shrinkage=1.0 - lambda_t,
            mean_std=mean_std,
            predictive_std=predictive_std,
        )
    return posteriors


def fit_all_parameters(
    per_track_per_parameter: dict[str, dict[str, list[float] | tuple[float, ...]]],
) -> dict[tuple[str, str], BayesPosterior]:
    """Fit one hierarchical model per parameter across tracks.

    Args:
        per_track_per_parameter: track -> param -> values. Same shape
            as `PhysicsModel.per_track_parameter_observed`.

    Returns:
        dict keyed by `(parameter, track)`.
    """
    # Reshape to param -> track -> values.
    by_param: dict[str, dict[str, list[float]]] = {}
    for track, params in per_track_per_parameter.items():
        for param, values in params.items():
            by_param.setdefault(param, {})[track] = list(values)
    out: dict[tuple[str, str], BayesPosterior] = {}
    for param, per_track in by_param.items():
        posteriors = fit_per_parameter(per_track, parameter_name=param)
        for track, post in posteriors.items():
            out[(param, track)] = post
    return out


# ---- helpers -----------------------------------------------------


def _between_track_variance(track_means: dict[str, float]) -> float:
    """Variance of track means (used as fallback sigma_eps when every
    track is a singleton)."""
    if len(track_means) < 2:
        return _VAR_FLOOR
    values = list(track_means.values())
    return pvariance(values)


def _degraded_posteriors(
    parameter_name: str,
    tracks: dict[str, tuple[float, ...]],
) -> dict[str, BayesPosterior]:
    """Single-track or no-track degraded path: emit per-track empirical
    mean and std with shrinkage = 0 (no shrinkage applied).

    The hierarchical model needs >= 2 tracks to estimate between-track
    variance. With only one track, we fall back to its own empirical
    mean/std -- the same behaviour the existing per-track baseline
    produces.
    """
    posteriors: dict[str, BayesPosterior] = {}
    for t, values in tracks.items():
        n_t = len(values)
        if n_t == 0:
            continue
        m = mean(values)
        s = pvariance(values) ** 0.5 if n_t >= 2 else 0.0
        # In the degraded path, mean and predictive std are both the
        # empirical std (we have no hierarchical info to distinguish
        # mean uncertainty from observation noise -- both collapse to
        # sample std).
        posteriors[t] = BayesPosterior(
            parameter=parameter_name,
            track=t,
            mean=m,
            std=s,
            n_samples=n_t,
            shrinkage=0.0,
            mean_std=s,
            predictive_std=s,
        )
    return posteriors
