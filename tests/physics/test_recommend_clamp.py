"""Pre/post clamp invariants for score_setup + recommend (spec §10).

The contract pinned in code:
- `score_setup(..., strict=False)` SILENTLY clamps out-of-bounds setups
  (the typical entry path — telemetry-derived baselines occasionally drift
  below the legal floor). `strict=True` raises ValueError instead.
- `recommend` pre-clamps the optimizer's variable bounds and post-clamps
  the optimizer's output as a defensive check.
"""
from __future__ import annotations

import pytest

from racingoptimizer.constraints import load_constraints
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics.recommend import recommend


@pytest.fixture
def bmw_model_track(bmw_model_session):
    model, track, _root = bmw_model_session
    return model, track


def test_score_setup_strict_rejects_out_of_bounds(bmw_model_track) -> None:
    """`score_setup(..., strict=True)` raises on drift; the lenient default
    silently clamps and returns a number."""
    from racingoptimizer.physics.score import score_setup as _score_setup
    model, track = bmw_model_track
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    constraints = model.constraints
    if constraints is None:
        pytest.skip("model carries no constraints table")
    bounded_param = next(
        (
            p for p in model.baseline_setup
            if constraints.bounds(model.car, p) is not None
        ),
        None,
    )
    if bounded_param is None:
        pytest.skip("no bounded parameter in baseline setup")
    bound = constraints.bounds(model.car, bounded_param)
    assert bound is not None
    bad_setup = dict(model.baseline_setup)
    bad_setup[bounded_param] = bound[1] * 100.0  # absurdly out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        _score_setup(model, bad_setup, track, env, strict=True)
    # The lenient default just clamps and returns a value.
    score = model.score_setup(bad_setup, track, env)
    assert isinstance(score, float)


def test_recommend_post_clamp_holds(bmw_model_track) -> None:
    """Post-clamp drift would raise ValueError. A clean run completes silently."""
    model, track = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)
    # Every recommended parameter must lie within its constraint bound.
    for name, (value, _conf) in rec.parameters.items():
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        lo, hi = bound
        assert lo <= value <= hi


def test_recommend_warns_when_observed_median_outside_bound(
    bmw_model_track,
) -> None:
    """If `model.baseline_setup` (observed median) sits outside the legal
    constraint range for a parameter, `recommend` must emit a clamp_warning
    so the briefing surfaces the constraint as suspect — this is the
    regression for the Cadillac tyre-pressure 165 vs 152 kPa bug.
    """
    model, track = bmw_model_track
    constraints = load_constraints()
    # Pick any bounded parameter and shove its baseline OUTSIDE the legal
    # range on a copy of the model. We mutate `model.baseline_setup` after
    # constructing the rec context — the dict is stored by reference on
    # the frozen dataclass.
    chosen: str | None = None
    chosen_bound: tuple[float, float] | None = None
    for name in model.baseline_setup:
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        chosen = name
        chosen_bound = bound
        break
    if chosen is None or chosen_bound is None:
        pytest.skip("no bounded parameter on this model")
    # Drop the observed median 5% below the constraint floor.
    span = chosen_bound[1] - chosen_bound[0]
    out_of_bound = chosen_bound[0] - max(span * 0.05, 1e-3)
    model.baseline_setup[chosen] = out_of_bound  # frozen dataclass; dict mutable

    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)
    assert chosen in rec.clamp_warnings, (
        f"expected clamp_warning for {chosen!r} when baseline "
        f"{out_of_bound} sat outside bound {chosen_bound}; got "
        f"warnings={rec.clamp_warnings!r}"
    )
    msg = rec.clamp_warnings[chosen]
    assert "verify" in msg.lower() or "constraints.md" in msg.lower(), (
        f"clamp_warning should point at constraints.md; got: {msg!r}"
    )
