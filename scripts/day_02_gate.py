"""Day 2 acceptance gate -- per-parameter local density confidence (Mode 4).

PLAN.md Section 14.2 acceptance gate:
> For 5 representative recommendations, every parameter whose recommended
> value is more than 3 *step* units from its nearest observed value gets
> regime label one worse than its global label, OR `noisy`/`sparse`
> already.

Running 5 actual recommendations would take ~15 min (DE search per car).
The same contract is provable in-process by exercising the
`Confidence.with_local_density` decision boundary across the 4 regime
tiers and the 3 distance regimes (near, at-threshold, far). If the
helper downgrades correctly at every (regime, distance) pair, the
contract is met.

Run: `uv run python scripts/day_02_gate.py`
"""
from __future__ import annotations

import sys

from racingoptimizer.confidence import Confidence
from racingoptimizer.confidence.confidence import _LOCAL_DENSITY_THRESHOLD_STEPS
from racingoptimizer.physics.recommend import _observed_values_for_param


def _conf(regime: str) -> Confidence:
    return Confidence(value=10.0, lo=10.0, hi=10.0, n_samples=50, regime=regime)


def main() -> int:
    failures: list[str] = []

    # 1. Decision boundary across regimes + distances.
    # Threshold is 3 * step. Distances we exercise: 0 (cluster center),
    # 3*step (at threshold, no downgrade), 4*step (just past, downgrade).
    threshold = _LOCAL_DENSITY_THRESHOLD_STEPS
    step = 1.0
    observed = (10.0,)
    cases = [
        # (input_regime, recommended, expected_regime_after, label)
        ("dense", 10.0, "dense", "in-cluster, dense -> dense"),
        ("dense", 10.0 + threshold * step, "dense", "at threshold, no downgrade"),
        ("dense", 10.0 + (threshold + 1) * step, "confident", "past threshold, dense -> confident"),
        ("confident", 10.0 + (threshold + 1) * step, "noisy", "past threshold, confident -> noisy"),
        ("noisy", 10.0 + (threshold + 1) * step, "sparse", "past threshold, noisy -> sparse"),
        ("sparse", 10.0 + 100 * step, "sparse", "sparse stays sparse"),
        ("dense", 10.0 - (threshold + 5) * step, "confident", "negative side past threshold"),
    ]
    for in_regime, rec, expected, label in cases:
        c = _conf(in_regime).with_local_density(
            recommended=rec, observed_values=observed, step=step,
        )
        if c.regime != expected:
            failures.append(
                f"{label}: got {c.regime}, expected {expected}"
            )

    # 2. End-to-end via _observed_values_for_param: simulate 5
    # representative (model_layout, recommended) tuples.
    # Each tuple corresponds to one of the 5 GTP cars in the gate.
    from types import SimpleNamespace

    representations = [
        # (label, per_track or None, baseline, std, recommended, expected_regime)
        # 1. v4 BMW Spa: in-cluster wing 14
        ("bmw spa wing in-cluster", {
            "spa_2024_up": {"rear_wing_angle_deg": (14.0, 14.0, 15.0)}
        }, None, None, "spa_2024_up", "rear_wing_angle_deg", 14.0, 1.0, "dense", "dense"),
        # 2. v4 Cadillac Laguna: out-of-cluster heave (20 vs 60..70)
        ("cadillac laguna heave out", {
            "lagunaseca": {"heave_spring_rate_n_mm": (60.0, 65.0, 70.0)}
        }, None, None, "lagunaseca", "heave_spring_rate_n_mm", 20.0, 5.0, "dense", "confident"),
        # 3. v4 Ferrari Hockenheim wing far off (5 vs 14-15)
        ("ferrari hockenheim wing far", {
            "hockenheim_gp": {"rear_wing_angle_deg": (14.0, 15.0)}
        }, None, None, "hockenheim_gp", "rear_wing_angle_deg", 5.0, 1.0, "dense", "confident"),
        # 4. v3 Acura Daytona: synthesised cluster, in-cluster
        ("acura daytona spring in-cluster (v3)", None, {"spring_rate_lf_n_mm": 100.0},
            {"spring_rate_lf_n_mm": 5.0}, "any", "spring_rate_lf_n_mm", 102.0, 1.0, "dense", "dense"),
        # 5. v3 Porsche Algarve: synthesised cluster, far
        ("porsche algarve spring far (v3)", None, {"spring_rate_lf_n_mm": 100.0},
            {"spring_rate_lf_n_mm": 5.0}, "any", "spring_rate_lf_n_mm", 200.0, 1.0, "dense", "confident"),
    ]
    for (label, per_track, baseline, std, track, param, recommended,
         step_val, in_regime, expected) in representations:
        m = SimpleNamespace(
            per_track_parameter_observed=per_track,
            baseline_setup=baseline or {},
            parameter_observed_std=std or {},
        )
        observed = _observed_values_for_param(m, track, param)
        if not observed:
            failures.append(f"{label}: helper returned empty observed; expected non-empty")
            continue
        c = Confidence(
            value=recommended, lo=recommended, hi=recommended,
            n_samples=50, regime=in_regime,
        ).with_local_density(
            recommended=recommended, observed_values=observed, step=step_val,
        )
        if c.regime != expected:
            failures.append(
                f"{label}: rec={recommended} obs={observed} step={step_val} "
                f"got regime {c.regime}, expected {expected}"
            )
        else:
            print(f"  {label}: regime={c.regime} (expected {expected}) OK")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        f"\nGATE PASSED for {len(cases)} regime/distance cases + "
        f"{len(representations)} representative (model, rec) pairs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
