"""Day 3 acceptance gate -- hierarchical Bayesian retrofit (Mode 1).

PLAN.md Section 14.3 acceptance gate (held-out form, runs Day 5):
> For BMW with H1 (Spa) held out, the Bayesian retrofit's posterior
> mean prediction on H1 must beat the current v4 surrogate prediction
> on H1 by >= 5% in MAE on setup-readout target columns.

That gate requires the wire-in to `fit_per_car` (Day 4) and a fresh
per-car retrain with H1 in the held-out set (Day 5). For Day 3, the
gate is the Mode 1 canonical-case proof + the broken-model canary:

- Inject the Mode 1 scenario verbatim (Hockenheim 24*17 vs Spa 6*14.5).
- Confirm the hierarchical posterior keeps Spa near 14.5 (NOT dragged
  to 16.5).
- Confirm a pooled-regression baseline (the broken model) DOES drag
  Spa to 16.5; verify the hierarchical improvement is >50% reduction
  in Spa's MAE vs pooled.

Run: `uv run python scripts/day_03_gate.py`
"""
from __future__ import annotations

import sys

from racingoptimizer.physics.bayes_retrofit import fit_per_parameter


def main() -> int:
    failures: list[str] = []

    # Mode 1 canonical case: BMW Ferrari wing values per CLAUDE.md
    # lines 112-118. Hockenheim n=24 at 17.0; Spa n=6 mean 14.5.
    data = {
        "hockenheim_gp": [17.0] * 24,
        "spa_2024_up": [14.0] * 3 + [15.0] * 3,
    }
    out = fit_per_parameter(data, parameter_name="rear_wing_angle_deg")

    if "spa_2024_up" not in out:
        failures.append("hierarchical fit dropped Spa entirely")
    else:
        spa = out["spa_2024_up"]
        spa_empirical = 14.5
        # Pooled-regression baseline (the broken model the canary uses).
        pooled = (24 * 17.0 + 6 * 14.5) / 30  # ~16.5
        spa_hierarchical_err = abs(spa.mean - spa_empirical)
        spa_pooled_err = abs(pooled - spa_empirical)
        improvement = (spa_pooled_err - spa_hierarchical_err) / spa_pooled_err

        print(f"  Spa empirical mean: {spa_empirical:.3f}")
        print(f"  Spa hierarchical posterior: {spa.mean:.3f}  (err={spa_hierarchical_err:.3f})")
        print(f"  Spa pooled-regression baseline: {pooled:.3f}  (err={spa_pooled_err:.3f})")
        print(f"  Improvement vs pooled: {improvement * 100:.1f}%")
        print(f"  Spa posterior std: {spa.std:.3f}  (n={spa.n_samples})")
        print(f"  Spa shrinkage: {spa.shrinkage:.3f}")

        if spa_hierarchical_err >= spa_pooled_err:
            failures.append(
                f"hierarchical no better than pooled: hierarchical_err="
                f"{spa_hierarchical_err:.3f}, pooled_err={spa_pooled_err:.3f}"
            )
        if improvement < 0.50:
            failures.append(
                f"improvement vs pooled only {improvement * 100:.1f}%, "
                f"expected >= 50%"
            )
        if abs(spa.mean - spa_empirical) > 1.0:
            failures.append(
                f"Spa posterior {spa.mean:.3f} drifted >1.0 from empirical "
                f"{spa_empirical:.3f}; this is the Mode 1 failure case"
            )
        if not (0.0 <= spa.shrinkage <= 1.0):
            failures.append(f"Spa shrinkage {spa.shrinkage} outside [0, 1]")

    # Hockenheim sanity: high-n high-uniform track should have
    # near-empirical mean and minimal posterior std.
    if "hockenheim_gp" in out:
        hock = out["hockenheim_gp"]
        if abs(hock.mean - 17.0) > 0.5:
            failures.append(
                f"Hockenheim posterior {hock.mean:.3f} drifted >0.5 from "
                f"empirical 17.0 (this is the majority track; should NOT shrink much)"
            )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "\nGATE PASSED: hierarchical Bayesian retrofit closes Mode 1 on "
        "the canonical synthetic case (Hockenheim 24*17 vs Spa 6*14.5)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
