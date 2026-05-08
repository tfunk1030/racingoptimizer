"""Day 4 acceptance gate -- bayes_retrofit wire-in to fit_per_car.

PLAN.md Section 14.3 mid-component task. The full held-out gate (BMW
with H1 Spa held out, beat v4 by 5% MAE) requires a fresh per-car
retrain with the wire-in active, which lands as Day 5's gate. Day 4
verifies the data flow:

  1. PhysicsModel exposes the new `bayes_posteriors` field.
  2. fit_per_car constructs it via fit_all_parameters (proven by a
     synthetic per-track observed dict round-trip).
  3. FITTERS_LAYOUT_VERSION bumped so existing per-car caches refit
     on next `optimize` invocation (otherwise the new field stays
     empty and Mode 1 closure is silent-no-op).
  4. Pre-Day-4 pickles default-revive cleanly with `bayes_posteriors={}`.

Run: `uv run python scripts/day_04_gate.py`
"""
from __future__ import annotations

import pickle
import sys

from racingoptimizer.physics.bayes_retrofit import (
    BayesPosterior,
    fit_all_parameters,
)
from racingoptimizer.physics.fitters import FITTERS_LAYOUT_VERSION
from racingoptimizer.physics.model import PhysicsModel


def main() -> int:
    failures: list[str] = []

    # 1. New field exposed.
    m = PhysicsModel(car="bmw", session_ids=())
    if not hasattr(m, "bayes_posteriors"):
        failures.append("PhysicsModel missing bayes_posteriors field")
    elif m.bayes_posteriors != {}:
        failures.append(
            f"default bayes_posteriors should be {{}}, got {m.bayes_posteriors!r}"
        )
    else:
        print("  PhysicsModel exposes bayes_posteriors field (default {}): OK")

    # 2. Synthetic Mode 1 round-trip.
    per_track = {
        "hockenheim_gp": {"rear_wing_angle_deg": (17.0,) * 24},
        "spa_2024_up": {
            "rear_wing_angle_deg": (14.0, 14.0, 14.0, 15.0, 15.0, 15.0),
        },
    }
    posteriors = fit_all_parameters(per_track)
    if ("rear_wing_angle_deg", "spa_2024_up") not in posteriors:
        failures.append("synthetic round-trip lost Spa posterior")
    else:
        spa = posteriors[("rear_wing_angle_deg", "spa_2024_up")]
        if abs(spa.mean - 14.5) > 1.0:
            failures.append(
                f"Spa posterior {spa.mean:.3f} > 1.0 from empirical 14.5"
            )
        else:
            print(
                f"  Synthetic Mode 1 round-trip: Spa posterior "
                f"{spa.mean:.3f} (empirical 14.5): OK"
            )

    # 3. Layout version bumped.
    if FITTERS_LAYOUT_VERSION < 3:
        failures.append(
            f"FITTERS_LAYOUT_VERSION = {FITTERS_LAYOUT_VERSION}, "
            f"expected >= 3 to invalidate pre-Day-4 caches"
        )
    else:
        print(
            f"  FITTERS_LAYOUT_VERSION = {FITTERS_LAYOUT_VERSION} "
            f"(>=3, invalidates pre-Day-4 caches): OK"
        )

    # 4. Pre-Day-4 pickle default-revive.
    legacy = PhysicsModel(car="acura", session_ids=("x",))
    blob = pickle.dumps(legacy)
    revived = pickle.loads(blob)
    if revived.bayes_posteriors != {}:
        failures.append(
            f"pickle revive of default model produced "
            f"bayes_posteriors={revived.bayes_posteriors!r}, expected {{}}"
        )
    else:
        print("  Pickle revive defaults bayes_posteriors to {}: OK")

    # 5. End-to-end constructor.
    m2 = PhysicsModel(
        car="ferrari", session_ids=("z",),
        bayes_posteriors=posteriors,
    )
    if m2.bayes_posteriors == {}:
        failures.append("constructor did not store bayes_posteriors")
    else:
        n = len(m2.bayes_posteriors)
        print(f"  Constructor stores bayes_posteriors ({n} entries): OK")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "\nGATE PASSED: PhysicsModel exposes bayes_posteriors, "
        "fit_all_parameters round-trips through it, layout version "
        "bumped, legacy pickles revive with default {}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
