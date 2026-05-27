# Acura ARX-06 GTP — driver/engineering setup spec

Source: driver/engineering guidance captured 2026-05-25.

This file is the authoritative per-car setup-philosophy reference for
the Acura ARX-06. It is the spec the optimizer's recommendations should
not contradict, and the language the briefing should use when phrasing
Acura moves.

## 1. Aero map sweet spot

| Channel                | Target (mid-corner / at speed) | Notes |
|------------------------|--------------------------------|-------|
| `dynamic_front_rh_at_speed_mm` | ≈ **15 mm**            | Peak downforce on the front |
| `dynamic_rear_rh_at_speed_mm`  | ≈ **40 mm**            | Peak downforce on the rear  |
| Both ends                       | As low as possible (no bottoming) | Lowest drag |

**Rake:** larger rear RH − front RH gap (more rake) moves aero balance
**forward** → more oversteer. Less rake moves balance rearward →
more understeer.

## 2. Heave springs

* **Front heave spring**: soft enough that the **nose sits at ~15 mm
  mid-corner** as much as possible, **without** excessive bottoming or
  an undesirable aero-balance shift under braking / turn-in.
* **Rear heave spring**: a soft rear heave lets the rear drop under
  aero load → less drag, but loses some downforce + efficiency mid-
  corner (rear RH falls below the 40 mm peak-downforce target).
  Compromise required between drag / downforce / dynamic aero balance.

Adjusting heave springs is the right lever for **dynamic** ride height
without changing roll stiffness.

## 3. Balance lever priority

The easiest way to rebalance the setup, in order of preference:

### 3.1 Rear wing angle (primary downforce-trim lever)

* **Lower wing** → more oversteer, less downforce, higher top speed.
* **Higher wing** → more understeer, more downforce, lower top speed.

If a move of more than one click in either direction looks indicated,
**switch downforce-trim setups** (high / medium / low) rather than
ride the wing further.

### 3.2 Rear pushrod length delta (aero balance, drag/DF unchanged)

Changes aero balance without changing the downforce / drag trim.
Particularly noticeable in mid- and high-speed corners.

* **Lower pushrod length delta** → lower rear ride height → less rake
  → **rearward** aero balance → understeer.
* **Higher pushrod length delta** → higher rear ride height → more
  rake → **forward** aero balance → oversteer.

### 3.3 ARBs / ARB blades (mechanical balance, aero unchanged)

Useful when the general balance is right but slow-speed rotation needs
adjusting.

| Adjustment                            | Effect          |
|---------------------------------------|-----------------|
| Bigger front ARB / more blades        | Less rotation   |
| Smaller front ARB / less blades       | More rotation   |
| Bigger rear ARB / more blades         | More rotation   |
| Smaller rear ARB / less blades        | Less rotation   |

### 3.4 Rear differential preload

The simplest diff change. While preload affects on-throttle behaviour,
the **bigger effect is on entry / turn-in**.

* **More preload** → less rotation off throttle, **more stability on entry**.
* **Less preload** → more rotation off throttle, less stability on entry.

The full Acura diff (coast/drive ramps, clutch plates) is a powerful
tool — experiment beyond preload once preload is set.

## 4. How the optimizer should use this

* **Target prior**: the per-car aero-RH targets in §1 are encoded in
  `physics/aero_targets.py::ACURA_AERO_TARGETS` and consumed by any
  scoring or explanation path that wants an honest "this move pushes
  rear RH away from the 40 mm peak-downforce target" statement.
* **Narrative phrasing**: rear-wing / rear-pushrod / ARB / preload
  explanations on Acura should follow the §3 vocabulary (rake,
  downforce trim, on-/off-throttle rotation, entry stability) rather
  than generic GTP language.
* **Setup-justifier subagent**: when validating a recommended Acura
  setup, this file is the rubric for "does the recommendation
  contradict the car's published handling characteristics".

## 5. Open items

* The default per-car evaluator weights for Acura are the generic
  `(0.5, 0.3, 0.2)` fallback because the corpus is below the
  per-car-calibration threshold. Once corpus density passes the
  calibration gate, weight Acura with explicit preference for
  `aero_balance` and `grip_headroom` consistent with §1's emphasis
  on aero-RH targets driving lap time.
* `physics/static_rh_kinematic.py` ships per-car linear fits for the
  static (garage) RH channels. The dynamic-at-speed channels in §1
  are still handled by the surrogate — the holdout gate's reading on
  `dynamic_*_rh_at_speed_mm` is the right indicator for whether the
  optimizer can actually steer Acura to the §1 targets.
