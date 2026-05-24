"""Held-out hybrid vs surrogate-only A/B regression test (H1-H5).

Background
----------
The 2026-05-23 audit (`docs/audit_2026-05-23/00_findings_and_fix_plan.md`)
identified P1-2 / 3.3: the hybrid scoring path (physics evaluator +
surrogate blend with guardrail penalties) shipped into the production
DE objective on 2026-05-23 without a non-regression gate against the
legacy `--surrogate-only` path on data the model has never seen.

This test closes that gap. For each of the five held-out IBTs
(H1 BMW Spa, H2 Cadillac Laguna, H3 Ferrari Hockenheim, H4 Acura
Daytona, H5 Porsche Algarve) it:

1. Fits a production-only per-car v4 model (held-out session excluded
   via the catalog's `held_out=1` flag).
2. Loads the held-out IBT's *observed* garage setup from the catalog
   row's setup blob.
3. Scores that observed setup under both `hybrid=True` (production
   default) and `hybrid=False` (legacy `--surrogate-only` path) via
   `racingoptimizer.physics.score.score_breakdown`.
4. Asserts non-regression invariants: identical key sets, finite
   totals, and a bounded relative delta between the two paths.

Why score the *observed* setup rather than re-run DE in both modes
---------------------------------------------------------------
The audit's metric menu was "score delta, readout MAE, guardrail
count" — all three are computable from `score_breakdown(...)` on a
fixed setup, no DE needed. DE has a hardcoded budget
(`maxiter=15, popsize=20`) so 5 cars x 2 paths x ~15 min = 2.5 h is
too slow even for `@pytest.mark.slow`. Scoring the observed
driver-validated setup is the strictly stronger non-regression check:
if hybrid evaluates a real driven setup drastically worse than
surrogate-only, the blend has a systematic bias.

Marked ``slow`` — fitting per-car v4 models on full production
corpora is the multi-minute path. The model pickle cache at
`corpus/models/<car>__per-car__<digest>.pickle` makes repeat runs
fast once the first fit completes.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import pytest

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner.states import corner_phase_states
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import laps as ingest_laps
from racingoptimizer.ingest.paths import catalog_path, default_corpus_root
from racingoptimizer.physics.corner_schedule import build_corner_schedule
from racingoptimizer.physics.fitter import fit_per_car
from racingoptimizer.physics.ontology import setup_value
from racingoptimizer.physics.score import score_breakdown

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]

# H1-H5 held-out session IDs (docs/physics-rebuild/holdout.sha256). Each
# is flagged `held_out=1` in the production catalog and excluded from
# `query_sessions(... include_held_out=False)`.
HELDOUT_CASES: list[tuple[str, str, str, str]] = [
    ("H1", "bmw",      "3f0a05d3f44527bd", "spa_2024_up"),
    ("H2", "cadillac", "d236a089300fc0ea", "lagunaseca"),
    ("H3", "ferrari",  "fc96805e3b1a27cc", "hockenheim_gp"),
    ("H4", "acura",    "72f43fa4527c4260", "daytona_2011_road"),
    ("H5", "porsche",  "a3d43056a952ff99", "algarve_gp"),
]
HELDOUT_IDS = [c[0] for c in HELDOUT_CASES]


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    root = default_corpus_root()
    if not catalog_path(root).is_file():
        pytest.skip(f"no corpus/catalog.sqlite at {root}; ingest first")
    return root


def _safe_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _env_from_rows(rows: list[dict]) -> EnvironmentFrame:
    """Median env over the held-out IBT's corner-phase rows."""
    def med(*keys: str, default: float) -> float:
        vals: list[float] = []
        for r in rows:
            for k in keys:
                v = r.get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    vals.append(fv)
                    break
        if not vals:
            return default
        return statistics.median(vals)

    return EnvironmentFrame(
        air_temp_c=med("air_temp_c_mean", default=25.0),
        air_density=med("air_density_kg_m3_mean", "air_density_mean", default=1.18),
        air_pressure_mbar=med(
            "air_pressure_mbar_mean", "air_pressure_pa_mean", default=1013.25,
        ),
        relative_humidity=med("relative_humidity_mean", default=0.5),
        wind_vel_ms=med("wind_vel_ms_mean", default=0.0),
        wind_dir_deg=med("wind_dir_deg_mean", "wind_dir_rad_mean", default=0.0),
        fog_level=med("fog_level_mean", default=0.0),
        track_temp_c=med("track_temp_crew_c_mean", "track_temp_c_mean", default=30.0),
        track_wetness=med("track_wetness_mean", default=0.0),
        weather_declared_wet=bool(med("weather_declared_wet_mean", default=0.0) > 0.5),
        precip_type=int(med("precip_type_mean", default=-1.0)),
        skies=int(med("skies_mean", default=-1.0)),
    )


def _observed_setup(car: str, held_setup_blob: str | None, model) -> dict[str, float]:
    """Decode the held-out catalog setup blob into the bounded-setup vector."""
    import json
    out = dict(model.baseline_setup)
    if not held_setup_blob:
        return out
    try:
        blob = json.loads(held_setup_blob)
    except (json.JSONDecodeError, TypeError):
        return out
    for name in model.baseline_setup:
        v = setup_value(car, name, blob)
        if v is None:
            continue
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            continue
    return out


@pytest.mark.parametrize(
    ("label", "car", "session_id", "expected_track"),
    HELDOUT_CASES,
    ids=HELDOUT_IDS,
)
def test_hybrid_vs_surrogate_only_at_observed_setup(
    label: str,
    car: str,
    session_id: str,
    expected_track: str,
    corpus_root: Path,
) -> None:
    """Score H{label}'s observed setup under hybrid and surrogate-only.

    Non-regression invariants (regression GUARDS, not calibration
    targets — deliberately loose so legitimate weight tweaks don't
    churn while catastrophic blend regressions still fail):

    * Identical corner-phase key sets under both modes.
    * Both modes produce a non-empty, finite total score.
    * Total relative delta |hybrid - surrogate| / |surrogate| <= 0.50
      — hybrid blend shouldn't shift the score by more than 50% on a
      real driver-validated setup.
    """
    with cat.open_catalog(catalog_path(corpus_root)) as conn:
        held = cat.get_session(conn, session_id)
        if held is None:
            pytest.skip(f"{label}: session {session_id} not in catalog")
        if held.held_out != 1:
            pytest.skip(f"{label}: session not flagged held_out=1")
        prod_sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )

    if not prod_sessions:
        pytest.skip(f"{label}: no production sessions for {car}")

    session_ids = sorted(s.session_id for s in prod_sessions)
    k_folds = 5 if len(session_ids) >= 3 else 2
    model = fit_per_car(
        car=car, session_ids=session_ids, corpus_root=corpus_root, k_folds=k_folds,
    )

    # Held-out corner-phase rows -- used for both the env and as schedule input.
    try:
        laps_df = ingest_laps(
            session_id=session_id, valid_only=True, corpus_root=corpus_root,
        )
    except Exception as exc:
        pytest.skip(f"{label}: cannot load laps for held-out: {exc}")
    if laps_df.height == 0:
        pytest.skip(f"{label}: no valid laps in held-out IBT")

    all_rows: list[dict] = []
    for lap_idx in laps_df["lap_index"].to_list():
        try:
            cps = corner_phase_states(session_id, int(lap_idx), corpus_root=corpus_root)
        except Exception:
            continue
        if cps.height == 0:
            continue
        all_rows.extend(cps.to_dicts())
    if not all_rows:
        pytest.skip(f"{label}: no corner-phase rows in held-out laps")

    env = _env_from_rows(all_rows)
    schedule = build_corner_schedule([session_id], corpus_root=corpus_root)
    if not schedule:
        pytest.skip(f"{label}: empty schedule from held-out laps")

    setup = _observed_setup(car, held.setup, model)

    track = held.track or expected_track

    scores_hybrid = score_breakdown(
        model, setup, track, env, schedule=schedule, hybrid=True,
    )
    scores_surr = score_breakdown(
        model, setup, track, env, schedule=schedule, hybrid=False,
    )

    assert scores_hybrid, f"{label}: hybrid scored zero corner-phases"
    assert scores_surr, f"{label}: surrogate-only scored zero corner-phases"

    # Same set of (corner, phase) keys -- hybrid mustn't silently drop or
    # add corners relative to surrogate-only.
    assert set(scores_hybrid.keys()) == set(scores_surr.keys()), (
        f"{label}: key set diverged hybrid={len(scores_hybrid)} vs "
        f"surrogate={len(scores_surr)}"
    )

    total_h = sum(scores_hybrid.values())
    total_s = sum(scores_surr.values())
    assert math.isfinite(total_h) and total_h > 0.0, (
        f"{label}: hybrid total={total_h!r}"
    )
    assert math.isfinite(total_s) and total_s > 0.0, (
        f"{label}: surrogate total={total_s!r}"
    )

    # Catastrophic-blend guard: hybrid shouldn't shift the total by more
    # than 50% on a real driver-validated setup. Wide tolerance by design;
    # this catches systematic bias, not fine drift.
    rel_delta = abs(total_h - total_s) / max(abs(total_s), 1e-9)
    assert rel_delta <= 0.50, (
        f"{label}: |hybrid - surrogate| / surrogate = {rel_delta:.3f} "
        f"(hybrid={total_h:.4f}, surrogate={total_s:.4f}) -- blend may have "
        "regressed; investigate evaluator weights, guardrail penalties, or "
        "phase weights."
    )
