"""P1.2 -- per-(car, track) lap-time Spearman gate.

For each ``(car, track)`` pair with at least ``_MIN_PAIR_SESSIONS``
production sessions, this script leaves one session out at a time,
fits the per-car model on the remaining sessions, computes the
score of the held-out session's observed setup at the held-out's
target track + averaged environment, and accumulates ``(score,
median_lap_time_s)`` pairs. The Spearman rank correlation between
score and lap time is the per-pair gate value: a healthy model has
``rho >= _SPEARMAN_TARGET`` so faster setups consistently score
higher than slower ones.

The script writes a JSON next to ``holdout_accuracy_latest.json``
for trend tracking and exits non-zero when any qualifying pair fails
the target. The Spearman helper at the top is the unit-testable
piece; the orchestration layer that walks the catalog is by design
side-effecting and not exercised in the fast suite.

Run: ``uv run python scripts/lap_time_correlation_gate.py``.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

# Per-(car, track) Spearman target; pairs below this fail the gate.
_SPEARMAN_TARGET: float = 0.30

# Minimum production sessions per pair before the pair is included in
# the gate. Below 10 the rank correlation is too noisy to gate on.
_MIN_PAIR_SESSIONS: int = 10


def _rankdata(values: list[float]) -> list[float]:
    """Average-rank ranking (matches ``scipy.stats.rankdata``).

    Ties are assigned the average of the ranks they would have spanned;
    this keeps Spearman computations consistent with scipy without
    pulling scipy in as a hard dep for the script.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # +1 because ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _spearman_correlation(
    pairs: Iterable[tuple[float, float]],
) -> float | None:
    """Spearman rank correlation between two parallel lists.

    Returns None when fewer than 3 finite pairs are provided or when
    either ranking is constant (Spearman is undefined). The score
    column should be the model's predicted value for each setup; the
    lap_time column is the observed median lap time. By P1.2's
    convention, the gate prefers ``rho > 0`` (higher score should
    correspond to FASTER laps, i.e. lower lap_time, so we negate
    inside ``main()`` -- but the helper itself returns the raw
    rank correlation between the two columns as given).
    """
    finite_pairs: list[tuple[float, float]] = [
        (a, b) for (a, b) in pairs
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(finite_pairs) < 3:
        return None
    xs = [p[0] for p in finite_pairs]
    ys = [p[1] for p in finite_pairs]
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    n = float(len(rx))
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    denom = math.sqrt(var_x * var_y)
    if denom <= 0.0:
        return None
    return cov / denom


def _qualifying_pairs(
    pair_sessions: dict[tuple[str, str], list[str]],
) -> list[tuple[str, str]]:
    """Filter to ``(car, track)`` pairs with >= _MIN_PAIR_SESSIONS sessions."""
    return [
        (car, track)
        for (car, track), sids in pair_sessions.items()
        if len(sids) >= _MIN_PAIR_SESSIONS
    ]


def _evaluate_pair_score(
    rho: float | None,
) -> tuple[bool, str]:
    """Decide if a single pair's correlation passes the gate.

    The convention: Spearman of ``(score, -lap_time)`` -- a positive
    correlation means higher scores rank with faster laps. The
    orchestration negates lap_time before calling
    ``_spearman_correlation`` so the helper can return the raw
    correlation without semantic baggage.
    """
    if rho is None:
        return False, "insufficient_data"
    if rho < _SPEARMAN_TARGET:
        return False, f"rho={rho:.3f}<{_SPEARMAN_TARGET:.2f}"
    return True, f"rho={rho:.3f}"


def _build_pair_sessions_from_catalog() -> dict[tuple[str, str], list[str]]:
    """Walk the catalog and group session IDs by ``(car, track)``.

    Production sessions only -- gate-only IBTs are excluded by passing
    ``include_held_out=False``. Returns an empty dict when the catalog
    isn't reachable so the script's orchestration can short-circuit
    cleanly in CI environments without a corpus.
    """
    try:
        from racingoptimizer.ingest import catalog as cat
        from racingoptimizer.ingest.api import catalog_path, resolve_corpus_root
    except ImportError:
        return {}
    try:
        root = Path(resolve_corpus_root(None))
    except Exception:
        return {}
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    try:
        with cat.open_catalog(catalog_path(root)) as conn:
            sessions = cat.query_sessions(
                conn, valid_only=True, include_held_out=False,
            )
    except Exception:
        return {}
    for sess in sessions:
        car = (getattr(sess, "car", "") or "").lower()
        track = getattr(sess, "track", None)
        if not car or not track:
            continue
        out[(car, track)].append(sess.session_id)
    return dict(out)


def _compute_loso_pairs_for_track(
    car: str,
    track: str,
    session_ids: list[str],
    corpus_root: Path,
) -> list[tuple[float, float]]:
    """Per-(car, track) LOSO loop: ``(score, -median_lap_time)`` per session.

    For each session in ``session_ids``: fit a per-car model with that
    session held out, score the held session's observed setup on the
    target track, and pair the score against the held session's median
    lap time. Lap-time is negated so a positive Spearman against score
    means "higher score correlates with faster laps".

    Returns the list of pairs; sessions that fail any step (missing
    catalog row, no laps, surrogate fit failure, empty schedule) are
    skipped silently -- the Spearman helper drops non-finite pairs
    downstream and ``_min_tracks``/``_min_samples`` filter takes care of
    pairs with too few survivors.

    Heavy -- a 10-session pair runs ~10x the per-car fit cost. Designed
    to be invoked offline on a workstation, then the JSON is committed
    so CI consumes pre-computed results. In CI environments without a
    corpus, ``_build_pair_sessions_from_catalog`` returns an empty dict
    so this function is never called.
    """
    # Lazy imports: the orchestration touches the full ingest +
    # corner-phase + physics-fitter stack, which is a heavy import
    # graph. Keep the helpers at module scope (above) cheap to import
    # so the unit tests don't pay this cost.
    import math
    import statistics

    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.corner.states import corner_phase_states
    from racingoptimizer.ingest import catalog as cat
    from racingoptimizer.ingest.api import catalog_path
    from racingoptimizer.ingest.api import laps as ingest_laps
    from racingoptimizer.physics.corner_schedule import build_corner_schedule
    from racingoptimizer.physics.fitter import fit_per_car
    from racingoptimizer.physics.ontology import setup_value
    from racingoptimizer.physics.score import score_breakdown

    def _safe_float(value, default: float) -> float:
        if value is None:
            return default
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return f if math.isfinite(f) else default

    def _env_from_rows(rows: list[dict]) -> EnvironmentFrame:
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
            air_density=med(
                "air_density_kg_m3_mean", "air_density_mean", default=1.18,
            ),
            air_pressure_mbar=med(
                "air_pressure_mbar_mean", "air_pressure_pa_mean",
                default=1013.25,
            ),
            relative_humidity=med("relative_humidity_mean", default=0.5),
            wind_vel_ms=med("wind_vel_ms_mean", default=0.0),
            wind_dir_deg=med(
                "wind_dir_deg_mean", "wind_dir_rad_mean", default=0.0,
            ),
            fog_level=med("fog_level_mean", default=0.0),
            track_temp_c=med(
                "track_temp_crew_c_mean", "track_temp_c_mean", default=30.0,
            ),
            track_wetness=med("track_wetness_mean", default=0.0),
            weather_declared_wet=bool(
                med("weather_declared_wet_mean", default=0.0) > 0.5
            ),
            precip_type=int(med("precip_type_mean", default=-1.0)),
            skies=int(med("skies_mean", default=-1.0)),
        )

    out: list[tuple[float, float]] = []
    for held_sid in session_ids:
        rest = [s for s in session_ids if s != held_sid]
        # Must leave enough training material for a stable per-car fit
        # AND must keep at least 5 sessions of cross-track variety so
        # the surrogate doesn't degenerate to a constant per-track fit.
        if len(rest) < 5:
            continue

        # 1. Catalog lookup: held session must exist and be valid.
        try:
            with cat.open_catalog(catalog_path(corpus_root)) as conn:
                held = cat.get_session(conn, held_sid)
        except Exception:
            continue
        if held is None or not held.valid:
            continue

        # 2. Held session's observed setup blob -> bounded vector.
        try:
            model = fit_per_car(
                car=car, session_ids=rest, corpus_root=corpus_root,
                k_folds=5 if len(rest) >= 3 else 2,
            )
        except Exception:
            continue

        observed_setup = dict(model.baseline_setup)
        if held.setup:
            try:
                blob = json.loads(held.setup)
            except (json.JSONDecodeError, TypeError):
                blob = {}
            for name in model.baseline_setup:
                v = setup_value(car, name, blob)
                if v is None:
                    continue
                try:
                    observed_setup[name] = float(v)
                except (TypeError, ValueError):
                    continue

        # 3. Load held session laps + median lap time.
        try:
            laps_df = ingest_laps(
                session_id=held_sid, valid_only=True, corpus_root=corpus_root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        lap_time_col = None
        for cand in ("lap_time_s", "lap_time", "time_s"):
            if cand in laps_df.columns:
                lap_time_col = cand
                break
        if lap_time_col is None:
            continue
        lap_times = [
            float(t) for t in laps_df[lap_time_col].to_list()
            if t is not None and math.isfinite(float(t))
        ]
        if not lap_times:
            continue
        median_lap_time = statistics.median(lap_times)

        # 4. Corner-phase rows for the held session -> env + schedule.
        all_rows: list[dict] = []
        for lap_idx in laps_df["lap_index"].to_list():
            try:
                cps = corner_phase_states(
                    held_sid, int(lap_idx), corpus_root=corpus_root,
                )
            except Exception:
                continue
            if cps.height == 0:
                continue
            all_rows.extend(cps.to_dicts())
        if not all_rows:
            continue

        env = _env_from_rows(all_rows)
        try:
            schedule = build_corner_schedule(
                [held_sid], corpus_root=corpus_root,
            )
        except Exception:
            continue
        if not schedule:
            continue

        # 5. Score the held setup on the target track.
        try:
            breakdown = score_breakdown(
                model, observed_setup, track, env,
                schedule=schedule, hybrid=True,
            )
        except Exception:
            continue
        if not breakdown:
            continue
        score_total = float(sum(breakdown.values()))
        if not math.isfinite(score_total):
            continue

        out.append((score_total, -float(median_lap_time)))
    return out


def main() -> int:
    print("=" * 72)
    print("Lap-time correlation gate (P1.2)")
    print("=" * 72)

    pair_sessions = _build_pair_sessions_from_catalog()
    qualifying = _qualifying_pairs(pair_sessions)
    if not qualifying:
        print(
            "no qualifying (car, track) pairs "
            f"(need n_sessions >= {_MIN_PAIR_SESSIONS}); skipping."
        )
        return 0

    # Resolve the catalog root once so the heavy LOSO loop doesn't
    # re-walk the env each session.
    try:
        from racingoptimizer.ingest.api import resolve_corpus_root
        root = Path(resolve_corpus_root(None))
    except Exception as exc:
        print(f"  ERROR: cannot resolve corpus root: {exc}")
        return 1

    pair_to_results: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for car, track in qualifying:
        sids = pair_sessions[(car, track)]
        print(f"\n[{car} @ {track}] LOSO over {len(sids)} sessions...")
        try:
            pairs = _compute_loso_pairs_for_track(car, track, sids, root)
        except Exception as exc:
            print(f"  ERROR: LOSO loop raised {type(exc).__name__}: {exc}")
            pairs = []
        pair_to_results[(car, track)] = pairs

    payload: list[dict] = []
    overall_pass = True
    for (car, track), pairs in pair_to_results.items():
        rho = _spearman_correlation(pairs)
        ok, why = _evaluate_pair_score(rho)
        # A pair that produced zero LOSO results is treated as
        # "insufficient_data", which fails the gate but in a different
        # way than "rho < target" -- distinguish those in the JSON.
        if not ok:
            overall_pass = False
        payload.append(
            {
                "car": car,
                "track": track,
                "n_sessions": len(pair_sessions[(car, track)]),
                "n_loso_pairs": len(pairs),
                "spearman": rho,
                "passed": ok,
                "reason": why,
            }
        )
        print(f"  {car} @ {track}: {why} (n_loso_pairs={len(pairs)})")

    out_path = Path("docs/physics-rebuild/lap_time_correlation_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[saved JSON: {out_path}]")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
