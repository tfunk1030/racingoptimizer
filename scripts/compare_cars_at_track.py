"""Rank every GTP car at one track by a weighted physics + lap-time composite.

This operationalises ``docs/watkins-glen-runbook.md`` Step 3. For each car it
invokes the *production* recommend path (``optimize <car> <track> --json``) to
get the optimised setup's aggregate physics utilization (``score_total``, added
to the JSON in 2026-06), and queries the catalog for the car's best clean lap at
the track. Both axes are min-max normalised across the cars, then combined:

    Composite(car) = W_LAP * LapScore + W_PHYS * PhysScore

where ``LapScore`` is normalised *inverse* lap time (fastest car = 1.0) and
``PhysScore`` is normalised ``score_total`` (best-exploiting setup = 1.0).

Lap time is the only strictly cross-car-comparable axis; ``score_total`` uses
per-car-calibrated evaluator weights, so it is meaningful only after the
per-car min-max normalisation here, and even then is a soft signal (it guards
against a car that is quick on a thin sample while riding its grip ceiling).

Usage (run under ``uv`` so the ``optimize`` console script + deps resolve)::

    uv run python scripts/compare_cars_at_track.py "watkins glen"
    uv run python scripts/compare_cars_at_track.py spa --w-lap 0.7 --w-phys 0.3
    uv run python scripts/compare_cars_at_track.py spa --cars bmw cadillac

Each car triggers a ~15-min per-car refit on a cold cache; expect the first run
to be slow. Cars that refuse (thin corpus) or have no clean laps at the track are
reported and excluded from the ranking.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

CANONICAL_CARS: tuple[str, ...] = ("acura", "bmw", "cadillac", "ferrari", "porsche")


@dataclass
class CarResult:
    car: str
    score_total: float | None  # aggregate physics utilization at recommended setup
    best_lap_s: float | None   # fastest clean lap at the track, from the catalog
    note: str = ""             # populated when the car is excluded / degraded


# --------------------------------------------------------------------------- #
# Pure scoring core (unit-testable without a corpus)
# --------------------------------------------------------------------------- #
def _min_max(values: dict[str, float], *, invert: bool) -> dict[str, float]:
    """Min-max normalise to ``0..1``. ``invert`` flips so the SMALLEST input
    maps to 1.0 (used for lap time, where lower is better).

    Degenerate cases: a single value, or all-equal values, map to 1.0 (there is
    no spread to discriminate on, so neither car is penalised on that axis).
    """
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo <= 0.0:
        return {k: 1.0 for k in values}
    out: dict[str, float] = {}
    for k, v in values.items():
        norm = (v - lo) / (hi - lo)
        out[k] = (1.0 - norm) if invert else norm
    return out


def rank(
    results: list[CarResult], *, w_lap: float, w_phys: float
) -> list[tuple[str, float, float, float, CarResult]]:
    """Return ``[(car, lap_score, phys_score, composite, result), ...]`` sorted
    by composite descending. Only cars with BOTH a lap time and a score_total
    participate; others are returned at the end with composite ``nan``.
    """
    ranked_cars = [
        r for r in results if r.best_lap_s is not None and r.score_total is not None
    ]
    lap_scores = _min_max(
        {r.car: r.best_lap_s for r in ranked_cars}, invert=True  # type: ignore[misc]
    )
    phys_scores = _min_max(
        {r.car: r.score_total for r in ranked_cars}, invert=False  # type: ignore[misc]
    )
    rows: list[tuple[str, float, float, float, CarResult]] = []
    for r in ranked_cars:
        ls, ps = lap_scores[r.car], phys_scores[r.car]
        composite = w_lap * ls + w_phys * ps
        rows.append((r.car, ls, ps, composite, r))
    rows.sort(key=lambda t: t[3], reverse=True)
    # Append excluded cars (composite nan) so they show in the report.
    for r in results:
        if r.best_lap_s is None or r.score_total is None:
            rows.append((r.car, float("nan"), float("nan"), float("nan"), r))
    return rows


# --------------------------------------------------------------------------- #
# Data collection (needs a populated corpus)
# --------------------------------------------------------------------------- #
def _recommend_score_total(
    car: str, track: str, corpus_root: str | None
) -> tuple[float | None, str]:
    """Run ``optimize <car> <track> --json`` and pull ``score_total``.

    Returns ``(score_total, note)``. ``note`` is non-empty on refusal/failure.
    """
    cmd = ["optimize", car, track, "--json", "--output-file", "-"]
    if corpus_root:
        cmd += ["--corpus-root", corpus_root]
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603,S607
    if proc.returncode != 0:
        return None, f"recommend exited {proc.returncode}: {proc.stderr.strip()[:160]}"
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, f"non-JSON output: {proc.stdout.strip()[:160]}"
    score = payload.get("score_total")
    if score is None:
        warn = "; ".join(payload.get("warnings", [])) or "no score_total in output"
        return None, f"refused/degraded: {warn[:160]}"
    return float(score), ""


def _best_clean_lap(car: str, track: str, corpus_root: str | None) -> float | None:
    """Fastest valid lap for ``(car, track)`` from the catalog, or None.

    The track string is resolved to the catalog slug with the same matcher the
    CLI uses, so ``"watkins glen"`` / ``watkinsglen`` line up with the stored
    slug regardless of spacing/underscores.
    """
    import polars as pl

    from racingoptimizer.cli.recommend import _match_track_slug
    from racingoptimizer.ingest import api as ingest_api
    from racingoptimizer.ingest.paths import resolve_corpus_root

    root = resolve_corpus_root(corpus_root)  # type: ignore[arg-type]
    sessions = ingest_api.sessions(car=car, corpus_root=root)
    if sessions.is_empty():
        return None
    available = sorted(set(sessions["track"].to_list()))
    slug, _ambiguous = _match_track_slug(track, available)
    if slug is None:
        return None
    laps = ingest_api.laps(car=car, track=slug, valid_only=True, corpus_root=root)
    if laps.is_empty():
        return None
    times = laps.filter(pl.col("lap_time_s").is_not_null() & (pl.col("lap_time_s") > 0.0))
    if times.is_empty():
        return None
    return float(times["lap_time_s"].min())


def collect(cars: list[str], track: str, corpus_root: str | None) -> list[CarResult]:
    results: list[CarResult] = []
    for car in cars:
        print(f"[{car}] recommending at {track!r} (cold cache => ~15 min)...", file=sys.stderr)
        score, note = _recommend_score_total(car, track, corpus_root)
        best_lap = _best_clean_lap(car, track, corpus_root)
        if best_lap is None and not note:
            note = "no clean laps at this track in the catalog"
        results.append(CarResult(car=car, score_total=score, best_lap_s=best_lap, note=note))
    return results


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return "  --  " if x != x else f"{x:.3f}"  # x != x is NaN


def render_table(rows, *, track: str, w_lap: float, w_phys: float) -> str:
    header = (
        f"{'rank':<5}{'car':<10}{'best lap':>10}"
        f"{'LapScore':>10}{'PhysScore':>11}{'Composite':>11}  notes"
    )
    out = [
        f"Best car at {track!r}  "
        f"(Composite = {w_lap:.2f}*LapScore + {w_phys:.2f}*PhysScore)",
        "",
        header,
        "-" * 78,
    ]
    rank_n = 0
    for car, ls, ps, comp, res in rows:
        excluded = comp != comp  # NaN
        rank_label = "-" if excluded else str(rank_n + 1)
        if not excluded:
            rank_n += 1
        lap_str = "  --  " if res.best_lap_s is None else f"{res.best_lap_s:.3f}s"
        out.append(
            f"{rank_label:<5}{car:<10}{lap_str:>10}{_fmt(ls):>10}{_fmt(ps):>11}"
            f"{_fmt(comp):>11}  {res.note}"
        )
    out += [
        "",
        "PhysScore uses per-car-calibrated weights; comparable only after the",
        "min-max normalisation above. With one Glen session per car the ranking",
        "is PROVISIONAL -- more laps (esp. Acura) firm it up.",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank GTP cars at one track (physics + lap time).")
    ap.add_argument("track", help="track name/slug, e.g. 'watkins glen'")
    ap.add_argument("--cars", nargs="+", default=list(CANONICAL_CARS), help="subset of cars")
    ap.add_argument("--w-lap", type=float, default=0.6, help="lap-time weight (default 0.6)")
    ap.add_argument("--w-phys", type=float, default=0.4, help="physics weight (default 0.4)")
    ap.add_argument("--corpus-root", default=None, help="override corpus root")
    args = ap.parse_args(argv)

    total_w = args.w_lap + args.w_phys
    if total_w <= 0:
        ap.error("weights must sum to a positive number")
    w_lap, w_phys = args.w_lap / total_w, args.w_phys / total_w  # renormalise

    results = collect(args.cars, args.track, args.corpus_root)
    rows = rank(results, w_lap=w_lap, w_phys=w_phys)
    print(render_table(rows, track=args.track, w_lap=w_lap, w_phys=w_phys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
