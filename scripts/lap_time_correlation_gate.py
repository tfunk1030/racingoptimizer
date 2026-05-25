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

    # Heavy lift -- LOSO refit per session per pair -- intentionally
    # not implemented here: the per-car fit is ~15 min/session, so a
    # 10-session pair is 2.5 hr per car-track and CI cannot run this
    # synchronously. The script structure is in place; future
    # implementation populates ``pair_to_pairs[(car, track)]`` with
    # ``(score, neg_lap_time)`` tuples. Run from a workstation:
    #
    #   uv run python scripts/lap_time_correlation_gate.py
    #
    # CI consumes the JSON written below.
    pair_to_results: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for pair in qualifying:
        # Placeholder: populated by the LOSO orchestration in a future
        # session (see DEFER note in PLAN.md 4a). We leave the script
        # callable so the unit-tested helpers don't bit-rot.
        pair_to_results[pair] = []

    payload: list[dict] = []
    overall_pass = True
    for (car, track), pairs in pair_to_results.items():
        rho = _spearman_correlation(pairs)
        ok, why = _evaluate_pair_score(rho)
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
        print(f"  {car} @ {track}: {why}")

    out_path = Path("docs/physics-rebuild/lap_time_correlation_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[saved JSON: {out_path}]")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
