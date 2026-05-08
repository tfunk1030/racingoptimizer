"""Unit tests for the narrative renderer's `_car_feel` lookup table.

`_CAR_FEEL` carries the handling-vocabulary phrases (effect + trade) that
the narrative renderer uses for every parameter that moves. The audit
flagged two issues we fixed in this session:

  T3.5 - Spa-specific landmarks ("Eau Rouge", "Pouhon", "Kemmel", "T9 / T13")
         baked into entries that fire on every track. Replaced with
         archetypal phrases.
  T3.6 - Missing per-axle entries: rear torsion bar, ride_height (front /
         rear), and corner_weight (front / rear). Added 8 entries.

These tests pin both fixes so the table can't regress to per-track
landmarks or lose the per-axle entries silently.
"""
from __future__ import annotations

from racingoptimizer.explain.narrative import _CAR_FEEL, _car_feel


def _all_strings() -> list[str]:
    out: list[str] = []
    for effect, trade in _CAR_FEEL.values():
        out.append(effect)
        out.append(trade)
    return out


def test_no_spa_landmark_phrases_in_car_feel() -> None:
    """T3.5 regression: every effect/trade string must be track-neutral.

    The narrative is rendered for every track the optimizer recommends a
    setup for; landmarks like "Eau Rouge" are misleading at Sebring.
    """
    forbidden = (
        "Eau Rouge", "Pouhon", "Blanchimont", "Kemmel",
    )
    bad: list[tuple[str, str]] = []
    for s in _all_strings():
        for token in forbidden:
            if token in s:
                bad.append((token, s))
    assert not bad, (
        "Spa-specific landmarks resurfaced in _CAR_FEEL — every entry "
        f"must use archetypal phrasing. Hits: {bad}"
    )


def test_no_explicit_corner_numbers_in_car_feel() -> None:
    """Per T3.5 regression: no `T1 / T3 / T9 / T13 / T16` style references.

    Allow ASCII fragments inside parens that are NOT corner numbers
    (e.g. `LSC`, `HSC`); narrowly check for `T<digits>` only.
    """
    import re
    pat = re.compile(r"\bT\d+\b")
    bad: list[str] = []
    for s in _all_strings():
        if pat.search(s):
            bad.append(s)
    assert not bad, (
        f"_CAR_FEEL must not name specific corner numbers — found: {bad}"
    )


def test_per_axle_torsion_entries_present() -> None:
    """T3.6 regression: rear torsion bar entries (both directions)."""
    assert ("torsion_bar", "rear", "+") in _CAR_FEEL
    assert ("torsion_bar", "rear", "-") in _CAR_FEEL


def test_ride_height_entries_present_both_axles() -> None:
    for axle in ("front", "rear"):
        for direction in ("+", "-"):
            assert ("ride_height", axle, direction) in _CAR_FEEL, (
                f"missing ride_height ({axle}, {direction})"
            )


def test_corner_weight_entries_present_both_axles() -> None:
    for axle in ("front", "rear"):
        for direction in ("+", "-"):
            assert ("corner_weight", axle, direction) in _CAR_FEEL, (
                f"missing corner_weight ({axle}, {direction})"
            )


def test_car_feel_returns_tuple_for_known_family() -> None:
    """Sanity: dispatch through `_car_feel` resolves a known family."""
    pair = _car_feel("rear_wing", "rear_wing_angle_deg", delta=+1.0)
    assert pair is not None
    effect, trade = pair
    assert isinstance(effect, str) and effect
    assert isinstance(trade, str) and trade


def test_car_feel_unknown_family_returns_none() -> None:
    """Unknown families fall through to None so the renderer can fall
    back to the generic phase-themed line.
    """
    assert _car_feel("nonexistent_family", "fake_param", delta=+1.0) is None


def test_car_feel_resolves_direction_sign() -> None:
    """Positive vs negative delta routes to different entries."""
    plus = _car_feel("rear_wing", "rear_wing_angle_deg", delta=+1.0)
    minus = _car_feel("rear_wing", "rear_wing_angle_deg", delta=-1.0)
    assert plus is not None
    assert minus is not None
    assert plus != minus, (
        "+/- direction must select different (effect, trade) tuples"
    )
