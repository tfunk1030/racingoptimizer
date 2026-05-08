"""Unit tests for the slug-resolution helper used by every CLI track lookup.

`_match_track_slug` is the single source of truth for going from a user-typed
track string ("Spa", "spa-2024", "lagunaseca") to a catalog slug. Four
call sites use it (audit punch-list T3.2): the per-(car, track) trust-radius
filter, the donor-track resolver, the per-car target resolver, and the
cross-car borrow path. The helper itself never mutates state — it just
returns ``(matched_slug, ambiguous_candidates)`` and lets callers apply
their own ambiguity policy. Pinning its rules here keeps the four call
sites from drifting back apart.
"""
from __future__ import annotations

from racingoptimizer.cli.recommend import _match_track_slug


def test_exact_slug_match() -> None:
    matched, ambiguous = _match_track_slug("spa_2024_up", ["spa_2024_up", "sebring"])
    assert matched == "spa_2024_up"
    assert ambiguous == []


def test_bare_alphanum_match() -> None:
    """`lagunaseca` matches a stored `laguna_seca` slug via the bare-form pass."""
    matched, ambiguous = _match_track_slug("lagunaseca", ["laguna_seca", "spa_2024_up"])
    assert matched == "laguna_seca"
    assert ambiguous == []


def test_slugify_normalises_dash_and_case() -> None:
    """`Laguna-Seca` / `LAGUNA SECA` both reach the same canonical slug."""
    available = ["lagunaseca", "spa_2024_up"]
    assert _match_track_slug("Laguna-Seca", available) == ("lagunaseca", [])
    assert _match_track_slug("LAGUNA SECA", available) == ("lagunaseca", [])


def test_substring_unique_match() -> None:
    """`spa` matches the only Spa-themed slug in the catalog."""
    matched, ambiguous = _match_track_slug(
        "spa", ["spa_2024_up", "sebring", "watkinsglen"],
    )
    assert matched == "spa_2024_up"
    assert ambiguous == []


def test_substring_multiple_returns_ambiguous_candidates() -> None:
    """Multiple substring hits surface to the caller — caller decides."""
    matched, ambiguous = _match_track_slug(
        "spa", ["spa_2024_up", "spa_old", "sebring"],
    )
    assert matched is None
    assert ambiguous == ["spa_2024_up", "spa_old"]


def test_no_match_returns_empty() -> None:
    matched, ambiguous = _match_track_slug("hockenheim", ["spa_2024_up", "sebring"])
    assert matched is None
    assert ambiguous == []


def test_empty_available_returns_empty() -> None:
    matched, ambiguous = _match_track_slug("anything", [])
    assert matched is None
    assert ambiguous == []


def test_whitespace_input_is_stripped() -> None:
    matched, _ = _match_track_slug("  spa_2024_up  ", ["spa_2024_up"])
    assert matched == "spa_2024_up"


def test_canonical_cars_drives_cross_car_borrow_loop() -> None:
    """`_maybe_borrow_cross_car_track` walks `CANONICAL_CARS`; pinning the
    coupling here surfaces drift if the loop is ever hardcoded again
    (audit T3.10 regression guard).
    """
    from racingoptimizer.cli.recommend import (
        CANONICAL_CARS,
        _maybe_borrow_cross_car_track,
    )
    import inspect
    src = inspect.getsource(_maybe_borrow_cross_car_track)
    assert "CANONICAL_CARS" in src, (
        "_maybe_borrow_cross_car_track must iterate CANONICAL_CARS so the "
        "list of GTP cars stays single-sourced."
    )
    # The five known GTP cars.
    assert set(CANONICAL_CARS) == {
        "acura", "bmw", "cadillac", "ferrari", "porsche",
    }
