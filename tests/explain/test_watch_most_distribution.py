"""P3.2 -- Watch-most picker distributes across corners.

Without normalization, the corner-duration weighting from e90e8fd
made one long corner (e.g. T18 on Belleisle) dominate every parameter's
``Watch most`` line. This test fabricates a 20-corner pool with a
realistic impact distribution where one "long corner" emits a
high-magnitude impact across every phase and other corners emit
mechanically-concentrated impacts in fewer phases. After P3.2's
normalisation, no single corner may account for more than 30 % of
``Watch most`` lines across a representative parameter set.
"""
from __future__ import annotations

import random

from racingoptimizer.confidence import Confidence
from racingoptimizer.corner.phase import Phase
from racingoptimizer.explain.justification import (
    CornerPhaseImpact,
    SetupJustification,
)
from racingoptimizer.explain.narrative import _dominant_impact_corner


def _impact(corner_id: int, phase: Phase, score: float) -> CornerPhaseImpact:
    return CornerPhaseImpact(
        corner_id=corner_id, phase=phase, score_delta=score, note="",
    )


def _justification_with_pool(
    parameter: str, helps: list[CornerPhaseImpact], hurts: list[CornerPhaseImpact],
) -> SetupJustification:
    return SetupJustification(
        parameter=parameter,
        value=0.0,
        unit="",
        confidence=Confidence(value=0.0, lo=-1.0, hi=1.0, n_samples=10, regime="dense"),
        corners_helped=tuple(helps),
        corners_hurt=tuple(hurts),
        sensitivity_minus_1_click=0.001,
        sensitivity_plus_1_click=0.001,
        telemetry_evidence=("synthetic",),
    )


def _build_realistic_pool_for_parameter(
    parameter: str, family: str, *, rng: random.Random,
) -> SetupJustification:
    """Build a 20-corner impact pool where one "long corner" (T18) emits
    high-magnitude impact across all 5 phases (duration-weighted), and
    other corners emit lower-magnitude impacts on a subset of phases
    (mechanically-localised).
    """
    helps: list[CornerPhaseImpact] = []
    hurts: list[CornerPhaseImpact] = []
    long_phases = (
        Phase.BRAKING,
        Phase.TRAIL_BRAKE,
        Phase.MID_CORNER,
        Phase.EXIT,
        Phase.STRAIGHT,
    )
    # T18 is the "long corner" -- duration-weighted high score on every
    # phase. Pre-P3.2 this dominates every parameter's Watch most.
    for phase in long_phases:
        score = rng.uniform(0.6, 0.9)
        (helps if rng.random() < 0.5 else hurts).append(_impact(18, phase, score))
    # 19 other corners with localised impact -- 1-2 phases each, lower
    # raw magnitude but higher PER-PHASE concentration.
    short_phase_options = list(long_phases)
    for cid in range(20):
        if cid == 18:
            continue
        n_phases = rng.choice([1, 2])
        chosen_phases = rng.sample(short_phase_options, n_phases)
        for phase in chosen_phases:
            score = rng.uniform(0.30, 0.55)
            (helps if rng.random() < 0.5 else hurts).append(
                _impact(cid, phase, score),
            )
    return _justification_with_pool(parameter, helps, hurts)


_FAMILY_PARAMS: tuple[tuple[str, str], ...] = (
    ("rear_wing", "rear_wing"),
    ("tyre_cold_pressure_kpa", "tyre_pressure"),
    ("heave_perch_offset_front_mm", "perch_offset"),
    ("pushrod_length_offset_front_mm", "pushrod"),
    ("third_spring_rate_n_per_mm", "spring_rate"),
    ("torsion_bar_turns_fl", "torsion_bar"),
    ("front_arb_size_idx", "arb"),
    ("front_lf_lo_compression_clicks", "damper"),
    ("camber_fl_deg", "camber"),
    ("brake_bias_pct", "brake_bias"),
    ("front_diff_preload_n_m", "diff"),
    ("fuel_level_l", "fuel"),
)


def test_no_single_corner_dominates_across_parameters() -> None:
    """Across a representative spread of parameters with families that
    map to different preferred phases, no single corner picks up more
    than 30 % of the Watch-most assignments.
    """
    rng = random.Random(20260524)
    picks: list[str] = []
    for parameter, family in _FAMILY_PARAMS:
        j = _build_realistic_pool_for_parameter(parameter, family, rng=rng)
        out = _dominant_impact_corner(j, family=family)
        assert out  # picker must always return a label when the pool is non-empty
        picks.append(out.split()[0])  # "T18 mid-corner" -> "T18"

    counts: dict[str, int] = {}
    for label in picks:
        counts[label] = counts.get(label, 0) + 1
    total = len(picks)
    most_picked, share = max(counts.items(), key=lambda kv: kv[1])
    fraction = share / total
    assert fraction <= 0.30, (
        f"single corner {most_picked} accounts for {fraction:.0%} of "
        f"Watch most lines (target <= 30 %). distribution: {counts}"
    )


def test_concentrated_impact_beats_spread_impact_at_same_total() -> None:
    """A corner with one high-impact phase beats a corner that emits
    the same total score across many phases. This is the core P3.2
    regression guard."""
    helps = [
        # T1: ONE big mid-corner hit
        _impact(1, Phase.MID_CORNER, 1.0),
        # T18: spread impact across five phases (total > T1, but
        # duration-driven not mechanism-driven)
        _impact(18, Phase.BRAKING, 0.6),
        _impact(18, Phase.TRAIL_BRAKE, 0.6),
        _impact(18, Phase.MID_CORNER, 0.6),
        _impact(18, Phase.EXIT, 0.6),
        _impact(18, Phase.STRAIGHT, 0.6),
    ]
    j = _justification_with_pool("rear_wing", helps=helps, hurts=[])
    out = _dominant_impact_corner(j, family="rear_wing")
    assert out.startswith("T1 "), (
        f"expected T1 (concentrated impact) to win, got: {out}"
    )


def test_pure_max_picks_long_corner_without_normalisation() -> None:
    """Sanity: without normalisation, T18 would win on raw |impact|.
    Verifies the test case is actually exercising the normalisation
    (not a degenerate case where the answer is the same either way).
    """
    helps = [
        _impact(1, Phase.MID_CORNER, 1.0),
        _impact(18, Phase.BRAKING, 0.6),
        _impact(18, Phase.TRAIL_BRAKE, 0.6),
        _impact(18, Phase.MID_CORNER, 1.5),  # bigger than T1
        _impact(18, Phase.EXIT, 0.6),
        _impact(18, Phase.STRAIGHT, 0.6),
    ]
    pool = helps
    raw_max = max(pool, key=lambda i: abs(i.score_delta))
    assert raw_max.corner_id == 18  # raw max is T18 mid_corner @ 1.5
    # But normalised: T18 mid_corner = 1.5 / 5 = 0.30; T1 = 1.0 / 1 = 1.0
    # So T1 wins after normalisation
    j = _justification_with_pool("rear_wing", helps=helps, hurts=[])
    out = _dominant_impact_corner(j, family="rear_wing")
    assert out.startswith("T1 "), (
        f"normalised picker should pick T1, got: {out}"
    )
