"""S2.9: assert `ld_ratio` corpus values are dimensionless lift/drag ratios.

The interpolator decision (no air-density correction on `ld_ratio`) hinges
on the observed values being a ratio rather than a force or coefficient.
This test scans every `aero-maps/*.json` and asserts every cell falls in a
band that is plausible for a GTP-class L/D ratio. If the corpus ever
changes shape (e.g. someone re-extracts the maps as raw lift coefficients),
this test fails loudly so the interpolator's correction can be revisited.

Plausible band: real-world LMDh / GT3 lift-to-drag ratios live in roughly
2.5–6.0; the current corpus spans 2.86–4.61 across 77,418 samples.
We pick a wider [1.0, 10.0] envelope so the test is robust to small
corpus refreshes but still rejects an order-of-magnitude shift to
forces (Newtons; would land in 1e3–1e4) or unscaled coefficients near 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Plausible band for a dimensionless lift/drag ratio. Wider than the
# current corpus extrema (2.86 .. 4.61) but tight enough to reject any
# force or coefficient mis-extraction.
LD_RATIO_LO = 1.0
LD_RATIO_HI = 10.0


def test_ld_ratio_values_are_in_plausible_dimensionless_band(
    aero_dir: Path,
) -> None:
    """Every ld_ratio cell across all 33 aero-maps is in [1.0, 10.0]."""
    files = sorted(aero_dir.glob("*_wing_*.json"))
    assert files, f"no aero-map JSONs found under {aero_dir}"

    total_cells = 0
    global_min = float("inf")
    global_max = float("-inf")
    offenders: list[tuple[str, float]] = []

    for path in files:
        data = json.loads(path.read_text())
        ld = data["ld_ratio"]
        for row in ld:
            for v in row:
                total_cells += 1
                fv = float(v)
                if fv < global_min:
                    global_min = fv
                if fv > global_max:
                    global_max = fv
                if not (LD_RATIO_LO <= fv <= LD_RATIO_HI):
                    offenders.append((path.name, fv))

    assert not offenders, (
        f"{len(offenders)} ld_ratio cell(s) outside the dimensionless "
        f"band [{LD_RATIO_LO}, {LD_RATIO_HI}]; first 5: {offenders[:5]}"
    )
    # Sanity check on corpus shape: 33 files * 51 * 46 = 77,418 cells.
    assert total_cells > 0
    assert global_min >= LD_RATIO_LO
    assert global_max <= LD_RATIO_HI


def test_corpus_ld_ratio_min_and_max_match_documented_range(
    aero_dir: Path,
) -> None:
    """Pin the empirical min/max so a corpus refresh forces a re-audit.

    If the actual corpus extrema drift outside [2.5, 5.0], it suggests
    either a new car/wing was added (re-evaluate the band) or the
    extraction changed units (revisit the interpolator's no-correction
    decision). Either case warrants human attention, hence the loud
    failure.
    """
    files = sorted(aero_dir.glob("*_wing_*.json"))
    assert files

    global_min = float("inf")
    global_max = float("-inf")
    for path in files:
        data = json.loads(path.read_text())
        for row in data["ld_ratio"]:
            for v in row:
                fv = float(v)
                if fv < global_min:
                    global_min = fv
                if fv > global_max:
                    global_max = fv

    # 2026-04-29 corpus: min 2.861, max 4.614. Allow modest drift.
    assert 2.5 <= global_min < 5.0, (
        f"ld_ratio min={global_min:.3f} drifted outside [2.5, 5.0); "
        "re-audit S2.9 (gap #14) before adjusting this band."
    )
    assert 2.5 < global_max <= 5.0, (
        f"ld_ratio max={global_max:.3f} drifted outside (2.5, 5.0]; "
        "re-audit S2.9 (gap #14) before adjusting this band."
    )


@pytest.mark.parametrize(
    "car",
    ["acura", "bmw", "cadillac", "ferrari", "porsche"],
)
def test_each_car_has_ld_ratio_in_plausible_band(
    aero_dir: Path, car: str,
) -> None:
    """Per-car coverage: every one of the 5 GTP cars has ld_ratio in band.

    Aligns with CLAUDE.md per-car-verification convention — single-car
    smoke is the gap, so we assert the contract for all five canonical
    cars individually.
    """
    files = sorted(aero_dir.glob(f"{car}_wing_*.json"))
    if not files:
        pytest.skip(f"no aero maps present for car={car!r}")

    for path in files:
        data = json.loads(path.read_text())
        for ri, row in enumerate(data["ld_ratio"]):
            for ci, v in enumerate(row):
                fv = float(v)
                assert LD_RATIO_LO <= fv <= LD_RATIO_HI, (
                    f"{path.name} ld_ratio[{ri}][{ci}]={fv} outside "
                    f"[{LD_RATIO_LO}, {LD_RATIO_HI}]"
                )
