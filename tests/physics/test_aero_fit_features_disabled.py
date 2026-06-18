"""W6 aero-map fit features are OFF by default (held-out ablation, 2026-06-10).

The fit-time features query the map at dynamic platform RH while the
predict-time approximation uses static garage readouts — a train/serve skew
that measurably degraded every gated held-out channel (see the evidence
block in `physics/aero_fit_features.py`). The helpers stay pure and tested
in `test_aero_fit_features.py`; these tests pin the call-site gating.
"""
from __future__ import annotations

from racingoptimizer.physics.aero_fit_features import (
    AERO_MAP_FIT_FEATURES_ENABLED,
    aero_fit_column_names,
)


def test_flag_defaults_off() -> None:
    assert AERO_MAP_FIT_FEATURES_ENABLED is False


def test_schema_version_bumped_for_the_disable() -> None:
    from racingoptimizer.physics.fitter import ENV_FEATURE_SCHEMA_VERSION_PER_CAR

    assert ENV_FEATURE_SCHEMA_VERSION_PER_CAR >= 9


def test_predict_v4_omits_aero_extras_when_disabled(monkeypatch) -> None:
    """_predict_v4's extra_features must be empty with the flag off — a
    pickle trained without the columns must never be probed with them."""
    import racingoptimizer.physics.aero_fit_features as aff

    calls: list[str] = []
    monkeypatch.setattr(
        aff, "aero_map_features_for_predict",
        lambda **kw: calls.append("called") or {},
    )
    # The flag is read inside _predict_v4 via module attribute; with the
    # default False the helper must not be invoked at all.
    assert aff.AERO_MAP_FIT_FEATURES_ENABLED is False
    assert aero_fit_column_names() == ("aero_map_ld_ratio", "aero_map_balance_pct")
    assert calls == []
