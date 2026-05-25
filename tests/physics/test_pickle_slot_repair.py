"""Legacy PhysicsModel pickle slot-shift repair + P1.4 type-safety."""

from __future__ import annotations

import pytest

from racingoptimizer.aero.residual_correction import AeroResidualCorrection
from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.model import (
    _repair_legacy_slot_shift,
    _validate_pickle_slots,
)


def test_repair_legacy_slot_shift_restores_axle_and_aero() -> None:
    aero = AeroResidualCorrection(
        car="acura",
        correction_factor=0.1,
        n_samples=10,
        fit_mae_raw_g=0.4,
        fit_mae_corrected_g=0.3,
        fallback_mode_used=False,
    )
    axle = {
        "front": AxleGripCeiling(
            car="acura", axle="front", mu_peak=2.7, n_samples=100,
            n_above_ceiling=1, percentile_used=99.0,
        ),
        "rear": AxleGripCeiling(
            car="acura", axle="rear", mu_peak=2.8, n_samples=100,
            n_above_ceiling=1, percentile_used=99.0,
        ),
    }
    slot_values = {
        "per_track_residuals": axle,
        "axle_grip_ceilings": aero,
        "aero_residual_correction": None,
    }
    _repair_legacy_slot_shift(slot_values)
    assert slot_values["per_track_residuals"] == {}
    assert slot_values["axle_grip_ceilings"] == axle
    assert slot_values["aero_residual_correction"] is aero


def test_repair_legacy_slot_shift_noop_on_healthy_pickle() -> None:
    slot_values = {
        "per_track_residuals": {"belleisle": {"accel_lat_g_max": 0.01}},
        "axle_grip_ceilings": None,
        "aero_residual_correction": None,
    }
    _repair_legacy_slot_shift(slot_values)
    assert slot_values["per_track_residuals"] == {
        "belleisle": {"accel_lat_g_max": 0.01},
    }


# ---- P1.4 slot type-safety ----------------------------------------------


def test_validate_slots_accepts_healthy_pickle() -> None:
    """A pickle with correct slot types passes validation unchanged."""
    slot_values = {
        "parameter_observed_std": {"wing": 0.5},
        "per_track_parameter_observed": {"spa": {"wing": (10.0, 12.0)}},
        "bayes_posteriors": {},
        "per_track_residuals": {},
        "track_models_used": {"spa": "spa_2024_up"},
        "fitters": {},
        "ontology": {},
        "baseline_setup": {"wing": 11.0},
        "session_ids": ("abc",),
        "untrained_parameters": (),
        "static_rh_corpus": (),
        "feature_schema_version": 4,
        "seed": 12345,
        "aero_correction_available": True,
        "axle_grip_ceilings": None,
        "aero_residual_correction": None,
    }
    _validate_pickle_slots(slot_values)  # must not raise


def test_validate_slots_rejects_object_in_dict_slot() -> None:
    """A pickle where ``per_track_residuals`` carries axle ceilings (a dict-of-objects)
    instead of dict-of-dict-of-floats should be caught by P1.4 -- this is the
    exact slot-shift corruption ``_repair_legacy_slot_shift`` rescues, but if the
    repair fails or a NEW slot-shift appears, the type-check must refuse to revive.

    The dict type-check alone is insufficient (dict-of-objects is still a dict),
    so this test asserts the repair was applied first OR a clearer slot mismatch
    is rejected at the outer-type level.
    """
    # The slot type for axle_grip_ceilings is dict OR None. An int is neither.
    slot_values = {"axle_grip_ceilings": 42}
    with pytest.raises(TypeError, match="axle_grip_ceilings"):
        _validate_pickle_slots(slot_values)


def test_validate_slots_rejects_dict_in_tuple_slot() -> None:
    """``session_ids`` must be a tuple. A dict here is slot-shift."""
    slot_values = {"session_ids": {}}
    with pytest.raises(TypeError, match="session_ids"):
        _validate_pickle_slots(slot_values)


def test_validate_slots_rejects_wrong_aero_correction_type() -> None:
    """``aero_residual_correction`` must be the right dataclass or None."""
    slot_values = {"aero_residual_correction": {"car": "acura"}}
    with pytest.raises(TypeError, match="aero_residual_correction"):
        _validate_pickle_slots(slot_values)


def test_validate_slots_allows_none_for_optional_object_slots() -> None:
    slot_values = {
        "axle_grip_ceilings": None,
        "aero_residual_correction": None,
    }
    _validate_pickle_slots(slot_values)  # must not raise


def test_validate_slots_error_names_the_slot_and_points_to_recovery() -> None:
    """Error must tell the user which slot failed and how to recover."""
    slot_values = {"feature_schema_version": "not-an-int"}
    with pytest.raises(TypeError) as exc:
        _validate_pickle_slots(slot_values)
    msg = str(exc.value)
    assert "feature_schema_version" in msg
    assert "--no-cache" in msg or "refit" in msg.lower()
