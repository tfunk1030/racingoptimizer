from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from racingoptimizer.aero.residual_correction import AeroResidualCorrection
from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.model import CornerPhaseStateWithConfidence, PhysicsModel
from racingoptimizer.physics.score import (
    _corner_phase_objective_value,
    guardrail_warnings_for_setup,
)


def _ceilings() -> dict[str, AxleGripCeiling]:
    return {
        "front": AxleGripCeiling(
            car="bmw", axle="front", mu_peak=1.5,
            n_samples=100, n_above_ceiling=5, percentile_used=95.0,
        ),
        "rear": AxleGripCeiling(
            car="bmw", axle="rear", mu_peak=1.5,
            n_samples=100, n_above_ceiling=5, percentile_used=95.0,
        ),
    }


def _state(corner_id: int, phase: Phase = Phase.MID_CORNER) -> CornerPhaseStateWithConfidence:
    cpkey = CornerPhaseKey(
        session_id="sid",
        lap_index=1,
        corner_id=corner_id,
        phase=phase,
    )
    return CornerPhaseStateWithConfidence(
        corner_phase_key=cpkey,
        states={
            "accel_lat_g_max": Confidence(1.7, 1.6, 1.8, 10, "dense"),
            "lf_ride_height_mean_mm": Confidence(35.0, 34.5, 35.5, 10, "dense"),
            "lr_ride_height_mean_mm": Confidence(45.0, 44.5, 45.5, 10, "dense"),
        },
        untrained_channels=(),
    )


def _correction() -> AeroResidualCorrection:
    return AeroResidualCorrection(
        car="bmw",
        correction_factor=0.10,
        n_samples=120,
        fit_mae_raw_g=0.20,
        fit_mae_corrected_g=0.18,
        fallback_mode_used=False,
    )


def test_corner_phase_objective_threads_aero_residual_correction(monkeypatch) -> None:
    correction = _correction()
    model = PhysicsModel(
        car="bmw",
        session_ids=("sid",),
        axle_grip_ceilings=_ceilings(),
        aero_residual_correction=correction,
    )

    def _fake_eval(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["aero_correction"] is correction
        return SimpleNamespace(composite_score=0.72)

    monkeypatch.setattr("racingoptimizer.physics.evaluator.evaluate_corner_phase", _fake_eval)
    monkeypatch.setattr(
        "racingoptimizer.physics.axle_grip.compute_axle_grip_ratios",
        lambda *_a, **_k: {"front": np.array([0.3]), "rear": np.array([0.3])},
    )
    monkeypatch.setattr(
        "racingoptimizer.physics.axle_grip.axle_grip_margin",
        lambda *_a, **_k: 0.2,
    )
    monkeypatch.setattr(
        "racingoptimizer.physics.evaluator.guardrail_check",
        lambda *_a, **_k: SimpleNamespace(
            over_axle_ceiling=False,
            severely_off_balance=False,
            grip_inconsistency=False,
        ),
    )
    monkeypatch.setattr(
        "racingoptimizer.physics.hybrid_optimizer.hybrid_score",
        lambda **_k: SimpleNamespace(hybrid_score=0.55),
    )

    value = _corner_phase_objective_value(
        model=model,
        setup={"rear_wing_angle_deg": 10.0},
        env=EnvironmentFrame(),
        aero=None,
        state=_state(1),
        corner_id=1,
        phase_str="mid_corner",
        weights={1: 1.0},
        baselines=model.resolved_baselines,
    )
    assert value == 0.55


def test_guardrail_warnings_threads_aero_residual_correction(monkeypatch) -> None:
    correction = _correction()
    model = PhysicsModel(
        car="bmw",
        session_ids=("sid",),
        axle_grip_ceilings=_ceilings(),
        aero_residual_correction=correction,
    )

    monkeypatch.setattr(
        PhysicsModel,
        "predict",
        lambda self, setup, env, cpkey, *, corner_archetype=None: _state(
            int(cpkey.corner_id)
        ),
    )

    def _fake_eval(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["aero_correction"] is correction
        return SimpleNamespace(
            car="bmw",
            corner_id=1,
            phase="mid_corner",
            aero_balance_score=0.8,
            grip_headroom_score=0.8,
        )

    monkeypatch.setattr("racingoptimizer.physics.evaluator.evaluate_corner_phase", _fake_eval)
    monkeypatch.setattr(
        "racingoptimizer.physics.axle_grip.compute_axle_grip_ratios",
        lambda *_a, **_k: {"front": np.array([0.3]), "rear": np.array([0.3])},
    )
    monkeypatch.setattr(
        "racingoptimizer.physics.axle_grip.axle_grip_margin",
        lambda *_a, **_k: 0.2,
    )
    monkeypatch.setattr(
        "racingoptimizer.physics.evaluator.guardrail_check",
        lambda *_a, **_k: SimpleNamespace(
            flagged=True,
            reason="test guardrail",
        ),
    )

    schedule = [
        SimpleNamespace(
            corner_id=1,
            phase="mid_corner",
            archetype={"corner_apex_speed_ms": 40.0},
        )
    ]
    warnings = guardrail_warnings_for_setup(
        model=model,
        setup={"rear_wing_angle_deg": 10.0},
        env=EnvironmentFrame(),
        schedule=schedule,
    )
    assert warnings
    assert "test guardrail" in warnings[0]
