"""Coupled-physics propagation test (Stage 3 — VISION §3 / §5).

Pre-Stage-3 the architecture trained one fitter per parameter and summed
their predictions linearly per output channel. That meant changing one
setup parameter affected only the predictions of the fitters keyed on
that parameter. The previous `test_score_locality::test_score_breakdown_locality`
verified that property — i.e. it AFFIRMED a non-coupled architecture.

Stage-3 trains one fitter per (corner_id, phase, output_channel) over the
FULL bounded setup vector + 12 env channels. Perturbing any single setup
parameter must therefore propagate to multiple (corner, phase) cells —
"stiffer springs change ride heights which change aero balance which
change load transfer which change tire temps which change grip. Chase
the chain." (VISION §5).

This test fits a tiny synthetic coupled model where front spring affects
ride heights and lateral G across multiple corners, then asserts that
perturbing the spring value moves ≥3 (corner, phase) cells in the
score_breakdown.
"""
from __future__ import annotations

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.fitter import _ENV_COLUMNS, ENV_FEATURE_SCHEMA_VERSION
from racingoptimizer.physics.fitters import GPFitter
from racingoptimizer.physics.model import FitRecord, PhysicsModel


def _make_record(
    *,
    parameters: list[str],
    coefficients: list[float],
    bias: float,
    n_train: int = 60,
    seed: int = 0,
) -> FitRecord:
    """Train a tiny GP on a synthetic linear coupling.

    The fitter sees feature vector = `parameters` + 12 env channels. We
    train it on samples whose target is `bias + sum(coef_i * param_i)`,
    so the trained surrogate is coupled to *every* parameter in the
    feature vector.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_features = len(parameters) + len(_ENV_COLUMNS)
    X = rng.uniform(0.0, 10.0, size=(n_train, n_features))
    coef_vec = np.zeros(n_features, dtype=np.float64)
    for i, _ in enumerate(parameters):
        coef_vec[i] = coefficients[i]
    y = bias + X @ coef_vec + rng.normal(0.0, 0.05, size=n_train)

    fitter = GPFitter(random_state=seed)
    fitter.fit(X, y)
    return FitRecord(
        fitter=fitter,
        n_samples=n_train,
        cv_residual_std=0.05,
        signal_std=float(y.std(ddof=0)),
        feature_names=tuple(parameters) + tuple(_ENV_COLUMNS),
    )


def test_perturb_one_parameter_changes_multiple_corners() -> None:
    """Perturbing front spring must change at least 3 (corner, phase) cells.

    This is the inverse of the old `test_score_breakdown_locality` claim:
    that one perturbed *fitter* changed only its own cell affirmed
    non-coupling. Now we perturb a single *parameter* and confirm the
    change propagates through every fitter that depends on it — proof
    the coupled architecture works end-to-end through `score_breakdown`.
    """
    parameters = [
        "heave_spring_mm",
        "static_ride_height_front_mm",
        "rear_wing_angle_deg",
    ]
    # Baselines and perturbation deltas live inside the [0, 10] training
    # envelope. The fixture parameters are NOT meant to mirror real iRacing
    # ranges — they exist only to exercise the joint multi-input path.
    baseline = {
        "heave_spring_mm": 5.0,
        "static_ride_height_front_mm": 5.0,
        "rear_wing_angle_deg": 5.0,
    }

    fitters: dict[tuple, FitRecord] = {}
    # Three corners × two phases = 6 cells, each with one channel that
    # depends on heave_spring_mm. Coefficients sized so the resulting
    # utilization values stay inside [0, 1] (saturation hides perturbation
    # signal) and the per-cell delta lands above the 1e-5 score-function
    # noise floor we use as the "moved cell" threshold.
    for corner_id in (1, 2, 3):
        for phase in ("mid_corner", "exit"):
            for channel, coefs, bias in (
                ("accel_lat_g_max", [0.05, 0.01, 0.01], 0.5),
                ("understeer_angle_mean_rad", [0.005, 0.001, 0.001], 0.01),
            ):
                fitters[(corner_id, phase, channel)] = _make_record(
                    parameters=parameters,
                    coefficients=coefs,
                    bias=bias,
                    seed=corner_id * 10 + len(channel),
                )

    model = PhysicsModel(
        car="bmw",
        session_ids=("synthetic_coupled",),
        track_models_used={"synthetic_coupled": "synthetic"},
        fitters=fitters,
        ontology={},
        constraints=None,
        untrained_parameters=(),
        aero_correction_available=False,
        baseline_setup=baseline,
        seed=0,
        feature_schema_version=ENV_FEATURE_SCHEMA_VERSION,
    )

    # Use env values inside the synthetic training envelope ([0, 10]) so
    # the GP doesn't collapse to its mean at far-out-of-distribution
    # input points — that would mask the parameter sensitivity.
    env = EnvironmentFrame(
        air_temp_c=5.0, air_density=5.0, air_pressure_mbar=5.0,
        relative_humidity=5.0, wind_vel_ms=5.0, wind_dir_deg=5.0,
        fog_level=5.0, track_temp_c=5.0, track_wetness=5.0,
        weather_declared_wet=False, precip_type=5, skies=5,
    )

    from racingoptimizer.physics.score import score_breakdown

    base_breakdown = score_breakdown(model, baseline, "synthetic", env)
    assert base_breakdown, "synthetic coupled model produced no breakdown"

    # Perturb just the heave spring — the coupled model must propagate.
    perturbed_setup = dict(baseline)
    perturbed_setup["heave_spring_mm"] = baseline["heave_spring_mm"] + 4.0
    perturbed_breakdown = score_breakdown(
        model, perturbed_setup, "synthetic", env
    )

    moved_cells = [
        cpkey for cpkey, base_val in base_breakdown.items()
        if abs(perturbed_breakdown.get(cpkey, base_val) - base_val) > 1e-5
    ]
    assert len(moved_cells) >= 3, (
        f"coupled-architecture failure: perturbing heave_spring_mm moved only "
        f"{len(moved_cells)} (corner, phase) cells; expected >= 3 (chase the "
        f"chain — VISION §3, §5). Moved cells: {moved_cells}"
    )


def test_predict_uses_full_feature_vector() -> None:
    """The Stage-3 predict path feeds the joint setup vector to each fitter.

    Direct test: build a one-channel model whose fitter coefficient on a
    setup parameter is non-zero. Two predict calls with different values
    of that parameter must return different mean values. (Pre-Stage-3 the
    fitter saw only one parameter at a time; Stage-3 sees the joint
    vector, so the parameter must materially affect the prediction.)
    """
    parameters = ["heave_spring_mm", "rear_wing_angle_deg"]
    record = _make_record(
        parameters=parameters,
        coefficients=[0.1, 0.05],
        bias=10.0,
        seed=42,
    )
    fitters: dict[tuple, FitRecord] = {(1, "mid_corner", "accel_lat_g_max"): record}
    model = PhysicsModel(
        car="bmw",
        session_ids=("synthetic_predict",),
        fitters=fitters,
        ontology={},
        constraints=None,
        baseline_setup={"heave_spring_mm": 5.0, "rear_wing_angle_deg": 12.0},
        feature_schema_version=ENV_FEATURE_SCHEMA_VERSION,
    )
    # Env inside the synthetic training envelope ([0, 10]) so the GP
    # doesn't collapse to its mean at far-out-of-distribution inputs.
    env = EnvironmentFrame(
        air_temp_c=5.0, air_density=5.0, air_pressure_mbar=5.0,
        relative_humidity=5.0, wind_vel_ms=5.0, wind_dir_deg=5.0,
        fog_level=5.0, track_temp_c=5.0, track_wetness=5.0,
        weather_declared_wet=False, precip_type=5, skies=5,
    )
    cpkey = CornerPhaseKey(
        session_id="synthetic_predict", lap_index=1,
        corner_id=1, phase=Phase.MID_CORNER,
    )
    setup_a = {"heave_spring_mm": 3.0, "rear_wing_angle_deg": 5.0}
    setup_b = {"heave_spring_mm": 8.0, "rear_wing_angle_deg": 5.0}
    out_a = model.predict(setup_a, env, cpkey)
    out_b = model.predict(setup_b, env, cpkey)
    val_a = out_a.states["accel_lat_g_max"].value
    val_b = out_b.states["accel_lat_g_max"].value
    assert abs(val_a - val_b) > 1e-3, (
        f"predict ignored the setup parameter delta: a={val_a}, b={val_b}. "
        f"Stage-3 fitter must see the joint setup vector."
    )
