"""`_model_cache_path` invalidates on ontology / schema change.

Pre-fix the cache key only hashed session ids, so an ontology mutation
(e.g. flipping a CE-gated parameter to fittable=True) silently reused
a stale pickle whose `feature_names` and `baseline_setup` reflected the
old parameter set. The fix adds the ontology fingerprint plus the feature
schema version to the digest. This test pins that fingerprinting against
two independent levers.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from racingoptimizer.cli.recommend import _model_cache_path
from racingoptimizer.physics.ontology import ParameterSpec


def test_cache_path_changes_when_ontology_mutates(tmp_path: Path) -> None:
    """Adding a parameter to the ontology invalidates the cache."""
    sids = ["sess_a", "sess_b"]
    base = _model_cache_path(tmp_path, "bmw", "sebring", sids)

    # Simulate the ontology gaining a new bounded parameter (the kind of
    # change the audit's clause 1 just made when ARBs flipped to fittable).
    from racingoptimizer.physics import ontology as ontology_mod
    onto_with_extra = {
        **ontology_mod.BMW,
        "_test_only_param": ParameterSpec(
            json_path=("Chassis", "Test"), dtype=float, units="mm",
            family="ride_height", fittable=True, user_settable=True,
        ),
    }
    with patch.object(ontology_mod, "BMW", onto_with_extra), \
         patch.object(ontology_mod, "_BY_CAR", {**ontology_mod._BY_CAR, "bmw": onto_with_extra}):
        mutated = _model_cache_path(tmp_path, "bmw", "sebring", sids)

    assert base != mutated, (
        "ontology mutation must produce a different cache path; "
        "stale pickles would otherwise leak with the old parameter set"
    )


def test_cache_path_changes_when_feature_schema_version_mutates(tmp_path: Path) -> None:
    """Bumping the feature-schema version invalidates the cache."""
    sids = ["sess_a"]
    base = _model_cache_path(tmp_path, "bmw", "sebring", sids)

    from racingoptimizer.physics import fitter as fitter_mod
    with patch.object(fitter_mod, "ENV_FEATURE_SCHEMA_VERSION", 99):
        bumped = _model_cache_path(tmp_path, "bmw", "sebring", sids)

    assert base != bumped


def test_cache_path_stable_for_identical_inputs(tmp_path: Path) -> None:
    """Same (car, track, session_ids, ontology, schema) -> same path."""
    sids = ["sess_a", "sess_b"]
    a = _model_cache_path(tmp_path, "bmw", "sebring", sids)
    b = _model_cache_path(tmp_path, "bmw", "sebring", sids)
    assert a == b


def test_cache_path_session_id_order_independent(tmp_path: Path) -> None:
    """Session id order should not change the cache key."""
    a = _model_cache_path(tmp_path, "bmw", "sebring", ["a", "b", "c"])
    b = _model_cache_path(tmp_path, "bmw", "sebring", ["c", "b", "a"])
    assert a == b


def test_cache_path_per_car_per_track_isolated(tmp_path: Path) -> None:
    sids = ["s1"]
    bmw_seb = _model_cache_path(tmp_path, "bmw", "sebring", sids)
    bmw_lag = _model_cache_path(tmp_path, "bmw", "lagunaseca", sids)
    acura_seb = _model_cache_path(tmp_path, "acura", "sebring", sids)
    assert bmw_seb != bmw_lag
    assert bmw_seb != acura_seb
