from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


def test_unbounded_families_show_in_untrained(tmp_path: Path) -> None:
    """With today's constraints.md (dampers/corner weights <TODO>), those families
    must appear in `untrained_parameters` and `fit()` must NOT raise.
    """
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_IBT))
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)

    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    untrained = set(model.untrained_parameters)
    # Bounded families (ARB / brake bias / diff preload) are now fittable.
    assert "anti_roll_bar_front" not in untrained
    assert "anti_roll_bar_rear" not in untrained
    assert "brake_bias_pct" not in untrained
    assert "diff_preload_nm" not in untrained
    # Remaining TODO-bound families still surface as untrained.
    assert "damper_lsc_fl" in untrained
    assert "damper_hsc_rr" in untrained
    assert "corner_weight_fl_kg" in untrained
