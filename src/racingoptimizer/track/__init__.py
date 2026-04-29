"""Track model — bin-indexed compounding aggregates per track."""
from racingoptimizer.track.bins import (
    DEFAULT_BIN_SIZE_M,
    bin_index,
    track_pos_m_from_pct,
)
from racingoptimizer.track.builder import TrackModel, build_track_model
from racingoptimizer.track.masks import (
    BUMP_RANGE_MAX_MM_S,
    BUMP_RANGE_MIN_MM_S,
    CURB_AGREEMENT_FRACTION,
    OFFTRACK_GRIP_HISTORY_MS,
    OFFTRACK_GRIP_LOSS_RATIO,
    OFFTRACK_MASK_WINDOW_S,
    OFFTRACK_WHEELSPEED_RATIO,
    T_CURB_AGGREGATE_MM_S,
    T_CURB_SESSION_MM_S,
    aggregate_curb_likelihood,
    aggregate_grip_map,
    compute_curb_mask,
    compute_off_track_mask,
    compute_session_shock_v_p99_per_bin,
)
from racingoptimizer.track.paths import (
    cache_path,
    latest_pointer_path,
    sessions_hash,
    summary_path,
    track_models_root,
)

__all__ = [
    "BUMP_RANGE_MAX_MM_S",
    "BUMP_RANGE_MIN_MM_S",
    "CURB_AGREEMENT_FRACTION",
    "DEFAULT_BIN_SIZE_M",
    "OFFTRACK_GRIP_HISTORY_MS",
    "OFFTRACK_GRIP_LOSS_RATIO",
    "OFFTRACK_MASK_WINDOW_S",
    "OFFTRACK_WHEELSPEED_RATIO",
    "T_CURB_AGGREGATE_MM_S",
    "T_CURB_SESSION_MM_S",
    "TrackModel",
    "aggregate_curb_likelihood",
    "aggregate_grip_map",
    "bin_index",
    "build_track_model",
    "cache_path",
    "compute_curb_mask",
    "compute_off_track_mask",
    "compute_session_shock_v_p99_per_bin",
    "latest_pointer_path",
    "sessions_hash",
    "summary_path",
    "track_models_root",
    "track_pos_m_from_pct",
]
