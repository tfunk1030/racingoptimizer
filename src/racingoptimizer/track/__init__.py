"""Track model — bin-indexed compounding aggregates per track."""
from racingoptimizer.track.anomaly import flag_anomalies_from_cache
from racingoptimizer.track.bins import (
    DEFAULT_BIN_SIZE_M,
    bin_index,
    track_pos_m_from_pct,
)
from racingoptimizer.track.builder import TrackModel, build_track_model
from racingoptimizer.track.corner_loading import classify_corner_loading
from racingoptimizer.track.corners import compute_corner_landmarks
from racingoptimizer.track.geometry import (
    GEOMETRY_SCHEMA,
    compute_track_geometry,
    empty_geometry_frame,
)
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
from racingoptimizer.track.predict import (
    PREDICTABLE_CHANNELS,
    Expected,
    expected_from_cache,
)
from racingoptimizer.track.rewrite import ApplyMaskResult, apply_quality_mask

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
    "ApplyMaskResult",
    "Expected",
    "GEOMETRY_SCHEMA",
    "PREDICTABLE_CHANNELS",
    "TrackModel",
    "aggregate_curb_likelihood",
    "aggregate_grip_map",
    "apply_quality_mask",
    "bin_index",
    "build_track_model",
    "cache_path",
    "classify_corner_loading",
    "compute_corner_landmarks",
    "compute_curb_mask",
    "compute_off_track_mask",
    "compute_session_shock_v_p99_per_bin",
    "compute_track_geometry",
    "empty_geometry_frame",
    "expected_from_cache",
    "flag_anomalies_from_cache",
    "latest_pointer_path",
    "sessions_hash",
    "summary_path",
    "track_models_root",
    "track_pos_m_from_pct",
]
