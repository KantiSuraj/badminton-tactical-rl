"""Schemas, constants, and quality flags for the ShuttleSet event pipeline.

One event per stroke. Output is a flat Parquet table; provenance is recorded
per-row rather than in nested structs so that it survives every downstream join
and filtering operation.
"""
from __future__ import annotations

from typing import Final

# Provenance axes (each evolves independently; never conflate in manifests)
LOADER_VERSION: Final = "load_shuttleset_v1"
COURT_TRANSFORM_VERSION: Final = "court_coordinates_v1"
SHOT_MAPPING_VERSION: Final = "shot_label_mapping_v1"
EVENT_SCHEMA_VERSION: Final = "events_v1"
SPLIT_SCHEME_VERSION: Final = "player_match_v1"

# ShuttleSet provenance. Full SHA must be filled in locally (see build_events manifest).
SHUTTLESET_REPO: Final = "wywyWang/CoachAI-Projects"
SHUTTLESET_SHORT_COMMIT: Final = "45517f7"
SHUTTLESET_FULL_COMMIT: Final = "VERIFY_LOCALLY"

# Canonical shot families (ordered; ids are the action-space axis-1)
SHOT_FAMILIES: Final = (
    "CLEAR", "DROP", "SMASH", "HALF_SMASH", "DRIVE",
    "PUSH", "LIFT", "NET_SHOT", "BLOCK", "CUT",
)
FAMILY_TO_ID: Final = {name: i for i, name in enumerate(SHOT_FAMILIES)}

# Quality-flag vocabulary. Stable strings; changes are a schema version bump.
QUALITY_FLAGS: Final = frozenset({
    # data quality, per stroke
    "non_increasing_frame_num",
    "unknown_shot_label",
    "unmapped_shot_label",
    "missing_hit_xy",
    "missing_landing_xy",
    "missing_player_xy",
    "missing_opponent_xy",
    "missing_hit_height",
    "missing_landing_height",
    "missing_landing_area",
    "invalid_landing_area",
    "same_player_twice_in_a_row",
    "fps_outlier",
    "invalid_player_role",
    "missing_player_identity",
    # data quality, per rally (broadcast to every stroke in the rally)
    "rally_winner_missing",
    "rally_winner_inconsistent",
    "rally_winner_invalid",
})

# Canonical column order. build_events emits exactly these columns; tests assert.
EVENT_COLUMNS: Final = (
    # identity
    "event_id", "match_id", "set_id", "raw_rally_id", "rally_id", "stroke_index",
    # player
    "player_role", "hitter_id", "receiver_id",
    # taxonomy
    "source_shot_label", "shot_family", "mapping_status", "mapping_version",
    # source annotations (verbatim; nothing derived is mixed in here)
    "hit_height", "landing_height", "hit_area", "landing_area", "opponent_area",
    "ball_round", "original_time", "frame_num", "db_raw",
    "lose_reason", "getpoint_player_raw",
    # timing
    "timestamp_s", "fps",
    # geometry (pair-or-null invariant; see test_data_events.test_xy_pair_invariant)
    "hit_x_m", "hit_y_m",
    "landing_x_m", "landing_y_m",
    "player_x_m", "player_y_m",
    "opponent_x_m", "opponent_y_m",
    # derived outcome (rally-level, repeated per row; consistent within rally)
    "rally_winner_role", "rally_winner_id", "rally_winner_status",
    # transform provenance
    "homography_id", "court_transform_id", "coordinate_units", "shuttle_xy_projection",
    # source provenance
    "source_file", "source_commit", "loader_version",
    # quality / vision
    "quality_flags", "vision_status",
)