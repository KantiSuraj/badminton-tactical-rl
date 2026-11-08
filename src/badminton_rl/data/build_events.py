"""Build the canonical ShuttleSet per-stroke event table (Step 1).

The builder is intentionally conservative:
- one output row is emitted for every row returned by ``load_strokes``;
- source annotations are retained separately from normalized/derived fields;
- invalid or ambiguous annotations are flagged, not guessed or silently fixed;
- identity, mapping, geometry, and provenance invariants are checked explicitly.

The output column order is defined by ``schemas.EVENT_COLUMNS``.  Nullable
integer/float dtypes are applied at the end so that Parquet has a stable schema.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from .load_shuttleset import DEFAULT_ROOT, load_matches, load_strokes
from .schemas import (
    COURT_TRANSFORM_VERSION,
    EVENT_COLUMNS,
    EVENT_SCHEMA_VERSION,
    LOADER_VERSION,
    QUALITY_FLAGS,
    SHOT_FAMILIES,
    SHOT_MAPPING_VERSION,
    SHUTTLESET_FULL_COMMIT,
    SHUTTLESET_REPO,
    SHUTTLESET_SHORT_COMMIT,
)

_HERE = Path(__file__).resolve().parent
_MAPPING_YAML = _HERE / "shot_label_mapping_v1.yaml"

# Nullable pandas extension dtypes.  These limits are validated before casting.
_INTEGER_DTYPES: dict[str, str] = {
    "set_id": "Int8",
    "raw_rally_id": "Int16",
    "stroke_index": "Int16",
    "ball_round": "Int16",
    "frame_num": "Int32",
}
_FLOAT_COLUMNS = (
    "hit_height",
    "landing_height",
    "timestamp_s",
    "fps",
    "hit_x_m",
    "hit_y_m",
    "landing_x_m",
    "landing_y_m",
    "player_x_m",
    "player_y_m",
    "opponent_x_m",
    "opponent_y_m",
)
_FLOAT_DTYPE = "Float32"

_REQUIRED_STROKE_COLUMNS = (
    "match_id",
    "video",
    "source_file",
    "set",
    "rally",
    "player",
    "type_en",
    "frame_num",
    "time",
    "t_s",
    "fps",
    "hit_x_m",
    "hit_y_m",
    "landing_x_m",
    "landing_y_m",
    "player_x_m",
    "player_y_m",
    "opponent_x_m",
    "opponent_y_m",
)
_REQUIRED_MATCH_COLUMNS = ("id", "winner", "loser")


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    frame_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _normalise_nonempty_string(series: pd.Series, name: str) -> pd.Series:
    """Convert identifiers to nullable strings after rejecting null/blank values."""
    out = series.astype("string")
    blank = out.isna() | out.str.strip().eq("")
    if blank.any():
        rows = blank[blank].index[:10].tolist()
        raise ValueError(f"{name} contains null/blank values at row indices {rows}")
    return out


def _strict_numeric(
    series: pd.Series,
    name: str,
    *,
    allow_missing: bool = True,
) -> pd.Series:
    """Parse numeric data without converting malformed nonblank strings to null."""
    numeric = pd.to_numeric(series, errors="coerce")
    source_text = series.astype("string").str.strip()
    nonblank_source = series.notna() & source_text.ne("")
    malformed = nonblank_source & numeric.isna()
    if malformed.any():
        examples = series.loc[malformed].astype(str).head(5).tolist()
        raise ValueError(f"{name} contains non-numeric values: {examples}")
    if not allow_missing and numeric.isna().any():
        rows = numeric[numeric.isna()].index[:10].tolist()
        raise ValueError(f"{name} contains missing values at row indices {rows}")
    return numeric


def _strict_integer(
    series: pd.Series,
    name: str,
    *,
    dtype: str,
    allow_missing: bool,
) -> pd.Series:
    """Validate integer-valued input and dtype bounds, then cast nullable."""
    numeric = _strict_numeric(series, name, allow_missing=allow_missing)
    present = numeric.dropna()
    if not np.isfinite(present.to_numpy(dtype="float64")).all():
        raise ValueError(f"{name} contains non-finite values")
    fractional = present.mod(1).ne(0)
    if fractional.any():
        examples = present.loc[fractional].head(5).tolist()
        raise ValueError(f"{name} must contain integer values; found {examples}")

    bounds = {
        "Int8": (-2**7, 2**7 - 1),
        "Int16": (-2**15, 2**15 - 1),
        "Int32": (-2**31, 2**31 - 1),
    }
    lower, upper = bounds[dtype]
    outside = present.lt(lower) | present.gt(upper)
    if outside.any():
        examples = present.loc[outside].head(5).tolist()
        raise ValueError(f"{name} exceeds {dtype} bounds: {examples}")
    return numeric.astype(dtype)


def _strict_float32(series: pd.Series, name: str) -> pd.Series:
    """Parse finite numeric values and cast to nullable Float32."""
    numeric = _strict_numeric(series, name, allow_missing=True)
    present = numeric.dropna().to_numpy(dtype="float64")
    if not np.isfinite(present).all():
        raise ValueError(f"{name} contains +/-inf or other non-finite values")
    out = numeric.astype(_FLOAT_DTYPE)
    casted = out.dropna().to_numpy(dtype="float64")
    if not np.isfinite(casted).all():
        raise ValueError(f"{name} overflows {_FLOAT_DTYPE}")
    return out


def _optional_source_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a source column unchanged, or an aligned all-null object Series."""
    if column in frame.columns:
        return frame[column].copy()
    return pd.Series(pd.NA, index=frame.index, dtype="object", name=column)


def _load_mapping() -> tuple[dict[str, Optional[str]], dict[str, str], str]:
    """Load and validate the versioned English-label -> family mapping.

    Returns family map, status map, and the SHA-256 of the exact YAML bytes.
    Duplicate labels and inconsistent mapped/unmapped entries are fatal.
    """
    if not _MAPPING_YAML.is_file():
        raise FileNotFoundError(f"Shot mapping YAML not found: {_MAPPING_YAML}")

    raw_bytes = _MAPPING_YAML.read_bytes()
    spec = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("raw_labels"), list):
        raise ValueError("Shot mapping YAML must contain a 'raw_labels' list")

    family: dict[str, Optional[str]] = {}
    status: dict[str, str] = {}
    valid_families = set(SHOT_FAMILIES)

    for index, entry in enumerate(spec["raw_labels"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Mapping entry {index} must be a mapping/object")
        english = entry.get("english")
        mapping_status = entry.get("mapping_status")
        family_name = entry.get("family")

        if not isinstance(english, str) or not english.strip():
            raise ValueError(f"Mapping entry {index} has an invalid English label")
        if english in family:
            raise ValueError(f"Duplicate English shot label in mapping: {english!r}")
        if mapping_status not in {"mapped", "unmapped"}:
            raise ValueError(
                f"Invalid mapping_status for {english!r}: {mapping_status!r}"
            )

        if mapping_status == "mapped":
            if family_name not in valid_families:
                raise ValueError(
                    f"Mapped label {english!r} has invalid family {family_name!r}; "
                    f"expected one of {sorted(valid_families)}"
                )
        elif family_name is not None:
            raise ValueError(
                f"Unmapped label {english!r} must have family: null, "
                f"not {family_name!r}"
            )

        family[english] = family_name
        status[english] = mapping_status

    if not family:
        raise ValueError("Shot mapping YAML contains no raw label entries")

    return family, status, hashlib.sha256(raw_bytes).hexdigest()


def _pair_or_null(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (x, y), nulling both coordinates whenever either is missing."""
    both_present = x.notna() & y.notna()
    return x.where(both_present), y.where(both_present)


def _validate_match_metadata(matches: pd.DataFrame) -> pd.DataFrame:
    """Validate match metadata and return a copy with canonical string IDs."""
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("load_matches() must return a pandas DataFrame")
    _require_columns(matches, _REQUIRED_MATCH_COLUMNS, "match metadata")
    out = matches.copy().reset_index(drop=True)
    out["id"] = _normalise_nonempty_string(out["id"], "matches.id")
    out["winner"] = _normalise_nonempty_string(out["winner"], "matches.winner")
    out["loser"] = _normalise_nonempty_string(out["loser"], "matches.loser")
    if "video" in out.columns:
        out["video"] = _normalise_nonempty_string(out["video"], "matches.video")
        duplicate_videos = out["video"].duplicated(keep=False)
        if duplicate_videos.any():
            values = out.loc[duplicate_videos, "video"].unique().tolist()[:10]
            raise ValueError(f"Duplicate video IDs in match metadata: {values}")

    duplicates = out["id"].duplicated(keep=False)
    if duplicates.any():
        values = out.loc[duplicates, "id"].unique().tolist()[:10]
        raise ValueError(f"Duplicate match IDs in match metadata: {values}")

    same_participant = out["winner"].eq(out["loser"])
    if same_participant.any():
        values = out.loc[same_participant, "id"].tolist()[:10]
        raise ValueError(f"Winner and loser are identical for matches: {values}")
    return out


def _prepare_strokes(strokes: pd.DataFrame) -> pd.DataFrame:
    """Validate core loader output and normalize only identity/numeric fields."""
    if not isinstance(strokes, pd.DataFrame):
        raise TypeError("load_strokes() must return a pandas DataFrame")
    _require_columns(strokes, _REQUIRED_STROKE_COLUMNS, "load_strokes() output")
    out = strokes.copy().reset_index(drop=True)

    # Validate before string conversion: astype(str) would turn NaN into 'nan'.
    out["match_id"] = _normalise_nonempty_string(out["match_id"], "strokes.match_id")
    out["video"] = _normalise_nonempty_string(out["video"], "strokes.video")
    out["source_file"] = _normalise_nonempty_string(
        out["source_file"], "strokes.source_file"
    )

    if out["match_id"].str.contains("|", regex=False).any():
        raise ValueError("match_id contains reserved '|' delimiter used in event IDs")

    out["set_id"] = _strict_integer(
        out["set"], "strokes.set", dtype="Int8", allow_missing=False
    )
    out["raw_rally_id"] = _strict_integer(
        out["rally"], "strokes.rally", dtype="Int16", allow_missing=False
    )
    if out["set_id"].le(0).any():
        raise ValueError("strokes.set must contain positive set numbers")
    if out["raw_rally_id"].le(0).any():
        raise ValueError("strokes.rally must contain positive rally numbers")

    # Role values are retained in the loader frame; only the normalized
    # player_role column later restricts them to A/B.
    out["player"] = out["player"].astype("string")

    # Verify all transformed numeric fields now, before quality checks.
    numeric_columns = (
        "t_s", "fps", "hit_x_m", "hit_y_m", "landing_x_m", "landing_y_m",
        "player_x_m", "player_y_m", "opponent_x_m", "opponent_y_m",
    )
    for column in numeric_columns:
        out[column] = _strict_numeric(out[column], f"strokes.{column}", allow_missing=True)
        values = out[column].dropna().to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"strokes.{column} contains non-finite values")

    # frame_num may be null on an individual source row, but must be integral.
    out["frame_num"] = _strict_integer(
        out["frame_num"], "strokes.frame_num", dtype="Int32", allow_missing=True
    )
    if out["frame_num"].dropna().lt(0).any():
        raise ValueError("strokes.frame_num cannot be negative")

    # Preserve original text, while using numeric seconds for temporal features.
    out["original_time"] = out["time"].astype("string")
    out["timestamp_s"] = out["t_s"]
    return out


def _resolve_hitter_receiver(
    strokes: pd.DataFrame,
    winner_map: dict[str, str],
    loser_map: dict[str, str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Resolve role and participant IDs vectorially, with no row-wise apply."""
    role = strokes["player"].where(strokes["player"].isin(["A", "B"]), other=pd.NA)
    winner_for_match = strokes["match_id"].map(winner_map)
    loser_for_match = strokes["match_id"].map(loser_map)

    hitter = winner_for_match.where(role.eq("A"), loser_for_match.where(role.eq("B")))
    receiver = loser_for_match.where(role.eq("A"), winner_for_match.where(role.eq("B")))
    return role.astype("string"), hitter, receiver


def _normalise_winner_annotation(value: Any) -> Optional[str]:
    """Normalize getpoint_player only for validation; raw annotation stays intact."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _derive_rally_outcomes(
    strokes: pd.DataFrame,
    winner_map: dict[str, str],
    loser_map: dict[str, str],
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, str]]:
    """Validate getpoint_player consistency per rally and broadcast the result."""
    role_by_rally: dict[str, Optional[str]] = {}
    id_by_rally: dict[str, Optional[str]] = {}
    status_by_rally: dict[str, str] = {}
    flag_by_rally: dict[str, str] = {}

    for rally_id, group in strokes.groupby("rally_id", sort=False, dropna=False):
        normalized_values = {
            candidate
            for candidate in group["getpoint_player_raw"].map(
                _normalise_winner_annotation
            ).tolist()
            if candidate is not None
        }

        role: Optional[str] = None
        participant_id: Optional[str] = None
        flag: Optional[str] = None

        if not normalized_values:
            status, flag = "missing", "rally_winner_missing"
        elif len(normalized_values) > 1:
            status, flag = "inconsistent", "rally_winner_inconsistent"
        else:
            candidate = next(iter(normalized_values))
            if candidate not in {"A", "B"}:
                status, flag = "invalid", "rally_winner_invalid"
            else:
                match_id = str(group["match_id"].iloc[0])
                participant_id = (
                    winner_map.get(match_id) if candidate == "A" else loser_map.get(match_id)
                )
                if participant_id is None:
                    status, flag = "invalid", "rally_winner_invalid"
                else:
                    role = candidate
                    status = "validated"

        role_by_rally[str(rally_id)] = role
        id_by_rally[str(rally_id)] = participant_id
        status_by_rally[str(rally_id)] = status
        if flag:
            flag_by_rally[str(rally_id)] = flag

    rally_keys = strokes["rally_id"].astype("string")
    return (
        rally_keys.map(role_by_rally).astype("string"),
        rally_keys.map(id_by_rally),
        rally_keys.map(status_by_rally).astype("string"),
        flag_by_rally,
    )


def _quality_flags_rowwise(
    strokes: pd.DataFrame,
    mapping_status: pd.Series,
    rally_flag_by_id: dict[str, str],
) -> pd.Series:
    """Return sorted, semicolon-delimited flags; rally flags are broadcast."""
    flags: list[set[str]] = [set() for _ in range(len(strokes))]

    def add(mask: pd.Series, flag: str) -> None:
        if flag not in QUALITY_FLAGS:
            raise ValueError(f"Quality flag {flag!r} is absent from QUALITY_FLAGS")
        safe_mask = mask.fillna(False).to_numpy(dtype=bool)
        for position in np.flatnonzero(safe_mask):
            flags[position].add(flag)

    add(~strokes["player_role"].isin(["A", "B"]), "invalid_player_role")
    add(
        strokes["hitter_id"].isna() | strokes["receiver_id"].isna(),
        "missing_player_identity",
    )
    add(strokes["source_shot_label"].eq("unknown"), "unknown_shot_label")
    add(mapping_status.eq("unmapped"), "unmapped_shot_label")

    coordinate_pairs = {
        "hit": ("hit_x_m", "hit_y_m"),
        "landing": ("landing_x_m", "landing_y_m"),
        "player": ("player_x_m", "player_y_m"),
        "opponent": ("opponent_x_m", "opponent_y_m"),
    }
    flag_names = {
        "hit": "missing_hit_xy",
        "landing": "missing_landing_xy",
        "player": "missing_player_xy",
        "opponent": "missing_opponent_xy",
    }
    for name, (x_col, y_col) in coordinate_pairs.items():
        add(strokes[x_col].isna() | strokes[y_col].isna(), flag_names[name])

    add(strokes["hit_height"].isna(), "missing_hit_height")
    add(strokes["landing_height"].isna(), "missing_landing_height")
    add(strokes["landing_area"].isna(), "missing_landing_area")
    add(
        strokes["fps"].notna() & ~strokes["fps"].between(24.0, 31.0),
        "fps_outlier",
    )

    # Compare only adjacent strokes in source order within each rally.
    grouped = strokes.groupby("rally_id", sort=False, dropna=False)
    previous_frame = grouped["frame_num"].shift(1)
    current_frame = strokes["frame_num"]
    add(
        previous_frame.notna() & current_frame.notna() & current_frame.le(previous_frame),
        "non_increasing_frame_num",
    )

    previous_player = grouped["player_role"].shift(1)
    same_player = (
        strokes["player_role"].notna()
        & previous_player.notna()
        & strokes["player_role"].eq(previous_player)
    )
    add(same_player, "same_player_twice_in_a_row")

    # Rally-level annotations are repeated on every stroke in that rally.
    for position, rally_id in enumerate(strokes["rally_id"].astype(str).tolist()):
        flag = rally_flag_by_id.get(rally_id)
        if flag:
            if flag not in QUALITY_FLAGS:
                raise ValueError(f"Quality flag {flag!r} is absent from QUALITY_FLAGS")
            flags[position].add(flag)

    unexpected = set().union(*flags) - set(QUALITY_FLAGS) if flags else set()
    if unexpected:
        raise ValueError(f"Quality flags not in vocabulary: {sorted(unexpected)}")

    values = [";".join(sorted(row_flags)) if row_flags else pd.NA for row_flags in flags]
    return pd.Series(values, index=strokes.index, dtype="string", name="quality_flags")


def _apply_event_dtypes(events: pd.DataFrame) -> pd.DataFrame:
    """Apply the stable nullable integer/float contract to the event table."""
    for column, dtype in _INTEGER_DTYPES.items():
        events[column] = _strict_integer(
            events[column], f"events.{column}", dtype=dtype, allow_missing=True
        )
    for column in _FLOAT_COLUMNS:
        events[column] = _strict_float32(events[column], f"events.{column}")

    # Text fields use pandas' nullable string dtype; raw non-text `db_raw` is
    # deliberately not coerced because its source representation is preserved.
    string_columns = (
        "event_id", "match_id", "rally_id", "player_role", "hitter_id", "receiver_id",
        "source_shot_label", "shot_family", "mapping_status", "mapping_version",
        "hit_area", "landing_area", "opponent_area", "original_time", "lose_reason",
        "getpoint_player_raw", "rally_winner_role", "rally_winner_id",
        "rally_winner_status", "homography_id", "court_transform_id",
        "coordinate_units", "shuttle_xy_projection", "source_file", "source_commit",
        "loader_version", "quality_flags", "vision_status",
    )
    for column in string_columns:
        events[column] = events[column].astype("string")
    return events


def _resolve_source_commit() -> tuple[str, Optional[str], bool]:
    """Return (row value, declared full SHA or None, full-SHA format validity).

    A placeholder in schemas.py is never represented as a real commit.  This
    validates the SHA's format and prefix only; it does not prove that the
    commit exists in the source clone.  That verification must be performed
    locally before the dataset is frozen.
    """
    full = str(SHUTTLESET_FULL_COMMIT).strip()
    short = str(SHUTTLESET_SHORT_COMMIT).strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", full):
        full = full.lower()
        if short and not full.startswith(short.lower()):
            raise ValueError(
                "SHUTTLESET_FULL_COMMIT does not start with "
                "SHUTTLESET_SHORT_COMMIT; check source provenance constants"
            )
        return full, full, True

    row_value = f"UNVERIFIED:{short}" if short else "UNVERIFIED"
    return row_value, None, False


def _detect_build_git_commit(start: Path = _HERE) -> Optional[str]:
    """Find the current project's HEAD SHA without requiring Git to be present."""
    repo_dir: Optional[Path] = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            repo_dir = candidate
            break
    if repo_dir is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def build_events(
    root: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the canonical event table; optionally write Parquet and manifest.

    ``root`` is the ShuttleSet root containing ``match.csv``, ``homography.csv``,
    and per-video set directories.  The function preserves loader row order.
    """
    data_root = Path(root) if root is not None else Path(DEFAULT_ROOT)
    raw_strokes = load_strokes(data_root)
    raw_matches = load_matches(data_root)

    if len(raw_strokes) == 0:
        raise ValueError("load_strokes() returned zero rows; refusing to build empty events")

    strokes = _prepare_strokes(raw_strokes)
    matches = _validate_match_metadata(raw_matches)

    match_ids_in_metadata = set(matches["id"].tolist())
    match_ids_in_strokes = set(strokes["match_id"].tolist())
    unknown_match_ids = sorted(match_ids_in_strokes - match_ids_in_metadata)
    if unknown_match_ids:
        raise ValueError(
            "Stroke rows reference match IDs absent from match metadata: "
            f"{unknown_match_ids[:20]}"
        )
    if "video" in matches.columns:
        video_by_match = dict(zip(matches["id"], matches["video"], strict=True))
        expected_video = strokes["match_id"].map(video_by_match)
        mismatch = expected_video.ne(strokes["video"])
        if mismatch.any():
            examples = strokes.loc[mismatch, ["match_id", "video"]].head(10).to_dict("records")
            raise ValueError(
                "Stroke video IDs disagree with match metadata; "
                f"examples: {examples}"
            )

    winner_map = dict(zip(matches["id"], matches["winner"], strict=True))
    loser_map = dict(zip(matches["id"], matches["loser"], strict=True))

    # IDs are derived from validated source keys. No sorting or row filtering.
    strokes["stroke_index"] = strokes.groupby(
        ["match_id", "set_id", "raw_rally_id"], sort=False, dropna=False
    ).cumcount()
    strokes["rally_id"] = (
        strokes["match_id"]
        + "|s"
        + strokes["set_id"].astype("string")
        + "|r"
        + strokes["raw_rally_id"].astype("string")
    )
    strokes["event_id"] = strokes["rally_id"] + "|i" + strokes["stroke_index"].map(
        lambda index: f"{index:03d}"
    ).astype("string")

    if strokes["rally_id"].isna().any() or strokes["event_id"].isna().any():
        raise ValueError("Generated rally_id/event_id contains nulls")
    if not strokes["event_id"].is_unique:
        duplicates = strokes.loc[strokes["event_id"].duplicated(False), "event_id"]
        raise ValueError(f"event_id collision; examples: {duplicates.head(10).tolist()}")

    # Player identities: ShuttleSet role A is the match winner, B the loser.
    role, hitter, receiver = _resolve_hitter_receiver(strokes, winner_map, loser_map)
    strokes["player_role"] = role
    strokes["hitter_id"] = hitter
    strokes["receiver_id"] = receiver

    # Shot taxonomy: labels missing from the versioned mapping are a hard error.
    family_of, status_of, mapping_sha256 = _load_mapping()
    strokes["source_shot_label"] = strokes["type_en"].astype("string")
    strokes["shot_family"] = strokes["source_shot_label"].map(family_of).astype("string")
    mapping_status = strokes["source_shot_label"].map(status_of).astype("string")
    missing_mapping = mapping_status.isna()
    if missing_mapping.any():
        labels = strokes.loc[missing_mapping, "source_shot_label"].drop_duplicates().tolist()
        raise ValueError(f"Shot labels absent from {SHOT_MAPPING_VERSION}: {labels}")

    invalid_mapping_pairs = (
        mapping_status.eq("mapped") & strokes["shot_family"].isna()
    ) | (
        mapping_status.eq("unmapped") & strokes["shot_family"].notna()
    )
    if invalid_mapping_pairs.any():
        raise ValueError("Shot mapping family/status relationship is inconsistent")

    strokes["mapping_status"] = mapping_status
    strokes["mapping_version"] = SHOT_MAPPING_VERSION

    # Copy source annotations verbatim into explicit raw/canonical fields.
    strokes["db_raw"] = _optional_source_column(strokes, "db")
    strokes["getpoint_player_raw"] = _optional_source_column(strokes, "getpoint_player")
    strokes["lose_reason"] = _optional_source_column(strokes, "lose_reason")
    strokes["hit_area"] = _optional_source_column(strokes, "player_location_area")
    strokes["landing_area"] = _optional_source_column(strokes, "landing_area")
    strokes["opponent_area"] = _optional_source_column(strokes, "opponent_location_area")

    # Optional annotation columns used above may be absent from older extracts.
    for column in ("hit_height", "landing_height", "ball_round"):
        if column not in strokes.columns:
            strokes[column] = pd.NA
    strokes["ball_round"] = _strict_integer(
        strokes["ball_round"], "strokes.ball_round", dtype="Int16", allow_missing=True
    )

    # Geometry. Reject infinities first, then enforce pair-or-null.
    geometry_columns = (
        "hit_x_m", "hit_y_m", "landing_x_m", "landing_y_m",
        "player_x_m", "player_y_m", "opponent_x_m", "opponent_y_m",
    )
    for column in geometry_columns:
        values = strokes[column].dropna().to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite coordinates")
    for name in ("hit", "landing", "player", "opponent"):
        x_column, y_column = f"{name}_x_m", f"{name}_y_m"
        strokes[x_column], strokes[y_column] = _pair_or_null(
            strokes[x_column], strokes[y_column]
        )

    # Transform and source provenance.
    strokes["homography_id"] = strokes["video"]
    strokes["court_transform_id"] = COURT_TRANSFORM_VERSION
    strokes["coordinate_units"] = "metres"
    strokes["shuttle_xy_projection"] = "floor_plane_homography"
    source_commit_value, declared_source_sha, source_commit_format_valid = _resolve_source_commit()
    strokes["source_commit"] = source_commit_value
    strokes["loader_version"] = LOADER_VERSION
    strokes["vision_status"] = "not_attempted"

    # Timing: original_time remains source text; timestamp_s is loader-derived.
    strokes["timestamp_s"] = strokes["t_s"]
    strokes["fps"] = _strict_numeric(strokes["fps"], "strokes.fps", allow_missing=True)

    # Rally outcome: only getpoint_player is used; lose_reason is not interpreted.
    rally_role, rally_id, rally_status, rally_flags = _derive_rally_outcomes(
        strokes, winner_map, loser_map
    )
    strokes["rally_winner_role"] = rally_role
    strokes["rally_winner_id"] = rally_id
    strokes["rally_winner_status"] = rally_status

    # Quality flags preserve anomalies instead of changing the source observations.
    strokes["quality_flags"] = _quality_flags_rowwise(
        strokes, mapping_status, rally_flags
    )

    missing_columns = [column for column in EVENT_COLUMNS if column not in strokes.columns]
    if missing_columns:
        raise ValueError(f"Event builder failed to populate required columns: {missing_columns}")
    events = strokes.loc[:, list(EVENT_COLUMNS)].copy()
    events = _apply_event_dtypes(events)

    # Explicit one-row-per-source-stroke invariant.
    if len(events) != len(raw_strokes):
        raise RuntimeError(
            f"Row-count preservation violated: {len(raw_strokes)} source strokes, "
            f"{len(events)} events"
        )

    if output_path is not None:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "events.parquet"
        manifest_path = output_dir / "events_manifest.yaml"

        # Write temp files and replace only after successful serialization.
        temporary_parquet = output_dir / "events.parquet.tmp"
        temporary_manifest = output_dir / "events_manifest.yaml.tmp"
        try:
            events.to_parquet(temporary_parquet, index=False)
            _write_manifest(
                events=events,
                matches=matches,
                source_stroke_count=len(raw_strokes),
                source_root=data_root,
                mapping_sha256=mapping_sha256,
                source_commit_value=source_commit_value,
                declared_source_sha=declared_source_sha,
                source_commit_format_valid=source_commit_format_valid,
                build_git_commit=_detect_build_git_commit(),
                path=temporary_manifest,
            )
            temporary_parquet.replace(parquet_path)
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_parquet.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    return events


def _write_manifest(
    *,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    source_stroke_count: int,
    source_root: Path,
    mapping_sha256: str,
    source_commit_value: str,
    declared_source_sha: Optional[str],
    source_commit_format_valid: bool,
    build_git_commit: Optional[str],
    path: Path,
) -> None:
    """Write a reproducibility manifest; counts distinguish source vs output."""
    family_counts = {
        str(label): int(count)
        for label, count in events["shot_family"].value_counts(dropna=False).items()
    }

    unmapped_counts_series = events.loc[
        events["mapping_status"].eq("unmapped"), "source_shot_label"
    ].value_counts(dropna=False)
    unmapped_counts = {
        str(label): int(count) for label, count in unmapped_counts_series.items()
    }

    quality_flag_counts: dict[str, int] = {}
    for value in events["quality_flags"].dropna():
        for flag in str(value).split(";"):
            if flag:
                quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1

    missing_value_counts = {
        column: int(events[column].isna().sum())
        for column in events.columns
        if events[column].isna().any()
    }
    rally_winner_status_counts = {
        str(status): int(count)
        for status, count in events["rally_winner_status"].value_counts(dropna=False).items()
    }

    matches_in_events = int(events["match_id"].nunique())
    manifest = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "provenance": {
            "repository": SHUTTLESET_REPO,
            "short_commit": SHUTTLESET_SHORT_COMMIT,
            "commit": declared_source_sha,
            "full_sha_format_valid": bool(source_commit_format_valid),
            "source_commit_status": (
                "full_sha_supplied_format_valid_but_not_repository_verified"
                if source_commit_format_valid
                else "short_sha_only_unverified"
            ),
            "event_source_commit_value": source_commit_value,
            "loader_version": LOADER_VERSION,
            "court_transform_id": COURT_TRANSFORM_VERSION,
            "mapping_version": SHOT_MAPPING_VERSION,
            "mapping_yaml_sha256": mapping_sha256,
            "build_code_commit": build_git_commit,
            "source_root": str(source_root.resolve()),
        },
        "row_counts": {
            "source_strokes": int(source_stroke_count),
            "events": int(len(events)),
            "matches_in_source_metadata": int(len(matches)),
            "matches_represented_in_events": matches_in_events,
            "sets_in_events": int(events.groupby(["match_id", "set_id"], dropna=False).ngroups),
            "rallies_in_events": int(events["rally_id"].nunique()),
        },
        "shot_family_counts_including_null": family_counts,
        "unmapped_label_counts": unmapped_counts,
        "missing_value_counts": missing_value_counts,
        "quality_flag_counts_by_affected_stroke": dict(sorted(quality_flag_counts.items())),
        "rally_winner_status_counts": rally_winner_status_counts,
        "landing_area_domain_validation": (
            "not_performed_no_canonical_domain_configured"
        ),
    }
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
