"""Event pipeline: schema, invariants, provenance, and row-count preservation.

The synthetic fixture follows the same per-match/per-set directory layout as
ShuttleSet.  The real dataset is used only by tests explicitly marked below.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from badminton_rl.data.schemas import EVENT_COLUMNS, QUALITY_FLAGS


# Keep the fixture's columns explicit.  In particular, frame_num is required
# by load_strokes() to estimate FPS; writing dictionaries prevents positional
# row/header drift when this schema changes.
_FIXTURE_COLUMNS = [
    "time",
    "set",
    "rally",
    "player",
    "opponent",
    "type",
    "frame_num",
    "player_location_x",
    "player_location_y",
    "opponent_location_x",
    "opponent_location_y",
    "hit_x",
    "hit_y",
    "landing_x",
    "landing_y",
    "hit_height",
    "landing_height",
    "ball_round",
    "player_location_area",
    "opponent_location_area",
    "landing_area",
    "db",
    "lose_reason",
    "getpoint_player",
]


def _stroke(
    *,
    second: int,
    frame_num: int,
    rally: int,
    ball_round: int,
    player: str,
    opponent: str,
    shot_type: str,
    getpoint_player: str = "A",
    hit_x: float | None = 100.0,
    hit_y: float | None = 200.0,
    landing_area: str = "front_left",
    db: str = "raw-db-sentinel",
) -> dict:
    """Return one structurally valid synthetic stroke."""
    return {
        "time": f"0:00:{second:02d}",
        "set": 1,
        "rally": rally,
        "player": player,
        "opponent": opponent,
        "type": shot_type,
        "frame_num": frame_num,
        "player_location_x": 100 if player == "A" else 200,
        "player_location_y": 200 if player == "A" else 300,
        "opponent_location_x": 200 if player == "A" else 100,
        "opponent_location_y": 300 if player == "A" else 200,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "landing_x": 175,
        "landing_y": 250,
        "hit_height": 1.5,
        "landing_height": 0.0,
        "ball_round": ball_round,
        "player_location_area": "rear_left" if player == "A" else "rear_right",
        "opponent_location_area": "rear_right" if player == "A" else "rear_left",
        "landing_area": landing_area,
        "db": db,
        "lose_reason": "",
        "getpoint_player": getpoint_player,
    }


def _write_set_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIXTURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_fake_dataset(
    tmp: Path,
    *,
    partial_hit_coordinate: bool = False,
    invalid_player_role: bool = False,
    inconsistent_winner: bool = False,
) -> Path:
    """Create four matches, including player overlap and edge-case variants.

    This fixture has seven unique players. With the 80/20 partition and seed
    42, only Dave is held out (round(0.20 * 7) == 1), so m3 is mixed and there
    is intentionally no fully held-out match. Split unit tests disable only
    the production test_player sample-size gate.
    """
    root = tmp / "set"
    root.mkdir()

    (root / "match.csv").write_text(
        "id,video,winner,loser\n"
        "m1,v1,Alice,Bob\n"
        "m2,v2,Alice,Cara\n"
        "m3,v3,Dave,Eve\n"
        "m4,v4,Frank,Frank2\n",
        encoding="utf-8",
    )

    # Identity homography; corners map to themselves.
    homography = "[[1,0,0],[0,1,0],[0,0,1]]"
    with (root / "homography.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video", "homography_matrix",
                "upleft_x", "upleft_y", "upright_x", "upright_y",
                "downleft_x", "downleft_y", "downright_x", "downright_y",
            ]
        )
        for video in ("v1", "v2", "v3", "v4"):
            writer.writerow([video, homography, 25, 150, 325, 150, 25, 810, 325, 810])

    v1_rows = [
        _stroke(second=1, frame_num=30, rally=1, ball_round=1,
                player="A", opponent="B", shot_type="長球"),
        _stroke(second=2, frame_num=60, rally=1, ball_round=2,
                player="B", opponent="A", shot_type="放小球"),
        _stroke(second=3, frame_num=90, rally=2, ball_round=1,
                player="A", opponent="B", shot_type="未知球種"),
        _stroke(second=4, frame_num=120, rally=2, ball_round=2,
                player="B", opponent="A", shot_type="殺球"),
    ]

    if partial_hit_coordinate:
        # One coordinate is absent while its pair is present.  The event
        # builder must null the whole pair rather than impute either value.
        v1_rows[0]["hit_x"] = None
    if invalid_player_role:
        v1_rows[0]["player"] = "X"
    if inconsistent_winner:
        # The first two strokes belong to rally 1 but disagree on its winner.
        v1_rows[1]["getpoint_player"] = "B"

    for video in ("v1", "v2", "v3", "v4"):
        (root / video).mkdir()

    _write_set_csv(root / "v1" / "set1.csv", v1_rows)

    # Two strokes in one rally for each remaining match.
    other_matches = {
        "v2": [
            _stroke(second=1, frame_num=30, rally=1, ball_round=1,
                    player="A", opponent="B", shot_type="長球"),
            _stroke(second=2, frame_num=60, rally=1, ball_round=2,
                    player="B", opponent="A", shot_type="放小球"),
        ],
        "v3": [
            _stroke(second=1, frame_num=30, rally=1, ball_round=1,
                    player="A", opponent="B", shot_type="長球"),
            _stroke(second=2, frame_num=60, rally=1, ball_round=2,
                    player="B", opponent="A", shot_type="殺球"),
        ],
        "v4": [
            _stroke(second=1, frame_num=30, rally=1, ball_round=1,
                    player="A", opponent="B", shot_type="放小球"),
            _stroke(second=2, frame_num=60, rally=1, ball_round=2,
                    player="B", opponent="A", shot_type="長球"),
        ],
    }
    for video, rows in other_matches.items():
        _write_set_csv(root / video / "set1.csv", rows)

    return root


@pytest.fixture
def events(tmp_path):
    from badminton_rl.data.build_events import build_events

    root = _make_fake_dataset(tmp_path)
    return build_events(root=root, output_path=tmp_path / "out")


def _flags(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {flag for flag in str(value).split(";") if flag}


def test_schema_columns_present(events):
    assert list(events.columns) == list(EVENT_COLUMNS)


def test_no_duplicate_event_ids(events):
    assert events["event_id"].is_unique
    assert events["event_id"].notna().all()


def test_rally_id_deterministic(events):
    assert (
        events["rally_id"]
        == events["match_id"]
        + "|s"
        + events["set_id"].astype(str)
        + "|r"
        + events["raw_rally_id"].astype(str)
    ).all()


def test_stroke_index_zero_based_and_contiguous(events):
    for _, group in events.groupby("rally_id", sort=False):
        assert list(group["stroke_index"]) == list(range(len(group)))


def test_xy_pair_or_null_invariant(events):
    for name in ("hit", "landing", "player", "opponent"):
        x = events[f"{name}_x_m"]
        y = events[f"{name}_y_m"]
        assert (x.isna() == y.isna()).all(), f"{name}: pair-or-null violated"


def test_missing_coordinates_not_imputed(tmp_path):
    from badminton_rl.data.build_events import build_events

    root = _make_fake_dataset(tmp_path, partial_hit_coordinate=True)
    events = build_events(root=root, output_path=tmp_path / "out")
    row = events.loc[events["source_file"].eq("v1/set1.csv")].iloc[0]
    assert pd.isna(row["hit_x_m"])
    assert pd.isna(row["hit_y_m"])
    assert "missing_hit_xy" in _flags(row["quality_flags"])


def test_unknown_label_flagged(events):
    unknown = events[events["source_shot_label"].eq("unknown")]
    assert not unknown.empty
    assert unknown["mapping_status"].eq("unmapped").all()
    assert unknown["shot_family"].isna().all()
    assert unknown["quality_flags"].map(
        lambda value: "unknown_shot_label" in _flags(value)
    ).all()


def test_player_a_is_winner_player_b_is_loser(events):
    participants = {
        "m1": ("Alice", "Bob"),
        "m2": ("Alice", "Cara"),
        "m3": ("Dave", "Eve"),
        "m4": ("Frank", "Frank2"),
    }
    for row in events.itertuples(index=False):
        winner, loser = participants[row.match_id]
        if row.player_role == "A":
            assert row.hitter_id == winner
            assert row.receiver_id == loser
        elif row.player_role == "B":
            assert row.hitter_id == loser
            assert row.receiver_id == winner


def test_valid_role_maps_to_correct_ids(events):
    for row in events.itertuples(index=False):
        assert row.player_role in {"A", "B"}
        assert row.hitter_id is not None and not pd.isna(row.hitter_id)
        assert row.receiver_id is not None and not pd.isna(row.receiver_id)
        assert row.hitter_id != row.receiver_id


def test_invalid_player_role_preserved(tmp_path):
    from badminton_rl.data.build_events import build_events

    root = _make_fake_dataset(tmp_path, invalid_player_role=True)
    events = build_events(root=root, output_path=tmp_path / "out")
    row = events.loc[events["source_file"].eq("v1/set1.csv")].iloc[0]
    assert len(events) == 10
    assert "invalid_player_role" in _flags(row["quality_flags"])


def test_getpoint_player_consistent_winner(events):
    v1 = events[events["match_id"].eq("m1")]
    assert v1["rally_winner_status"].eq("validated").all()
    for _, group in v1.groupby("rally_id"):
        assert group["rally_winner_role"].nunique() == 1
        assert group["rally_winner_role"].iloc[0] == "A"


def test_inconsistent_winner_flagged(tmp_path):
    from badminton_rl.data.build_events import build_events

    root = _make_fake_dataset(tmp_path, inconsistent_winner=True)
    events = build_events(root=root, output_path=tmp_path / "out")
    rally = events.loc[events["rally_id"].eq("m1|s1|r1")]
    assert len(rally) == 2
    assert rally["rally_winner_status"].eq("inconsistent").all()
    assert rally["rally_winner_role"].isna().all()
    assert rally["rally_winner_id"].isna().all()
    assert rally["quality_flags"].map(
        lambda value: "rally_winner_inconsistent" in _flags(value)
    ).all()


def test_landing_area_preserved_when_present(events):
    assert events["landing_area"].eq("front_left").all()


def test_db_preserved_verbatim(events):
    assert "db_raw" in events.columns
    assert events["db_raw"].eq("raw-db-sentinel").all()


def test_quality_flags_are_in_vocabulary(events):
    for value in events["quality_flags"].dropna():
        for flag in _flags(value):
            assert flag in QUALITY_FLAGS, f"unexpected flag: {flag}"


def test_parquet_round_trip(tmp_path):
    from badminton_rl.data.build_events import build_events

    root = _make_fake_dataset(tmp_path)
    built = build_events(root=root, output_path=tmp_path / "out")
    read_back = pd.read_parquet(tmp_path / "out" / "events.parquet")

    assert len(read_back) == len(built)
    for column in EVENT_COLUMNS:
        pd.testing.assert_series_equal(
            read_back[column].reset_index(drop=True),
            built[column].reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )


_REAL_DATA_ROOT = Path("data/raw/shuttleset/set")


@pytest.mark.skipif(
    not _REAL_DATA_ROOT.exists(),
    reason="real ShuttleSet not present",
)
def test_real_row_count_preservation(tmp_path):
    from badminton_rl.data.build_events import build_events
    from badminton_rl.data.load_shuttleset import load_strokes

    built = build_events(output_path=tmp_path / "out")
    source = load_strokes()
    assert len(built) == len(source)


@pytest.mark.skipif(
    not _REAL_DATA_ROOT.exists(),
    reason="real ShuttleSet not present",
)
def test_documented_readme_gap_is_present(tmp_path):
    """If this changes, update the manifest and methods text; do not hide it."""
    from badminton_rl.data.build_events import build_events

    events = build_events(output_path=tmp_path / "out")
    assert len(events) == 36484
    assert events["rally_id"].nunique() == 3683
