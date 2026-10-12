"""Test A from the blueprint: data and coordinate integrity.

Tests marked `needs_data` run only when SHUTTLESET_ROOT points at the ShuttleSet `set/` folder
(pinned upstream commit 45517f7d4cb936b03f3eabf939cc7959d39226fe). A count mismatch there means
the upstream data changed and the audit must be re-run.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from badminton_rl.data.court_coordinates import (
    camera_to_court_units, camera_to_court_metres, parse_homography, units_to_metres,
)

ROOT = Path(os.environ.get("SHUTTLESET_ROOT", "data/raw/shuttleset/set"))
needs_data = pytest.mark.skipif(not (ROOT / "match.csv").exists(),
                                reason="set SHUTTLESET_ROOT to the ShuttleSet 'set' directory")


# ----------------------------------------------------------- pure functions (no data needed)

def test_identity_homography_and_nan_rows():
    out = camera_to_court_units(np.eye(3), np.array([[10.0, 20.0], [np.nan, 5.0]]))
    assert out[0] == pytest.approx([10.0, 20.0])
    assert np.all(np.isnan(out[1]))


def test_units_to_metres_corners_and_net():
    m = units_to_metres(np.array([[25.0, 150.0], [325.0, 810.0], [175.0, 480.0]]))
    assert m[0] == pytest.approx([-3.05, -6.70])
    assert m[1] == pytest.approx([3.05, 6.70])
    assert m[2] == pytest.approx([0.0, 0.0])


def test_parse_homography_validates_shape():
    assert parse_homography("[[1,0,0],[0,1,0],[0,0,1]]").shape == (3, 3)
    with pytest.raises(ValueError):
        parse_homography("[[1,0],[0,1]]")


# ----------------------------------------------------------- real data

def _corners(h):
    return np.array([[h.upleft_x, h.upleft_y], [h.upright_x, h.upright_y],
                     [h.downright_x, h.downright_y], [h.downleft_x, h.downleft_y]], float)


@needs_data
def test_homography_direction_and_corner_offsets():
    from badminton_rl.data.court_coordinates import corner_offset_units
    want = np.array([[25, 150], [325, 150], [325, 810], [25, 810]], float)
    dx = []
    for h in pd.read_csv(ROOT / "homography.csv").itertuples():
        H, cam = parse_homography(h.homography_matrix), _corners(h)
        raw = camera_to_court_units(H, cam) - want
        assert np.abs(raw[:, 1]).max() < 0.01                       # y is exact in every match
        assert np.abs(raw[:, 0] - raw[:, 0].mean()).max() < 0.15    # x error is a pure shift (corner pixels are annotated to ~0.2 px)
        dx.append(round(float(raw[:, 0].mean()), 1))
        # after the per-match offset the corners land on the canonical rectangle
        fixed = camera_to_court_units(H, cam) - corner_offset_units(H, cam)
        assert np.abs(fixed - want).max() < 0.15                      # 0.15 unit = 3 mm
        # the inverse direction is wrong
        assert np.abs(camera_to_court_units(np.linalg.inv(H), cam) - want).max() > 100
    assert set(dx) == {0.0, 2.5} and dx.count(2.5) == 24            # two conventions: 20 + 24 matches


@needs_data
def test_counts_labels_and_fps():
    from badminton_rl.data.load_shuttleset import load_strokes
    df = load_strokes(ROOT)
    assert df.match_id.nunique() == 44 and df.groupby(["match_id", "set"]).ngroups == 104
    assert len(df) == 36484 and df.type.nunique() == 19
    assert df.type_en.notna().all()
    fps = df.groupby("video").fps.first()
    assert np.all(np.minimum(np.abs(fps - 25), np.abs(fps - 30)) < 0.1)
    assert set(fps.round()) == {25.0, 30.0}


@needs_data
def test_players_are_inside_court_and_on_opposite_sides_of_net():
    from badminton_rl.data.load_shuttleset import load_strokes
    df = load_strokes(ROOT)
    p = df.dropna(subset=["player_x_m", "player_y_m", "opponent_y_m"])
    inside = (p.player_x_m.abs() <= 3.55) & (p.player_y_m.abs() <= 7.2)
    assert inside.mean() > 0.999
    assert (p.player_y_m * p.opponent_y_m < 0).mean() > 0.97   # hitter and receiver are on opposite halves