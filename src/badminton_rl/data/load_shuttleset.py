"""Load ShuttleSet into one tidy DataFrame (court coordinates in metres, English shot labels).

Everything the audit found that a downstream user must know is handled here:
  * 19 shot labels, one more than the README documents ('未知球種' = unknown, 1,407 strokes);
    'passive drop' is spelled 過度切球 in the data and 過渡切球 in the README
  * `time` has 1 s resolution; `frame_num` is the usable clock, and fps differs per video
    (25 or 30), estimated here by regressing frame_num on time
  * coordinates are camera pixels and are converted with the per-video homography, then shifted by the
    per-match corner offset (24/44 matches have a +2.5 unit = 5 cm x-shift in the raw mapping)
  * Strokes keep their original row order; `ball_round` restarts at 1 every rally
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .court_coordinates import camera_to_court_units, corner_offset_units, parse_homography, units_to_metres

TYPE_EN = {
    "放小球": "net shot", "擋小球": "return net", "殺球": "smash", "點扣": "wrist smash",
    "挑球": "lob", "防守回挑": "defensive return lob", "長球": "clear", "平球": "drive",
    "小平球": "driven flight", "後場抽平球": "back-court drive", "切球": "drop",
    "過度切球": "passive drop", "過渡切球": "passive drop",   # data spelling / README spelling
    "推球": "push", "撲球": "rush", "防守回抽": "defensive return drive", "勾球": "cross-court net shot",
    "發短球": "short service", "發長球": "long service", "未知球種": "unknown",
}

DEFAULT_ROOT = Path(os.environ.get("SHUTTLESET_ROOT", "data/raw/shuttleset/set"))

_POINTS = {  # name -> (x column, y column)
    "player": ("player_location_x", "player_location_y"),
    "opponent": ("opponent_location_x", "opponent_location_y"),
    "hit": ("hit_x", "hit_y"),
    "landing": ("landing_x", "landing_y"),
}


def load_matches(root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(root or DEFAULT_ROOT)
    m = pd.read_csv(root / "match.csv")
    h = pd.read_csv(root / "homography.csv")[["video", "homography_matrix", "upleft_x", "upright_x", "downleft_x",
                                              "downright_x", "upleft_y", "upright_y", "downleft_y", "downright_y"]]
    return m.merge(h, on="video", how="left", validate="one_to_one")


def estimate_fps(frames: np.ndarray, seconds: np.ndarray) -> float:
    ok = np.isfinite(frames) & np.isfinite(seconds)
    return float(np.polyfit(seconds[ok], frames[ok], 1)[0])


def load_strokes(root: Optional[Path] = None) -> pd.DataFrame:
    """One row per stroke. Adds: match_id, video, set, t_s (frame/fps), fps, type_en,
    and <name>_x_m / <name>_y_m court coordinates for player/opponent/hit/landing."""
    root = Path(root or DEFAULT_ROOT)
    matches = load_matches(root)
    frames = []
    for m in matches.itertuples():
        H = parse_homography(m.homography_matrix)
        corners = np.array([[m.upleft_x, m.upleft_y], [m.upright_x, m.upright_y],
                            [m.downright_x, m.downright_y], [m.downleft_x, m.downleft_y]], float)
        offset = corner_offset_units(H, corners)     # 2.5 units in x for 24/44 matches, ~0 otherwise
        parts = []
        for s in (1, 2, 3):
            f = root / m.video / f"set{s}.csv"
            if f.exists():
                d = pd.read_csv(f)
                d["set"] = s
                parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        d["match_id"], d["video"] = m.id, m.video
        sec = pd.to_timedelta(d["time"]).dt.total_seconds().to_numpy()
        fps = estimate_fps(d["frame_num"].to_numpy(float), sec)
        d["fps"] = fps
        d["t_s"] = d["frame_num"] / fps
        for name, (cx, cy) in _POINTS.items():
            xy = units_to_metres(camera_to_court_units(H, d[[cx, cy]].to_numpy(float)) - offset)
            d[f"{name}_x_m"], d[f"{name}_y_m"] = xy[:, 0], xy[:, 1]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["type_en"] = out["type"].map(TYPE_EN)
    if out["type_en"].isna().any():
        raise ValueError(f"unmapped shot labels: {sorted(out.loc[out.type_en.isna(), 'type'].unique())}")
    return out