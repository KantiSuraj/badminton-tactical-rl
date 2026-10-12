"""Camera pixels <-> court coordinates for ShuttleSet.

Audit findings (see reports/audit/AUDIT.md):
  * `homography.csv` stores H such that  court_units = H @ camera_pixel  (the README states
    the inverse direction in words; applying H to the four annotated court corners gives the
    rectangle x in [25, 325], y in [150, 810] for all 44 matches, H^-1 does not).
  * That rectangle is the DOUBLES outer boundary: 300 x 660 units for 6.10 m x 13.40 m, so one
    unit is 2.033 cm (x) / 2.030 cm (y), not exactly 2 cm as CoachAI's `pixel * 2 / 100` assumes.
  * Net is at y = 480 units.
  * CAVEAT found by the Test A suite: the corner mapping is NOT identical across matches. For 24 of
    44 matches H maps the annotated court corners to x in [27.5, 327.5] instead of [25, 325]
    (a constant +2.5 unit = 5.1 cm shift, y is exact). `corner_offset_units` measures it per match
    and the loader subtracts it, so every match ends up in the same court frame.

Known limitation: the homography is a FLOOR-plane mapping. Player feet are on the floor, so
`player_location` is valid; the shuttle is airborne when hit, so `hit_*`/`landing_*` points
that are not floor contacts carry a perspective (parallax) error of unknown size.

Output frame (matches `dynamics/shuttle_flight.py`): metres, x lateral, y longitudinal,
net at y = 0, court centre at x = 0.
"""

from __future__ import annotations

import ast

import numpy as np

COURT_X0, COURT_X1 = 25.0, 325.0       # units, doubles side lines
COURT_Y0, COURT_Y1 = 150.0, 810.0      # units, back lines
NET_Y = 480.0
CENTRE_X = 0.5 * (COURT_X0 + COURT_X1)
M_PER_UNIT_X = 6.10 / (COURT_X1 - COURT_X0)    # 0.020333
M_PER_UNIT_Y = 13.40 / (COURT_Y1 - COURT_Y0)   # 0.020303


def parse_homography(text: str) -> np.ndarray:
    H = np.array(ast.literal_eval(text), dtype=float)
    if H.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    return H


def camera_to_court_units(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply H (camera pixel -> court units). Accepts (N, 2); NaN rows stay NaN."""
    pts = np.asarray(pts_xy, float).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = np.all(np.isfinite(pts), axis=1)
    if ok.any():
        p = np.c_[pts[ok], np.ones(ok.sum())] @ H.T
        out[ok] = p[:, :2] / p[:, 2:3]
    return out


def corner_offset_units(H: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    """Mean (dx, dy) by which H maps the 4 annotated court corners (UL, UR, DR, DL order) away
    from the canonical rectangle. Subtract it from transformed points to align matches."""
    want = np.array([[COURT_X0, COURT_Y0], [COURT_X1, COURT_Y0], [COURT_X1, COURT_Y1], [COURT_X0, COURT_Y1]])
    return (camera_to_court_units(H, corners_xy) - want).mean(axis=0)


def units_to_metres(units_xy: np.ndarray) -> np.ndarray:
    """Court units -> centred metres (net at y=0, centre line at x=0)."""
    u = np.asarray(units_xy, float).reshape(-1, 2)
    return np.stack([(u[:, 0] - CENTRE_X) * M_PER_UNIT_X, (u[:, 1] - NET_Y) * M_PER_UNIT_Y], axis=1)


def camera_to_court_metres(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    return units_to_metres(camera_to_court_units(H, pts_xy))