"""Compare CoachAI's closed-form ball model with this project's RK4 quadratic-drag model.

CoachAI (CoachAI Badminton Environment/StrategicEnvironment/multiagent/core.py, class Ball) uses
g = 9.8, V_T = 6.86 m/s and the world default `resistance_force_n = 1`, i.e. LINEAR drag,
with the closed forms (x horizontal distance, y height gain, relative to the launch point):

    n = 1:  x = V_T*Vx/g * (1 - exp(-g t/V_T))
            y = V_T/g * (V_T + Vy) * (1 - exp(-g t/V_T)) - V_T t          (exact for linear drag)
    n = 2:  x = V_T^2/g * ln((Vx g t + V_T^2)/V_T^2)                      (horizontal-only drag)
            y = V_T^2/g * ln|sin(g t/V_T + atan(V_T/Vy)) / sin(atan(V_T/Vy))|   (ascent solution)

This script re-implements those formulas (not imported: the repo targets Python 3.8/old gym) and
compares landing range and time with the quadratic, coupled-axes model used here, for the same V_T.
Run:  PYTHONPATH=src python scripts/audit/compare_flight_models.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from badminton_rl.dynamics.shuttle_flight import ShuttleFlightModel, ShuttleParams, make_launch_state

V_T, G_COACHAI = 6.86, 9.8


def coachai_xy(t, v, elev_deg, n):
    el = math.radians(elev_deg)
    vx, vy = v * math.cos(el), v * math.sin(el)
    if n == 1:
        k = 1 - math.exp(-G_COACHAI * t / V_T)
        return V_T * vx / G_COACHAI * k, V_T / G_COACHAI * (V_T + vy) * k - V_T * t
    x = V_T**2 / G_COACHAI * math.log((vx * G_COACHAI * t + V_T**2) / V_T**2)
    num = math.sin(G_COACHAI * t / V_T + math.atan(V_T / vy))
    den = math.sin(math.atan(V_T / vy))
    y = V_T**2 / G_COACHAI * math.log(abs(num / den))
    return x, y


def coachai_landing(v, elev_deg, z0, n, t_max=6.0):
    """(range, time) at which height z0 + y(t) first returns to 0, or None."""
    ts = np.arange(0.0, t_max, 0.001)
    prev = z0
    for t in ts[1:]:
        try:
            _, y = coachai_xy(t, v, elev_deg, n)
        except (ValueError, ZeroDivisionError):
            return None
        h = z0 + y
        if h <= 0 < prev:
            return coachai_xy(t, v, elev_deg, n)[0], float(t)
        prev = h
    return None


CASES = [  # name, speed m/s, elevation deg, contact height m
    ("net shot", 8.0, 35.0, 1.0),
    ("drop", 18.0, 12.0, 2.5),
    ("lob", 24.0, 50.0, 0.8),
    ("clear", 35.0, 35.0, 2.5),
    ("clear (fast)", 50.0, 25.0, 2.5),
    ("smash", 60.0, -15.0, 2.8),
]


def main() -> None:
    quad = ShuttleFlightModel(ShuttleParams.from_terminal_speed(V_T), max_time=10.0)
    rows = []
    for name, v, el, z0 in CASES:
        res = quad.integrate_to_event(make_launch_state([0.0, 0.0, z0], v, el, 0.0))
        r_q, t_q = float(res.terminal_event.position[1]), res.flight_time
        row = {"case": name, "speed": v, "elev_deg": el, "z0": z0, "quadratic_range_m": round(r_q, 2),
               "quadratic_time_s": round(t_q, 2)}
        for n in (1, 2):
            if n == 2 and el <= 0:
                row["coachai_n2"] = None
                continue
            out = coachai_landing(v, el, z0, n)
            row[f"coachai_n{n}"] = None if out is None else {
                "range_m": round(out[0], 2), "time_s": round(out[1], 2),
                "range_ratio_vs_quadratic": round(out[0] / r_q, 2), "time_ratio_vs_quadratic": round(out[1] / t_q, 2)}
        rows.append(row)
    for r in rows:
        print(json.dumps(r))
    out = Path("reports/audit/flight_model_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"V_T": V_T, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()