"""Inverse problem: which launch makes this shot type land on this target? (blueprint 6.3)

A landing point alone does not determine a launch velocity (speed and elevation trade off),
so each shot type carries a *family prior*: an allowed speed range, an allowed elevation
range, a nominal elevation and the contact heights it is played from. The solver

  1. fixes the azimuth by aiming the horizontal velocity straight at the target
     (with zero wind, drag keeps the trajectory in a vertical plane),
  2. tries elevations in the family's range, nominal first and then outward in steps,
  3. for each elevation finds the speed whose range equals the target distance
     (bracketing + Illinois root finder, using a fast float-only 2-D integrator),
     where "range" is the horizontal distance at which the shuttle DESCENDS THROUGH the
     target height (`target_height`; 0 = floor landing, the default),
  4. rejects solutions that cannot clear the net by `min_net_margin`, and returns the first
     feasible one,
  5. optionally re-flies the answer through the authoritative 3-D `ShuttleFlightModel`
     and reports the residual from THAT, not from the fast integrator.

Height-aware targets: an opponent contacts the shuttle in the air, so the point to aim at is
(x, y, h) with h around 1-2.5 m, not the floor point below it. Aiming at the floor point
forces unrealistically steep, slow-descending shots. The shuttle continues past the target;
`floor_landing_xyz` reports where it would land if nobody hit it.

Infeasible shots return `feasible=False` with a reason and NaN launch fields. The solver never
fabricates a shot (blueprint 6.2). With `elevation_jitter_deg = 0` the result is deterministic;
with jitter it is deterministic given a seeded `rng`.

The `ShotType` list and every number in `DEFAULT_FAMILIES` are PLACEHOLDERS. Map `ShotType` to
the real ShuttleSet / CoachAI classes after the data audit and replace the ranges with
values fitted on tracked trajectories. The inverse solver does not support wind (it raises).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

from .shuttle_flight import (
    G,
    CourtGeometry,
    EventType,
    ShuttleFlightModel,
    make_launch_state,
)


class ShotType(IntEnum):
    """Placeholder set of 10 shot classes. Re-map to ShuttleSet/CoachAI classes later."""

    CLEAR = 0
    DROP = 1
    SMASH = 2
    HALF_SMASH = 3
    DRIVE = 4
    PUSH = 5
    LIFT = 6
    NET_SHOT = 7
    BLOCK = 8
    CUT = 9


@dataclass(frozen=True)
class ShotFamily:
    name: str
    speed_range: Tuple[float, float]            # m/s
    elevation_range_deg: Tuple[float, float]    # + above horizontal, - downward
    nominal_elevation_deg: float
    contact_height_range: Tuple[float, float]   # m, heights this shot can be played from


DEFAULT_FAMILIES: Dict[ShotType, ShotFamily] = {
    ShotType.CLEAR:      ShotFamily("clear",      (20.0, 55.0), (25.0, 60.0),  42.0, (1.6, 3.0)),
    ShotType.DROP:       ShotFamily("drop",       (8.0, 28.0),  (-10.0, 35.0), 12.0, (1.8, 3.0)),
    ShotType.SMASH:      ShotFamily("smash",      (40.0, 90.0), (-50.0, -10.0), -24.0, (2.2, 3.1)),
    ShotType.HALF_SMASH: ShotFamily("half_smash", (28.0, 60.0), (-40.0, 0.0),  -14.0, (1.8, 3.0)),
    ShotType.DRIVE:      ShotFamily("drive",      (22.0, 50.0), (-12.0, 15.0),  0.0, (0.9, 2.1)),
    ShotType.PUSH:       ShotFamily("push",       (12.0, 32.0), (-8.0, 22.0),   6.0, (0.7, 1.7)),
    ShotType.LIFT:       ShotFamily("lift",       (12.0, 40.0), (30.0, 70.0),  50.0, (0.25, 1.4)),
    ShotType.NET_SHOT:   ShotFamily("net_shot",   (3.0, 12.0),  (10.0, 65.0),  35.0, (0.5, 1.6)),
    ShotType.BLOCK:      ShotFamily("block",      (6.0, 22.0),  (-8.0, 28.0),  10.0, (0.7, 1.9)),
    ShotType.CUT:        ShotFamily("cut",        (14.0, 38.0), (-30.0, 10.0), -6.0, (1.8, 3.0)),
}


@dataclass(frozen=True)
class ContactState:
    """Where the shuttle is hit. Side of the court is the sign of position_xyz[1]."""

    position_xyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_xyz", np.array(self.position_xyz, dtype=float).reshape(3))


@dataclass(frozen=True)
class ShotExecutionConfig:
    min_net_margin: float = 0.05        # m above the net tape the shuttle must clear
    elevation_step_deg: float = 2.5
    range_tolerance_m: float = 5e-4     # accepted landing error of the fast solver
    max_speed_iterations: int = 60
    solver_dt: float = 1.0 / 60.0       # step of the fast integrator
    elevation_jitter_deg: float = 0.0   # std-dev of nominal-elevation noise (needs an rng)
    verify_tolerance_m: float = 5e-3    # max landing error accepted when re-flying the answer
    max_flight_time: float = 8.0


@dataclass(frozen=True, eq=False)
class LaunchResult:
    feasible: bool
    initial_position_xyz: np.ndarray
    initial_velocity_xyz: np.ndarray
    predicted_landing_xyz: np.ndarray
    flight_time_s: float
    margin_to_net: float
    residual: float             # distance between predicted landing and requested target (m)
    uncertainty: float          # 0.0 until a calibrated stochastic model exists
    reason: str                 # "ok" or why infeasible
    speed: float = float("nan")
    elevation_deg: float = float("nan")
    azimuth_deg: float = float("nan")
    target_height: float = 0.0
    floor_landing_xyz: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))

    # NOTE: `predicted_landing_xyz` is the point where the shuttle crosses `target_height`
    # on its way down (the floor landing when target_height == 0) and `flight_time_s` is
    # the time from contact to that crossing. `floor_landing_xyz` is the later floor landing.
    @property
    def predicted_target_xyz(self) -> np.ndarray:
        return self.predicted_landing_xyz


# failure reasons
OK = "ok"
BAD_CONTACT = "contact_invalid"
CONTACT_HEIGHT = "contact_height_outside_family_range"
TARGET_OUT = "target_out_of_bounds"
TARGET_OWN_SIDE = "target_on_own_side"
TOO_FAR = "target_too_far_for_family_speed"
TOO_CLOSE = "target_too_close_for_family_speed"
NET = "net_collision"
HEIGHT_NOT_REACHED = "target_height_not_reached"
HEIGHT_INVALID = "target_height_invalid"
SOLVER = "solver_failed"
VERIFY = "verification_failed"


def _infeasible(p0, target_xy, reason: str) -> LaunchResult:
    nan3 = np.full(3, np.nan)
    return LaunchResult(False, np.asarray(p0, float).copy(), nan3, nan3.copy(), float("nan"),
                        float("nan"), float("nan"), 0.0, reason)


class ShotExecutionModel:
    def __init__(
        self,
        flight: Optional[ShuttleFlightModel] = None,
        families: Optional[Dict[ShotType, ShotFamily]] = None,
        config: Optional[ShotExecutionConfig] = None,
    ) -> None:
        self.flight = flight or ShuttleFlightModel()
        if np.any(self.flight.wind != 0.0):
            raise NotImplementedError("the inverse solver assumes zero wind")
        self.families = families or DEFAULT_FAMILIES
        self.cfg = config or ShotExecutionConfig()
        self.court: CourtGeometry = self.flight.court
        self._k = self.flight.params.drag_k

    # ------------------------------------------------------------------ public API

    def solve_launch(
        self,
        contact: ContactState,
        shot_type: ShotType,
        target_xy,
        rng: Optional[np.random.Generator] = None,
        verify: bool = True,
        target_height: float = 0.0,
    ) -> LaunchResult:
        """Solve for a launch that passes over `target_xy` at `target_height` (m, descending).

        target_height = 0 reproduces the original floor-landing behaviour exactly.
        """
        fam = self.families[ShotType(shot_type)]
        p0 = contact.position_xyz
        target = np.asarray(target_xy, float).reshape(2)
        z_t = float(target_height)
        if not math.isfinite(z_t) or z_t < 0.0:
            return _infeasible(p0, target, HEIGHT_INVALID)

        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(target))) or p0[2] <= 0.0 or p0[1] == 0.0:
            return _infeasible(p0, target, BAD_CONTACT)
        if not (fam.contact_height_range[0] <= p0[2] <= fam.contact_height_range[1]):
            return _infeasible(p0, target, CONTACT_HEIGHT)
        if not self.court.is_in_singles(target[0], target[1]):
            return _infeasible(p0, target, TARGET_OUT)
        if target[1] * p0[1] >= 0.0:
            return _infeasible(p0, target, TARGET_OWN_SIDE)

        dx, dy = target[0] - p0[0], target[1] - p0[1]
        dist = math.hypot(dx, dy)
        az = math.atan2(dx, dy)
        s_net = abs(p0[1]) / abs(math.cos(az))          # horizontal distance to the net plane
        x_net = p0[0] + s_net * math.sin(az)

        failures = []
        for elev in self._elevation_candidates(fam, rng):
            sol = self._solve_speed(p0[2], elev, dist, s_net, fam, z_t)
            if isinstance(sol, str):
                failures.append(sol)
                continue
            speed, z_net = sol
            margin = z_net - self.court.net_top(x_net)
            if z_net != z_net or margin < self.cfg.min_net_margin:  # NaN or too low
                failures.append(NET)
                continue

            result = self._finalize(p0, target, speed, elev, math.degrees(az), margin, verify, z_t)
            if result.feasible:
                return result
            failures.append(result.reason)

        # report the most informative failure (the net is the most specific physical reason)
        if NET in failures:
            return _infeasible(p0, target, NET)
        if VERIFY in failures:
            return _infeasible(p0, target, VERIFY)
        if HEIGHT_NOT_REACHED in failures and all(f == HEIGHT_NOT_REACHED for f in failures):
            return _infeasible(p0, target, HEIGHT_NOT_REACHED)
        return _infeasible(p0, target, failures[0] if failures else SOLVER)

    # ------------------------------------------------------------------ internals

    def _elevation_candidates(self, fam: ShotFamily, rng) -> list:
        lo, hi = fam.elevation_range_deg
        nominal = fam.nominal_elevation_deg
        if self.cfg.elevation_jitter_deg > 0.0 and rng is not None:
            nominal = float(np.clip(nominal + rng.normal(0.0, self.cfg.elevation_jitter_deg), lo, hi))
        grid = list(np.arange(lo, hi + 1e-9, self.cfg.elevation_step_deg))
        cands = [nominal] + [g for g in grid if abs(g - nominal) > 1e-9]
        return sorted(cands, key=lambda e: abs(e - nominal))

    def _acc(self, vs: float, vz: float):
        sp = math.sqrt(vs * vs + vz * vz)
        return -self._k * sp * vs, -G - self._k * sp * vz

    def _rk4(self, z: float, s: float, vs: float, vz: float, h: float):
        a1s, a1z = self._acc(vs, vz)
        v2s, v2z = vs + 0.5 * h * a1s, vz + 0.5 * h * a1z
        a2s, a2z = self._acc(v2s, v2z)
        v3s, v3z = vs + 0.5 * h * a2s, vz + 0.5 * h * a2z
        a3s, a3z = self._acc(v3s, v3z)
        v4s, v4z = vs + h * a3s, vz + h * a3z
        a4s, a4z = self._acc(v4s, v4z)
        s_n = s + (h / 6.0) * (vs + 2.0 * v2s + 2.0 * v3s + v4s)
        z_n = z + (h / 6.0) * (vz + 2.0 * v2z + 2.0 * v3z + v4z)
        vs_n = vs + (h / 6.0) * (a1s + 2.0 * a2s + 2.0 * a3s + a4s)
        vz_n = vz + (h / 6.0) * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)
        return z_n, s_n, vs_n, vz_n

    def _fly_plane(self, z0: float, speed: float, elev_deg: float, s_net: float, z_t: float = 0.0):
        """Fast vertical-plane flight.

        Returns (s_cross, t_cross, z_at_net) where s_cross is the horizontal distance at which
        the shuttle first passes DOWNWARD through height z_t (z_t = 0: floor landing). If it
        never does (apex below z_t), s_cross is NaN.
        """
        el = math.radians(elev_deg)
        z, s, vs, vz = z0, 0.0, speed * math.cos(el), speed * math.sin(el)
        dt, t = self.cfg.solver_dt, 0.0
        z_net = float("nan")
        while t < self.cfg.max_flight_time:
            z1, s1, vs1, vz1 = self._rk4(z, s, vs, vz, dt)
            if z_net != z_net and s < s_net <= s1:
                h = dt * (s_net - s) / (s1 - s)
                for _ in range(3):  # Newton polish of the crossing time
                    zh, sh, vsh, _ = self._rk4(z, s, vs, vz, h)
                    h -= (sh - s_net) / max(vsh, 1e-9)
                z_net = self._rk4(z, s, vs, vz, h)[0]
            if z > z_t >= z1:       # downward crossing of the target height
                h = dt * (z - z_t) / (z - z1)
                for _ in range(3):
                    zh, sh, vsh, vzh = self._rk4(z, s, vs, vz, h)
                    if abs(vzh) < 1e-9:
                        break
                    h -= (zh - z_t) / vzh
                _, s_cross, _, _ = self._rk4(z, s, vs, vz, h)
                return s_cross, t + h, z_net
            if z1 <= 0.0:           # reached the floor without ever descending through z_t
                return float("nan"), t + dt, z_net
            z, s, vs, vz = z1, s1, vs1, vz1
            t += dt
        return float("nan"), t, z_net

    def _solve_speed(self, z0: float, elev_deg: float, dist: float, s_net: float, fam: ShotFamily, z_t: float = 0.0):
        """Speed whose range is `dist` at this elevation: (speed, z_net) or a failure reason."""
        tol = self.cfg.range_tolerance_m
        a, b = fam.speed_range
        s_b, _, zn_b = self._fly_plane(z0, b, elev_deg, s_net, z_t)
        if s_b != s_b:                       # NaN: never descends through z_t even at top speed
            return HEIGHT_NOT_REACHED
        if s_b < dist - tol:
            return TOO_FAR
        s_a, _, zn_a = self._fly_plane(z0, a, elev_deg, s_net, z_t)
        if s_a != s_a:                       # too slow to get above z_t: treat as "falls short"
            s_a = 0.0
        if s_a > dist + tol:
            return TOO_CLOSE
        fa, fb = s_a - dist, s_b - dist
        if abs(fa) <= tol:
            return a, zn_a
        if abs(fb) <= tol:
            return b, zn_b
        side = 0
        for _ in range(self.cfg.max_speed_iterations):  # Illinois (modified regula falsi)
            c = (a * fb - b * fa) / (fb - fa)
            s_c, _, zn_c = self._fly_plane(z0, c, elev_deg, s_net, z_t)
            if s_c != s_c:
                s_c = 0.0
            fc = s_c - dist
            if abs(fc) <= tol:
                return c, zn_c
            if fc * fb > 0:
                b, fb = c, fc
                if side == -1:
                    fa *= 0.5
                side = -1
            else:
                a, fa = c, fc
                if side == 1:
                    fb *= 0.5
                side = 1
        return SOLVER

    def _finalize(self, p0, target, speed, elev_deg, az_deg, fast_margin, verify, z_t=0.0) -> LaunchResult:
        az = math.radians(az_deg)
        vel = make_launch_state(p0, speed, elev_deg, az_deg).velocity_xyz

        if not verify:
            # fast-integrator answer, with a cheap second fly for the floor landing
            s_cross, t_cross, _ = self._fly_plane(p0[2], speed, elev_deg, 1e9, z_t)
            s_floor, _, _ = self._fly_plane(p0[2], speed, elev_deg, 1e9, 0.0)
            at = np.array([p0[0] + s_cross * math.sin(az), p0[1] + s_cross * math.cos(az), z_t])
            floor = np.array([p0[0] + s_floor * math.sin(az), p0[1] + s_floor * math.cos(az), 0.0])
            return LaunchResult(True, p0.copy(), vel, at, t_cross, float(fast_margin),
                                float(np.linalg.norm(at[:2] - target)), 0.0, OK,
                                speed, elev_deg, az_deg, z_t, floor)

        launch = make_launch_state(p0, speed, elev_deg, az_deg)
        res = self.flight.integrate_to_event(launch, descend_through=z_t if z_t > 0.0 else None)
        if res.terminal_event.type == EventType.NET_COLLISION:
            return _infeasible(p0, target, NET)
        if not res.landed:
            return _infeasible(p0, target, VERIFY)
        floor = res.terminal_event.position.copy()
        if z_t > 0.0:
            crossings = res.events_of(EventType.HEIGHT_CROSSING)
            if not crossings:
                return _infeasible(p0, target, VERIFY)
            at, t_cross = crossings[0].position.copy(), crossings[0].time
        else:
            at, t_cross = floor.copy(), res.flight_time
        residual = float(np.linalg.norm(at[:2] - target))
        if residual > self.cfg.verify_tolerance_m:
            return _infeasible(p0, target, VERIFY)
        net_crossings = res.events_of(EventType.NET_CROSSING)
        margin = fast_margin
        if net_crossings and net_crossings[0].info.get("clearance") is not None:
            margin = float(net_crossings[0].info["clearance"])
        if margin < self.cfg.min_net_margin:
            return _infeasible(p0, target, NET)
        return LaunchResult(True, p0.copy(), launch.velocity_xyz.copy(), at, t_cross,
                            margin, residual, 0.0, OK, speed, elev_deg, az_deg, z_t, floor)