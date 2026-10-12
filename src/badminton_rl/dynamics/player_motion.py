"""Bounded kinematic player model: speed limit, acceleration limit, reaction delay.

A player is a point mass on the floor plane (court coordinates, same frame as
`shuttle_flight`: net at y = 0). The model is deliberately simple (blueprint section 7.3):

  * velocity is limited to `v_max`; acceleration norm is limited to `a_max` when speeding up
    and `a_brake` when slowing down
  * before the player reacts (`reaction_remaining > 0`) no steering is applied and the player
    only coasts, losing speed at `coast_decel`
  * the player cannot cross the net and cannot leave the court plus a run-off margin
  * steering is "seek / arrive": aim the velocity at the target at full speed (seek, used when
    the player only has to get within racket reach of a moving shuttle), or additionally slow
    down so the player stops at the target (arrive, used for recovery movement)

This is not time-optimal control. It is a feasible, deterministic, monotone controller, and
`advance` and `arrival_times` share the exact same step function, so a reachability answer
and the movement actually simulated can never disagree.

All numeric defaults below are PLACEHOLDERS to be calibrated from tracked player
displacements (blueprint 7.4). Do not report results using them as if they were measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .shuttle_flight import CourtGeometry


@dataclass(frozen=True)
class PlayerMotionParams:
    v_max: float = 4.5            # m/s, top court-movement speed (placeholder)
    a_max: float = 7.0            # m/s^2, acceleration limit (placeholder)
    a_brake: float = 9.0          # m/s^2, deceleration limit (placeholder)
    reaction_delay: float = 0.20  # s between opponent contact and first steering (placeholder)
    coast_decel: float = 3.0      # m/s^2, speed lost while still reacting (placeholder)
    dt: float = 1.0 / 120.0       # controller step
    run_off_margin: float = 1.5   # m the player may stand outside the singles lines
    net_standoff: float = 0.3     # m, closest the player's position may be to the net plane

    def __post_init__(self) -> None:
        for name in ("v_max", "a_max", "a_brake", "dt"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.reaction_delay < 0 or self.coast_decel < 0:
            raise ValueError("reaction_delay and coast_decel must be non-negative")


@dataclass(frozen=True, eq=False)
class PlayerState:
    """Player on the floor plane.

    side: +1 if the player defends the +y half, -1 for the -y half.
    reaction_remaining: seconds before the player starts steering toward a new target.
    """

    position_xy: np.ndarray
    velocity_xy: np.ndarray
    side: int
    reaction_remaining: float = 0.0

    def __post_init__(self) -> None:
        pos = np.array(self.position_xy, dtype=float).reshape(2)
        vel = np.array(self.velocity_xy, dtype=float).reshape(2)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError("player state contains NaN or inf")
        if self.side not in (-1, 1):
            raise ValueError("side must be +1 or -1")
        if self.reaction_remaining < 0 or not np.isfinite(self.reaction_remaining):
            raise ValueError("reaction_remaining must be finite and >= 0")
        object.__setattr__(self, "position_xy", pos)
        object.__setattr__(self, "velocity_xy", vel)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_xy))


class PlayerMotionModel:
    def __init__(self, params: Optional[PlayerMotionParams] = None, court: Optional[CourtGeometry] = None):
        self.params = params or PlayerMotionParams()
        self.court = court or CourtGeometry()
        self.x_limit = self.court.half_width_singles + self.params.run_off_margin
        self.y_far = self.court.half_length + self.params.run_off_margin
        self.y_near = self.params.net_standoff

    # ---- public API ---------------------------------------------------------------

    def start_reaction(self, player: PlayerState) -> PlayerState:
        """Call when the opponent hits the shuttle: the player's reaction timer restarts."""
        return replace(player, reaction_remaining=self.params.reaction_delay)

    def clip_velocity(self, velocity_xy: np.ndarray) -> np.ndarray:
        v = np.asarray(velocity_xy, float)
        sp = float(np.linalg.norm(v))
        return v if sp <= self.params.v_max else v * (self.params.v_max / sp)

    def advance(
        self,
        player: PlayerState,
        target_xy,
        elapsed_s: float,
        arrive: bool = True,
    ) -> PlayerState:
        """Move the player toward `target_xy` for `elapsed_s` seconds.

        arrive=True  : decelerate so the player stops at the target (recovery movement).
        arrive=False : keep full speed through the target (interception).
        An over-speed initial velocity is clipped to v_max rather than rejected.
        """
        if elapsed_s < 0:
            raise ValueError("elapsed_s must be >= 0")
        p = player.position_xy[None, :].copy()
        v = self.clip_velocity(player.velocity_xy)[None, :]
        tgt = np.asarray(target_xy, float).reshape(1, 2)
        p, v, _, _ = self._simulate(p, v, tgt, player.side, player.reaction_remaining, elapsed_s, arrive)
        return PlayerState(p[0], v[0], player.side, max(0.0, player.reaction_remaining - elapsed_s))

    def arrival_times(
        self,
        player: PlayerState,
        targets_xy: np.ndarray,
        radius: float,
        t_max: float,
        arrive: bool = False,
    ) -> np.ndarray:
        """First time (s) each target is within `radius` of the player (inf if not by t_max).

        Vectorised: one simulation step serves all N targets, each with its own steering.
        Times are step-end times, so they are conservative by at most one controller `dt`.
        """
        tg = np.asarray(targets_xy, float).reshape(-1, 2)
        n = tg.shape[0]
        p = np.repeat(player.position_xy[None, :], n, axis=0)
        v = np.repeat(self.clip_velocity(player.velocity_xy)[None, :], n, axis=0)
        _, _, arrival, _ = self._simulate(p, v, tg, player.side, player.reaction_remaining, t_max, arrive, radius)
        return arrival

    # ---- internals ----------------------------------------------------------------

    def _simulate(self, p, v, targets, side, reaction, duration, arrive, radius=None):
        dt = self.params.dt
        n = p.shape[0]
        arrival = np.full(n, np.inf)
        if radius is not None:
            arrival[np.linalg.norm(targets - p, axis=1) <= radius] = 0.0
        t = 0.0
        while t < duration - 1e-12:
            if radius is not None and np.all(np.isfinite(arrival)):
                break
            in_react = (reaction - t) > 1e-12
            h = min(dt, duration - t)
            if in_react:
                h = min(h, reaction - t)
            p, v = self._step(p, v, targets, side, in_react, h, arrive)
            t += h
            if radius is not None:
                newly = np.isinf(arrival) & (np.linalg.norm(targets - p, axis=1) <= radius)
                arrival[newly] = t
        return p, v, arrival, t

    def _step(self, p, v, targets, side, in_react, h, arrive):
        prm = self.params
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        if in_react:
            unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 1e-12)
            v_new = v - unit_v * np.minimum(prm.coast_decel * h, speed)
        else:
            to = targets - p
            dist = np.linalg.norm(to, axis=1, keepdims=True)
            unit = np.divide(to, dist, out=np.zeros_like(to), where=dist > 1e-9)
            if arrive:
                # brake-limited approach speed, also capped so one step can never overshoot the target
                des_speed = np.minimum(np.minimum(prm.v_max, np.sqrt(2.0 * prm.a_brake * dist)), dist / h)
            else:
                des_speed = np.full_like(dist, prm.v_max)
            dv = unit * des_speed - v
            dv_norm = np.linalg.norm(dv, axis=1, keepdims=True)
            decelerating = np.sum(dv * v, axis=1, keepdims=True) < 0.0
            limit = np.where(decelerating, prm.a_brake, prm.a_max) * h
            scale = np.minimum(1.0, limit / np.maximum(dv_norm, 1e-12))
            v_new = v + dv * scale
            sp = np.linalg.norm(v_new, axis=1, keepdims=True)
            v_new = v_new * np.minimum(1.0, prm.v_max / np.maximum(sp, 1e-12))

        p_new = p + 0.5 * (v + v_new) * h

        x = np.clip(p_new[:, 0], -self.x_limit, self.x_limit)
        y = np.clip(p_new[:, 1] * side, self.y_near, self.y_far) * side
        v_new = v_new.copy()
        v_new[x != p_new[:, 0], 0] = 0.0
        v_new[y != p_new[:, 1], 1] = 0.0
        p_new = np.stack([x, y], axis=1)
        return p_new, v_new