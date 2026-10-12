"""Can a player intercept an incoming shuttle? (blueprint section 7.3, PLAYER_INTERCEPT / PLAYER_MISS)

The shuttle trajectory (a `FlightResult` from `shuttle_flight`) is sampled in time. A sample at
time tau is a *contact candidate* if

  1. tau <= deadline_s and the shuttle has not landed yet,
  2. the shuttle height is inside the contact envelope's height band,
  3. the shuttle is on the player's own half (nobody hits over the net),
  4. the player can get within `horizontal_radius` of the shuttle's (x, y) at or before tau,
     as simulated by the SAME controller `PlayerMotionModel.advance` uses (reaction delay,
     speed and acceleration limits, current velocity all included).

The earliest candidate is the contact. This is a modelling choice ("take the first chance"),
not a claim about what players do; the margin it returns tells you how comfortable it was.

Because the candidate set only grows when the deadline grows, feasibility is monotone in the
deadline (tested). The envelope numbers are placeholders until calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .player_motion import PlayerMotionModel, PlayerState
from .shuttle_flight import FlightResult


@dataclass(frozen=True)
class ContactEnvelope:
    """Where around the player a stroke can be made (all placeholders)."""

    horizontal_radius: float = 1.1  # m, racket + lunge reach in the floor plane
    min_height: float = 0.25        # m, lowest scoop
    max_height: float = 2.9         # m, highest (jump) contact

    def __post_init__(self) -> None:
        if self.horizontal_radius <= 0 or self.min_height >= self.max_height:
            raise ValueError("invalid contact envelope")


@dataclass(frozen=True, eq=False)
class ReachabilityResult:
    feasible: bool
    contact_time_s: float                    # nan if infeasible
    contact_position_xyz: np.ndarray         # nan array if infeasible
    arrival_time_s: float                    # when the player gets within reach (nan/inf if never)
    time_margin_s: float                     # contact_time - arrival_time, >= 0 when feasible
    player_at_contact: Optional[PlayerState]
    reason: str                              # "ok" or why not


NO_CANDIDATE_HEIGHT = "shuttle_never_in_height_band"
NO_CANDIDATE_SIDE = "shuttle_never_on_player_side"
TOO_FAR = "unreachable_before_deadline"
NO_TIME = "deadline_before_any_contact_candidate"


def _infeasible(reason: str, arrival: float = float("nan")) -> ReachabilityResult:
    return ReachabilityResult(False, float("nan"), np.full(3, np.nan), arrival, float("nan"), None, reason)


def can_intercept(
    player: PlayerState,
    trajectory: FlightResult,
    deadline_s: float,
    envelope: ContactEnvelope,
    motion: PlayerMotionModel,
    sample_dt: float = 0.01,
) -> ReachabilityResult:
    """Earliest feasible contact between `player` and the incoming `trajectory`.

    Time zero is the start of the trajectory (the opponent's contact). If the player's
    reaction timer is running (`player.reaction_remaining`) it is counted from that same
    instant, so call `motion.start_reaction(player)` when the opponent hits.
    """
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")
    t_end = min(float(deadline_s), trajectory.flight_time)
    if t_end < 0:
        return _infeasible(NO_TIME)

    taus = np.arange(0.0, t_end + 1e-12, sample_dt)
    pos = np.stack([np.interp(taus, trajectory.times, trajectory.states[:, k]) for k in range(3)], axis=1)

    in_band = (pos[:, 2] >= envelope.min_height) & (pos[:, 2] <= envelope.max_height)
    if not np.any(in_band):
        return _infeasible(NO_CANDIDATE_HEIGHT)
    own_side = pos[:, 1] * player.side >= -1e-9
    cand = in_band & own_side
    if not np.any(cand):
        return _infeasible(NO_CANDIDATE_SIDE)

    idx = np.flatnonzero(cand)
    arrival = motion.arrival_times(
        player, pos[idx, :2], envelope.horizontal_radius, t_max=float(taus[idx[-1]]), arrive=False
    )
    ok = arrival <= taus[idx] + 1e-9
    if not np.any(ok):
        return _infeasible(TOO_FAR, float(np.min(arrival)))

    first = int(np.argmax(ok))
    tau = float(taus[idx[first]])
    contact_xyz = pos[idx[first]].copy()
    at_contact = motion.advance(player, contact_xyz[:2], tau, arrive=False)
    return ReachabilityResult(
        feasible=True,
        contact_time_s=tau,
        contact_position_xyz=contact_xyz,
        arrival_time_s=float(arrival[first]),
        time_margin_s=float(tau - arrival[first]),
        player_at_contact=at_contact,
        reason="ok",
    )