"""Legal-action mask for the 1,350 macro-actions (blueprint 6.1 / 6.2).

    action = (shot_type, landing_cell, recovery_cell)
    index  = (shot_type * 15 + landing_cell) * 9 + recovery_cell

An action is valid when ALL of the following hold at the decision (contact) state:

  1. shot compatibility : the shot type can be played from this contact height
  2. target legality    : the landing cell centre is in the opponent's half of the singles court
  3. flight feasibility : `ShotExecutionModel` finds a launch that lands on the cell centre and
                          clears the net by the configured margin
  4. recovery           : the player can get within `recovery_tolerance_m` of the recovery cell
                          centre, starting from the contact state, before the opponent's
                          return can arrive (own flight time + `opponent_turnaround_s`)

The incoming-shuttle check ("can the player even reach the shuttle?") is NOT here: it happens
earlier via `reachability.can_intercept`; if the player cannot reach, the rally is lost without
an outgoing action.

Every masked action carries a reason string (`result.reason(index)`), and if nothing at all is
feasible the mask raises `NoFeasibleActionError` instead of inventing a shot.

The grid layout is an assumption you may change (3 lateral columns x 5 depth rows for landing,
3 x 3 for recovery); the codec here is a stand-in for `env/action_codec.py`. Cost: one launch
solve per (shot, landing cell) = 150 solves per decision (about 1 s in pure Python), cached per
exact contact position. For training, precompute a lookup table offline.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .player_motion import PlayerMotionModel, PlayerState
from .shot_execution import ContactState, LaunchResult, ShotExecutionModel, ShotType
from .shuttle_flight import CourtGeometry

log = logging.getLogger(__name__)

N_SHOTS = len(ShotType)       # 10
N_LANDING = 15                # 5 depth rows x 3 lateral columns over the opponent half
N_RECOVERY = 9                # 3 depth rows x 3 lateral columns over the own half
N_ACTIONS = N_SHOTS * N_LANDING * N_RECOVERY  # 1350

REASON_OK = "ok"
REASON_RECOVERY = "recovery_unreachable_in_time"
REASON_WRONG_SIDE = "contact_not_on_player_side"


class NoFeasibleActionError(RuntimeError):
    """Raised when every macro-action is infeasible (do not fabricate a shot)."""


# ---------------------------------------------------------------------- codec + grids

def encode_action(shot: int, landing: int, recovery: int) -> int:
    if not (0 <= shot < N_SHOTS and 0 <= landing < N_LANDING and 0 <= recovery < N_RECOVERY):
        raise ValueError("action component out of range")
    return (int(shot) * N_LANDING + int(landing)) * N_RECOVERY + int(recovery)


def decode_action(index: int) -> Tuple[ShotType, int, int]:
    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"action index {index} out of range")
    shot, rest = divmod(int(index), N_LANDING * N_RECOVERY)
    landing, recovery = divmod(rest, N_RECOVERY)
    return ShotType(shot), landing, recovery


@dataclass(frozen=True)
class CellGrid:
    """Cell centres. `side` is the half of the court the *player* defends (+1 / -1).

    landing cell  l: row = l // 3 (0 nearest the net .. 4 deepest), col = l % 3 (left x .. right x)
    recovery cell r: row = r // 3 (0 nearest the net .. 2 deepest), col = r % 3
    Landing cells cover the opponent half, recovery cells the player's own half.
    """

    court: CourtGeometry = field(default_factory=CourtGeometry)
    landing_rows: int = 5
    landing_cols: int = 3
    recovery_rows: int = 3
    recovery_cols: int = 3

    def _centre(self, idx: int, rows: int, cols: int, half_sign: int) -> np.ndarray:
        row, col = divmod(idx, cols)
        x = -self.court.half_width_singles + (col + 0.5) * (2 * self.court.half_width_singles / cols)
        depth = (row + 0.5) * (self.court.half_length / rows)
        return np.array([x, half_sign * depth])

    def landing_centre(self, landing: int, side: int) -> np.ndarray:
        return self._centre(landing, self.landing_rows, self.landing_cols, -side)

    def recovery_centre(self, recovery: int, side: int) -> np.ndarray:
        return self._centre(recovery, self.recovery_rows, self.recovery_cols, side)

    def sample_in_landing_cell(self, landing: int, side: int, rng: np.random.Generator, inner_frac: float = 0.8):
        """Seeded jitter inside a cell. Feasibility is checked on centres only, so a jittered
        target can still fail; callers should fall back to the centre if the solver says so."""
        c = self.landing_centre(landing, side)
        half_w = self.court.half_width_singles / self.landing_cols * inner_frac
        half_d = 0.5 * self.court.half_length / self.landing_rows * inner_frac
        return c + np.array([rng.uniform(-half_w, half_w), rng.uniform(-half_d, half_d)])


@dataclass(frozen=True)
class FeasibilityParams:
    opponent_turnaround_s: float = 0.7   # s added to own flight time before the return arrives (placeholder)
    recovery_tolerance_m: float = 0.75   # how close to the recovery cell centre counts as "there"


@dataclass(frozen=True, eq=False)
class ActionMaskResult:
    mask: np.ndarray                                  # bool[1350]
    reasons: np.ndarray                               # object[10, 15, 9] reason strings
    launches: Dict[Tuple[int, int], LaunchResult]     # per (shot, landing)

    @property
    def n_valid(self) -> int:
        return int(self.mask.sum())

    def reason(self, index: int) -> str:
        s, l, r = decode_action(index)
        return str(self.reasons[int(s), l, r])

    def masked_summary(self) -> Counter:
        flat = self.reasons.reshape(-1)
        return Counter(str(x) for x, ok in zip(flat, self.mask) if not ok)


class ActionFeasibility:
    def __init__(
        self,
        executor: Optional[ShotExecutionModel] = None,
        motion: Optional[PlayerMotionModel] = None,
        grid: Optional[CellGrid] = None,
        params: Optional[FeasibilityParams] = None,
        cache_limit: int = 20000,
    ) -> None:
        self.executor = executor or ShotExecutionModel()
        self.motion = motion or PlayerMotionModel(court=self.executor.court)
        self.grid = grid or CellGrid(self.executor.court)
        self.params = params or FeasibilityParams()
        self._cache: Dict[tuple, LaunchResult] = {}
        self._cache_limit = cache_limit

    def _launch(self, contact: ContactState, shot: ShotType, landing: int, side: int) -> LaunchResult:
        key = (tuple(np.round(contact.position_xyz, 4)), int(shot), landing, side)
        hit = self._cache.get(key)
        if hit is None:
            target = self.grid.landing_centre(landing, side)
            hit = self.executor.solve_launch(contact, shot, target, rng=None, verify=False)
            if len(self._cache) >= self._cache_limit:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def valid_action_mask(self, player: PlayerState, contact: ContactState) -> ActionMaskResult:
        """Mask for the player standing at the contact state (player hits the shuttle at `contact`).

        The player's velocity at contact matters for the recovery check; any pending
        reaction timer is ignored because the player has already reacted to hit the shuttle.
        """
        side = player.side
        reasons = np.full((N_SHOTS, N_LANDING, N_RECOVERY), "", dtype=object)
        mask = np.zeros((N_SHOTS, N_LANDING, N_RECOVERY), dtype=bool)
        launches: Dict[Tuple[int, int], LaunchResult] = {}

        if contact.position_xyz[1] * side <= 0.0:
            reasons[:] = REASON_WRONG_SIDE
            return self._finish(ActionMaskResult(mask.reshape(-1), reasons, launches))

        # 1-3: shot compatibility, target legality, flight feasibility
        for shot in ShotType:
            for landing in range(N_LANDING):
                res = self._launch(contact, shot, landing, side)
                launches[(int(shot), landing)] = res
                reasons[int(shot), landing, :] = res.reason

        # 4: recovery reachability, one batched simulation for all 9 recovery cells
        feasible_times = [r.flight_time_s for r in launches.values() if r.feasible]
        if feasible_times:
            rec_targets = np.stack([self.grid.recovery_centre(r, side) for r in range(N_RECOVERY)])
            standing = PlayerState(player.position_xy, player.velocity_xy, side, 0.0)
            t_max = max(feasible_times) + self.params.opponent_turnaround_s
            arrival = self.motion.arrival_times(
                standing, rec_targets, self.params.recovery_tolerance_m, t_max, arrive=False
            )
            for (shot, landing), res in launches.items():
                if not res.feasible:
                    continue
                budget = res.flight_time_s + self.params.opponent_turnaround_s
                ok = arrival <= budget
                mask[shot, landing, ok] = True
                reasons[shot, landing, ok] = REASON_OK
                reasons[shot, landing, ~ok] = REASON_RECOVERY

        return self._finish(ActionMaskResult(mask.reshape(-1), reasons, launches))

    def _finish(self, result: ActionMaskResult) -> ActionMaskResult:
        if result.n_valid == 0:
            log.warning("no feasible action; masked reasons: %s", dict(result.masked_summary()))
            raise NoFeasibleActionError(f"all {N_ACTIONS} actions infeasible: {dict(result.masked_summary())}")
        if log.isEnabledFor(logging.DEBUG):
            log.debug("valid actions %d/%d; masked reasons %s", result.n_valid, N_ACTIONS,
                      dict(result.masked_summary()))
        return result