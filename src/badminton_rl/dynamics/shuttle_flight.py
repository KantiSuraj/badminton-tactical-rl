"""Reduced-order shuttlecock flight model (gravity + quadratic drag) with event detection.

Coordinate frame (metres, seconds), origin at the centre of the court on the floor:

    x : lateral, +x to the right when looking toward +y
    y : longitudinal, the net is the plane y = 0; the two halves are y < 0 and y > 0
    z : up, the floor is z = 0

Equation of motion (relative to optional wind w):

    dp/dt = v
    dv/dt = g_vec - k * |v - w| * (v - w)          k = 0.5 * rho * CdA / m   [1/m]

`k` is the only drag quantity that matters for the trajectory. A convenient way to think
about it is the terminal speed v_t = sqrt(g / k). Feather shuttles fall at roughly 6.5-7 m/s.

Event detection
---------------
RK4 advances the state in small fixed steps. After every step each event function
g(state) is checked for a sign change. When one flips, the exact event time inside the step
is found by bisection (re-integrating from the step start with a shorter step), so event
times/positions are not quantised to the integration grid.

Events: NET_CROSSING / NET_COLLISION (plane y = 0), SHUTTLE_LANDING (z = 0),
BOUNDARY_CROSSING (leaving the singles court footprint in the air, informational),
HEIGHT_CROSSING (optional: the shuttle descends through a requested height, e.g. where an
opponent would contact it), TIMEOUT (safety cap).
NET_COLLISION, SHUTTLE_LANDING and TIMEOUT are terminal; the others are not.

Known simplifications (state these in your report):
  * shuttle is a point mass; no feather orientation/spin/flip effects
  * drag coefficient is constant (real shuttles change Cd with speed)
  * net is a thin sheet with a height profile; no tape, no shuttle radius, no bounce
  * a very fast shuttle crossing the same plane twice inside one 1/240 s step is missed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

G = 9.80665  # m/s^2
GRAVITY_VEC = np.array([0.0, 0.0, -G])


# --------------------------------------------------------------------------------------
# Parameters and geometry
# --------------------------------------------------------------------------------------

_DEFAULT_MASS = 0.0051          # kg (rules allow about 4.74-5.50 g)
_DEFAULT_RHO = 1.2              # kg/m^3
_DEFAULT_TERMINAL_SPEED = 6.8   # m/s, typical feather shuttle (placeholder until calibrated)
_DEFAULT_CDA = 2.0 * _DEFAULT_MASS * G / (_DEFAULT_RHO * _DEFAULT_TERMINAL_SPEED**2)


@dataclass(frozen=True)
class ShuttleParams:
    """Physical parameters of the shuttle. `drag_area` is the effective Cd*A in m^2.

    These defaults are placeholders. In the project they are calibrated from tracked
    trajectories (milestone: weeks 6-9), so treat any number here as an initial guess.
    """

    mass: float = _DEFAULT_MASS
    air_density: float = _DEFAULT_RHO
    drag_area: float = _DEFAULT_CDA

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.air_density < 0 or self.drag_area < 0:
            raise ValueError("air_density and drag_area must be non-negative")

    @classmethod
    def from_terminal_speed(
        cls,
        terminal_speed: float,
        mass: float = _DEFAULT_MASS,
        air_density: float = _DEFAULT_RHO,
    ) -> "ShuttleParams":
        """Build parameters so a vertical drop reaches the given terminal speed."""
        if terminal_speed <= 0:
            raise ValueError("terminal_speed must be positive")
        cda = 2.0 * mass * G / (air_density * terminal_speed**2)
        return cls(mass=mass, air_density=air_density, drag_area=cda)

    @classmethod
    def no_drag(cls) -> "ShuttleParams":
        """Vacuum projectile motion. Useful for tests against analytic solutions."""
        return cls(drag_area=0.0)

    @property
    def drag_k(self) -> float:
        """k = 0.5 * rho * CdA / m, units 1/m."""
        return 0.5 * self.air_density * self.drag_area / self.mass

    @property
    def terminal_speed(self) -> float:
        k = self.drag_k
        return math.inf if k == 0 else math.sqrt(G / k)


@dataclass(frozen=True)
class CourtGeometry:
    """Singles court, centred coordinates. Net plane at y = 0."""

    half_length: float = 6.70           # 13.40 m long
    half_width_singles: float = 2.59    # 5.18 m wide
    net_half_width: float = 3.05        # posts stand on the doubles sidelines
    net_height_center: float = 1.524
    net_height_post: float = 1.55

    def net_top(self, x: float) -> float:
        """Height of the net's top edge at lateral position x (sags slightly mid-court)."""
        frac = min(abs(x) / self.net_half_width, 1.0)
        return self.net_height_center + (self.net_height_post - self.net_height_center) * frac

    def is_in_singles(self, x: float, y: float) -> bool:
        """Landing on the line counts as in."""
        return abs(x) <= self.half_width_singles and abs(y) <= self.half_length

    def margin_to_line(self, x: float, y: float) -> float:
        """Distance to the nearest boundary line. Positive = inside, negative = outside."""
        return min(self.half_width_singles - abs(x), self.half_length - abs(y))


# --------------------------------------------------------------------------------------
# State and results
# --------------------------------------------------------------------------------------

@dataclass
class ShuttleState:
    """Minimal shuttle state. Later this belongs in world/state.py."""

    position_xyz: np.ndarray
    velocity_xyz: np.ndarray
    flight_phase: str = "free_flight"

    def as_vector(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.position_xyz, float), np.asarray(self.velocity_xyz, float)])


def make_launch_state(
    position_xyz,
    speed: float,
    elevation_deg: float,
    azimuth_deg: float = 0.0,
) -> ShuttleState:
    """Create a launch state from speed and angles.

    elevation_deg: angle above the horizontal (negative = downward, e.g. a smash).
    azimuth_deg:   horizontal direction, 0 = toward +y (the net-crossing direction),
                   90 = toward +x, -90 = toward -x, 180 = toward -y.
    """
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    vel = speed * np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    return ShuttleState(np.asarray(position_xyz, float), vel)


class EventType(str, Enum):
    NET_CROSSING = "NET_CROSSING"            # passed over (or around) the net: non-terminal
    NET_COLLISION = "NET_COLLISION"          # hit the net: terminal
    SHUTTLE_LANDING = "SHUTTLE_LANDING"      # reached the floor: terminal
    BOUNDARY_CROSSING = "BOUNDARY_CROSSING"  # left the singles footprint in the air: non-terminal
    HEIGHT_CROSSING = "HEIGHT_CROSSING"      # descended through the requested height: non-terminal
    TIMEOUT = "TIMEOUT"                      # safety cap reached: terminal


@dataclass(frozen=True)
class FlightEvent:
    type: EventType
    time: float
    position: np.ndarray
    velocity: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.type in (EventType.NET_COLLISION, EventType.SHUTTLE_LANDING, EventType.TIMEOUT)


@dataclass
class FlightResult:
    times: np.ndarray       # (N,)
    states: np.ndarray      # (N, 6): x y z vx vy vz
    events: list            # list[FlightEvent], in time order; last one is terminal

    @property
    def terminal_event(self) -> FlightEvent:
        return self.events[-1]

    @property
    def flight_time(self) -> float:
        return float(self.times[-1])

    @property
    def positions(self) -> np.ndarray:
        return self.states[:, :3]

    def events_of(self, kind: EventType) -> list:
        return [e for e in self.events if e.type == kind]

    @property
    def landed(self) -> bool:
        return self.terminal_event.type == EventType.SHUTTLE_LANDING

    @property
    def landing_xy(self) -> Optional[np.ndarray]:
        return self.terminal_event.position[:2].copy() if self.landed else None

    @property
    def in_bounds(self) -> Optional[bool]:
        return self.terminal_event.info["in_bounds"] if self.landed else None


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Detector:
    name: str
    kind: str                           # "net" | "floor" | "boundary" | "height"
    fn: Callable[[np.ndarray], float]   # event function g(state); sides are g >= 0 / g < 0
    direction: int                      # 0 = any flip, +1 = False->True (g becomes >= 0), -1 = True->False
    level: float = 0.0                  # height of a "height" detector

    def side(self, s: np.ndarray) -> bool:
        return self.fn(s) >= 0.0

    def triggered(self, prev_side: bool, new_side: bool) -> bool:
        if self.direction == 0:
            return prev_side != new_side
        if self.direction > 0:
            return (not prev_side) and new_side
        return prev_side and (not new_side)


class ShuttleFlightModel:
    """Integrates the shuttle until the first terminal event.

    Parameters
    ----------
    dt        : internal RK4 step in seconds (1/240 by default, as in the blueprint)
    max_time  : safety cap, a TIMEOUT event is emitted if reached
    wind      : optional constant wind vector (m/s); zero in the first version
    event_tol : time tolerance (s) of the event-location bisection
    """

    MAX_EVENTS = 64  # guard against pathological event loops

    def __init__(
        self,
        params: Optional[ShuttleParams] = None,
        court: Optional[CourtGeometry] = None,
        dt: float = 1.0 / 240.0,
        max_time: float = 10.0,
        wind=(0.0, 0.0, 0.0),
        event_tol: float = 1e-12,
    ) -> None:
        if dt <= 0 or max_time <= 0:
            raise ValueError("dt and max_time must be positive")
        self.params = params or ShuttleParams()
        self.court = court or CourtGeometry()
        self.dt = float(dt)
        self.max_time = float(max_time)
        self.wind = np.asarray(wind, float)
        self.event_tol = float(event_tol)
        self._k = self.params.drag_k
        self._detectors = self._build_detectors()

    # ---- dynamics --------------------------------------------------------------

    def derivative(self, s: np.ndarray) -> np.ndarray:
        v = s[3:]
        v_rel = v - self.wind
        acc = GRAVITY_VEC - self._k * np.linalg.norm(v_rel) * v_rel
        return np.concatenate([v, acc])

    def _rk4_step(self, s: np.ndarray, h: float) -> np.ndarray:
        k1 = self.derivative(s)
        k2 = self.derivative(s + 0.5 * h * k1)
        k3 = self.derivative(s + 0.5 * h * k2)
        k4 = self.derivative(s + h * k3)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- events ----------------------------------------------------------------

    def _build_detectors(self) -> list:
        c = self.court
        hw, hl = c.half_width_singles, c.half_length
        return [
            _Detector("net", "net", lambda s: s[1], 0),
            _Detector("floor", "floor", lambda s: s[2], -1),
            _Detector("x_pos", "boundary", lambda s: s[0] - hw, +1),
            _Detector("x_neg", "boundary", lambda s: -s[0] - hw, +1),
            _Detector("y_pos", "boundary", lambda s: s[1] - hl, +1),
            _Detector("y_neg", "boundary", lambda s: -s[1] - hl, +1),
        ]

    def _locate(self, s: np.ndarray, h: float, det: _Detector, prev_side: bool) -> float:
        """Bisect for the event time in (0, h]. Returns the *upper* bracket, so the state at
        that time is already on the new side of the event function (prevents re-triggering)."""
        lo, hi = 0.0, h
        while hi - lo > self.event_tol:
            mid = 0.5 * (lo + hi)
            if det.side(self._rk4_step(s, mid)) == prev_side:
                lo = mid
            else:
                hi = mid
        return hi

    def _make_event(self, det: _Detector, t: float, s: np.ndarray) -> FlightEvent:
        pos, vel = s[:3].copy(), s[3:].copy()
        c = self.court

        if det.kind == "net":
            direction = "toward +y" if vel[1] > 0 else "toward -y"
            if abs(pos[0]) > c.net_half_width:
                return FlightEvent(EventType.NET_CROSSING, t, pos, vel,
                                   {"clearance": None, "around_net": True, "direction": direction})
            clearance = pos[2] - c.net_top(pos[0])
            kind = EventType.NET_COLLISION if clearance <= 0.0 else EventType.NET_CROSSING
            return FlightEvent(kind, t, pos, vel,
                               {"clearance": float(clearance), "around_net": False, "direction": direction})

        if det.kind == "floor":
            pos[2] = 0.0  # the located state is within ~1e-12 s of the floor; snap it
            return FlightEvent(
                EventType.SHUTTLE_LANDING, t, pos, vel,
                {
                    "in_bounds": bool(c.is_in_singles(pos[0], pos[1])),
                    "margin_to_line": float(c.margin_to_line(pos[0], pos[1])),
                    "side": "+y" if pos[1] > 0 else "-y",
                },
            )

        if det.kind == "height":
            pos[2] = det.level  # located within ~1e-12 s of the crossing; snap it
            return FlightEvent(EventType.HEIGHT_CROSSING, t, pos, vel, {"height": det.level})

        return FlightEvent(EventType.BOUNDARY_CROSSING, t, pos, vel, {"plane": det.name})

    # ---- main entry point -------------------------------------------------------

    def integrate_to_event(self, initial: ShuttleState, descend_through: Optional[float] = None) -> FlightResult:
        """Integrate from `initial` until the first terminal event (or max_time).

        Non-terminal events (net crossing, boundary crossing) are recorded and integration
        continues. The final element of `result.events` is always the terminal event.

        descend_through: optional height h > 0 (m). Adds a non-terminal HEIGHT_CROSSING event
        each time the shuttle passes DOWNWARD through z = h, e.g. the opponent's contact height.
        A shuttle that starts below h does not trigger it until it has risen above h and
        come back down; one whose apex stays below h never triggers it.
        """
        dets = self._detectors
        if descend_through is not None:
            h = float(descend_through)
            if not (math.isfinite(h) and h > 0.0):
                raise ValueError("descend_through must be a finite height > 0")
            dets = dets + [_Detector("height", "height", lambda st, h=h: st[2] - h, -1, h)]
        s = self._validate(initial)
        t = 0.0
        times, states, events = [0.0], [s.copy()], []
        prev_sides = [d.side(s) for d in dets]

        while True:
            if t >= self.max_time - 1e-12:
                events.append(FlightEvent(EventType.TIMEOUT, t, s[:3].copy(), s[3:].copy(), {}))
                break

            h = min(self.dt, self.max_time - t)
            s_new = self._rk4_step(s, h)

            # earliest event inside this step (if any)
            best = None  # (event_dt, detector_index)
            for i, det in enumerate(dets):
                if det.triggered(prev_sides[i], det.side(s_new)):
                    te = self._locate(s, h, det, prev_sides[i])
                    if best is None or te < best[0]:
                        best = (te, i)

            if best is None:
                t += h
                s = s_new
                prev_sides = [d.side(s) for d in dets]
                times.append(t)
                states.append(s.copy())
                continue

            te, i = best
            s = self._rk4_step(s, te)
            t += te
            event = self._make_event(dets[i], t, s)
            if event.type == EventType.SHUTTLE_LANDING:
                s[2] = 0.0
            events.append(event)
            times.append(t)
            states.append(s.copy())
            if event.terminal:
                break
            if len(events) > self.MAX_EVENTS:
                raise RuntimeError("too many events in one flight; check geometry/parameters")
            prev_sides = [d.side(s) for d in dets]

        return FlightResult(np.asarray(times), np.asarray(states), events)

    # ---- helpers ----------------------------------------------------------------

    @staticmethod
    def _validate(initial: ShuttleState) -> np.ndarray:
        s = np.asarray(initial.as_vector(), float)
        if s.shape != (6,):
            raise ValueError("state must contain 3 position and 3 velocity components")
        if not np.all(np.isfinite(s)):
            raise ValueError("state contains NaN or inf")
        if s[2] <= 0.0:
            raise ValueError("initial height must be > 0 (shuttle must start above the floor)")
        return s


if __name__ == "__main__":
    # Quick demo: a clear from the back of the -y half.
    model = ShuttleFlightModel()
    launch = make_launch_state([0.0, -5.5, 2.2], speed=32.0, elevation_deg=38.0)
    res = model.integrate_to_event(launch)
    for e in res.events:
        print(f"t={e.time:6.3f}s  {e.type.value:18s} pos={np.round(e.position, 3)}  {e.info}")
    print(f"flight time {res.flight_time:.3f}s, terminal speed {model.params.terminal_speed:.2f} m/s")