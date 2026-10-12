"""Test D from the blueprint: player constraints and reachability.

  * a stationary player cannot teleport
  * velocities and accelerations stay within the configured bounds
  * reachable / unreachable cases behave monotonically with available time
"""

import math

import numpy as np
import pytest

from badminton_rl.dynamics.player_motion import PlayerMotionModel, PlayerMotionParams, PlayerState
from badminton_rl.dynamics.reachability import ContactEnvelope, can_intercept
from badminton_rl.dynamics.shuttle_flight import (
    CourtGeometry,
    ShuttleFlightModel,
    ShuttleState,
)

PRM = PlayerMotionParams()
MODEL = PlayerMotionModel(PRM)


def player(pos=(0.0, -3.0), vel=(0.0, 0.0), side=-1, react=0.0):
    return PlayerState(pos, vel, side, react)


# ------------------------------------------------------------------ player motion

def max_distance_from_rest(t):
    """Furthest a player starting at rest can travel in t seconds under (a_max, v_max)."""
    t_acc = PRM.v_max / PRM.a_max
    if t <= t_acc:
        return 0.5 * PRM.a_max * t**2
    return 0.5 * PRM.v_max * t_acc + PRM.v_max * (t - t_acc)


def test_stationary_player_cannot_teleport():
    p = player()
    for t in (0.1, 0.3, 0.6, 1.0, 2.0):
        q = MODEL.advance(p, (0.0, -9.0), t, arrive=False)
        assert np.linalg.norm(q.position_xy - p.position_xy) <= max_distance_from_rest(t) + 1e-6
    # and with a reaction delay the bound shrinks by that delay
    q = MODEL.advance(player(react=0.2), (0.0, -9.0), 1.0, arrive=False)
    assert np.linalg.norm(q.position_xy - np.array([0.0, -3.0])) <= max_distance_from_rest(0.8) + 1e-6


def test_never_exceeds_speed_and_acceleration_limits():
    p = player(vel=(1.0, -1.0))
    dt = 0.01
    accel_cap = max(PRM.a_max, PRM.a_brake)
    for target in [(3.0, -7.0), (-3.0, -1.0), (0.0, -5.0)]:
        cur = p
        for _ in range(300):
            nxt = MODEL.advance(cur, target, dt, arrive=True)
            assert nxt.speed <= PRM.v_max + 1e-9
            assert np.linalg.norm(nxt.velocity_xy - cur.velocity_xy) / dt <= accel_cap * 1.001
            cur = nxt


def test_reaction_delay_blocks_steering():
    p = player(vel=(0.0, 0.0), react=0.2)
    q = MODEL.advance(p, (0.0, -9.0), 0.2)
    assert np.allclose(q.position_xy, p.position_xy) and q.speed == 0.0
    assert q.reaction_remaining == pytest.approx(0.0)
    # a moving player only coasts (loses speed) while reacting
    moving = player(vel=(0.0, -3.0), react=0.2)
    c = MODEL.advance(moving, (3.0, -3.0), 0.2)
    assert c.speed == pytest.approx(3.0 - PRM.coast_decel * 0.2, abs=1e-9)
    assert abs(c.velocity_xy[0]) < 1e-12   # no steering toward the +x target yet


def test_start_reaction_sets_timer():
    assert MODEL.start_reaction(player()).reaction_remaining == PRM.reaction_delay


def test_arrive_stops_at_target():
    q = MODEL.advance(player(), (1.0, -5.0), 6.0, arrive=True)
    assert np.linalg.norm(q.position_xy - np.array([1.0, -5.0])) < 0.05
    assert q.speed < 0.05


def test_cannot_cross_net_or_leave_court():
    q = MODEL.advance(player(), (0.0, +4.0), 5.0)        # target on the other side of the net
    assert q.position_xy[1] <= -PRM.net_standoff + 1e-9
    q = MODEL.advance(player(), (50.0, -50.0), 20.0)
    assert abs(q.position_xy[0]) <= MODEL.x_limit + 1e-9
    assert -q.position_xy[1] <= MODEL.y_far + 1e-9
    # same on the other half
    q = MODEL.advance(player((0.0, 3.0), side=1), (0.0, -4.0), 5.0)
    assert q.position_xy[1] >= PRM.net_standoff - 1e-9


def test_arrival_time_monotone_in_distance_and_matches_advance():
    p = player()
    dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tg = np.stack([np.zeros_like(dists), -3.0 - dists], axis=1)
    t = MODEL.arrival_times(p, tg, radius=0.2, t_max=5.0)
    assert np.all(np.diff(t) > 0)
    # arrival_times and advance share one controller: advance(t_arrive) is within radius
    for i in range(len(dists)):
        q = MODEL.advance(p, tg[i], float(t[i]), arrive=False)
        assert np.linalg.norm(q.position_xy - tg[i]) <= 0.2 + 1e-9
        q_early = MODEL.advance(p, tg[i], float(t[i]) - 2 * PRM.dt, arrive=False)
        assert np.linalg.norm(q_early.position_xy - tg[i]) > 0.2


def test_initial_velocity_direction_matters():
    tgt = np.array([[0.0, -7.0]])
    toward = MODEL.arrival_times(player(vel=(0.0, -3.0)), tgt, 0.3, 5.0)[0]
    rest = MODEL.arrival_times(player(vel=(0.0, 0.0)), tgt, 0.3, 5.0)[0]
    away = MODEL.arrival_times(player(vel=(0.0, 3.0)), tgt, 0.3, 5.0)[0]
    assert toward < rest < away


def test_reaction_delay_delays_arrival():
    tgt = np.array([[0.0, -6.0]])
    t0 = MODEL.arrival_times(player(react=0.0), tgt, 0.3, 5.0)[0]
    t1 = MODEL.arrival_times(player(react=0.3), tgt, 0.3, 5.0)[0]
    assert t1 == pytest.approx(t0 + 0.3, abs=2 * PRM.dt)


def test_overspeed_initial_velocity_is_clipped_not_trusted():
    q = MODEL.advance(player(vel=(30.0, 0.0)), (0.0, -3.0), 0.01, arrive=False)
    assert q.speed <= PRM.v_max + 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        PlayerState((0, 0), (0, 0), side=0)
    with pytest.raises(ValueError):
        PlayerState((float("nan"), 0), (0, 0), side=1)
    with pytest.raises(ValueError):
        MODEL.advance(player(), (0, -4), -1.0)
    with pytest.raises(ValueError):
        PlayerMotionParams(v_max=0.0)


def test_mirror_symmetry_between_halves():
    a = MODEL.advance(player((1.0, -3.0), (0.5, -1.0), side=-1), (-1.0, -6.0), 1.2, arrive=False)
    b = MODEL.advance(player((1.0, 3.0), (0.5, 1.0), side=1), (-1.0, 6.0), 1.2, arrive=False)
    assert b.position_xy == pytest.approx(a.position_xy * np.array([1, -1]), abs=1e-9)


# ------------------------------------------------------------------ reachability

ENV = ContactEnvelope()
FLIGHT = ShuttleFlightModel()


def incoming(land_y=-5.0):
    """A clear from the +y half that lands near y=land_y on the -y half."""
    # launch from +y back court toward -y; speed chosen by trial so it lands roughly at land_y
    s = ShuttleState(np.array([0.0, 5.5, 2.5]), np.array([0.0, -17.0, 16.0]))
    return FLIGHT.integrate_to_event(s)


def test_player_under_the_shuttle_intercepts():
    traj = incoming()
    land = traj.landing_xy
    res = can_intercept(player(tuple(land), react=PRM.reaction_delay), traj, 10.0, ENV, MODEL)
    assert res.feasible and res.reason == "ok"
    assert ENV.min_height <= res.contact_position_xyz[2] <= ENV.max_height
    assert res.contact_position_xyz[1] * -1 >= 0           # contact on the player's own half
    assert res.time_margin_s >= -1e-9
    assert np.linalg.norm(res.player_at_contact.position_xy - res.contact_position_xyz[:2]) \
        <= ENV.horizontal_radius + 1e-6


def fast_incoming():
    """A fast flat shot (0.7 s flight) that crosses the net near x = 0."""
    s = ShuttleState(np.array([0.0, 4.0, 2.8]), np.array([0.0, -40.0, -6.0]))
    return FLIGHT.integrate_to_event(s)


def test_far_player_misses_and_gets_a_reason():
    traj = fast_incoming()
    near = can_intercept(player((0.0, -3.0), react=PRM.reaction_delay), traj, 10.0, ENV, MODEL)
    assert near.feasible
    res = can_intercept(player((2.5, -3.0), react=PRM.reaction_delay), traj, 10.0, ENV, MODEL)
    assert not res.feasible
    assert res.reason == "unreachable_before_deadline"
    assert np.isnan(res.contact_time_s) and res.player_at_contact is None
    assert res.arrival_time_s > 0.0   # earliest the player could get within reach, reported for diagnostics


def test_feasibility_is_monotone_in_deadline():
    traj = incoming()
    p = player((traj.landing_xy[0] + 1.5, -4.5), react=PRM.reaction_delay)
    feas = [can_intercept(p, traj, d, ENV, MODEL).feasible for d in np.arange(0.0, traj.flight_time + 0.5, 0.05)]
    assert feas == sorted(feas)     # once True, stays True as the deadline grows
    assert feas[-1] and not feas[0]


def test_contact_time_never_exceeds_deadline_and_is_earliest():
    traj = incoming()
    p = player(tuple(traj.landing_xy), react=0.0)
    full = can_intercept(p, traj, 10.0, ENV, MODEL)
    cut = can_intercept(p, traj, full.contact_time_s, ENV, MODEL)
    assert cut.feasible and cut.contact_time_s == pytest.approx(full.contact_time_s)
    before = can_intercept(p, traj, full.contact_time_s - 0.02, ENV, MODEL)
    assert not before.feasible


def test_more_time_to_react_never_hurts_margin_ordering():
    traj = incoming()
    p0 = player((traj.landing_xy[0] + 1.0, -4.0), react=0.0)
    p1 = player((traj.landing_xy[0] + 1.0, -4.0), react=0.4)
    r0 = can_intercept(p0, traj, 10.0, ENV, MODEL)
    r1 = can_intercept(p1, traj, 10.0, ENV, MODEL)
    assert r0.feasible
    assert (not r1.feasible) or r1.contact_time_s >= r0.contact_time_s


def test_shuttle_never_on_players_side():
    traj = incoming()      # lands on the -y half
    wrong = PlayerState((0.0, 4.0), (0.0, 0.0), side=1)
    # shuttle spends its early flight on +y, but near the floor it is on -y; with a tiny
    # height band over the landing zone the +y player has no valid candidate
    env = ContactEnvelope(horizontal_radius=1.1, min_height=0.25, max_height=0.5)
    res = can_intercept(wrong, traj, 10.0, env, MODEL)
    assert not res.feasible and res.reason in ("shuttle_never_on_player_side", "shuttle_never_in_height_band")


def test_height_band_miss():
    traj = incoming()
    env = ContactEnvelope(horizontal_radius=1.1, min_height=50.0, max_height=60.0)
    res = can_intercept(player(tuple(traj.landing_xy)), traj, 10.0, env, MODEL)
    assert res.reason == "shuttle_never_in_height_band"


def test_envelope_validation():
    with pytest.raises(ValueError):
        ContactEnvelope(min_height=2.0, max_height=1.0)