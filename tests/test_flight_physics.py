"""Physical-invariant tests for the shuttle flight model (Test C in the blueprint).

Strategy: compare against closed-form solutions wherever they exist
  * no drag             -> parabola
  * vertical drop, drag -> z(t) = z0 - (vt^2/g) ln cosh(g t / vt)
and check qualitative invariants (energy, monotone speed loss, convergence order, symmetry).
"""

import math

import numpy as np
import pytest

from badminton_rl.dynamics.shuttle_flight import (
    G,
    CourtGeometry,
    EventType,
    ShuttleFlightModel,
    ShuttleParams,
    ShuttleState,
    make_launch_state,
)


# ---------------------------------------------------------------- helpers

def vacuum_model(**kw):
    return ShuttleFlightModel(ShuttleParams.no_drag(), **kw)


def vacuum_landing(p0, v0):
    """Closed-form landing time and (x, y) for projectile motion without drag."""
    x0, y0, z0 = p0
    vx, vy, vz = v0
    t = (vz + math.sqrt(vz * vz + 2 * G * z0)) / G
    return t, x0 + vx * t, y0 + vy * t


def state(p, v):
    return ShuttleState(np.array(p, float), np.array(v, float))


# ---------------------------------------------------------------- analytic checks

def test_vacuum_matches_parabola():
    p0, v0 = (0.5, -3.0, 1.5), (1.0, 9.0, 7.0)
    res = vacuum_model().integrate_to_event(state(p0, v0))
    t_exp, x_exp, y_exp = vacuum_landing(p0, v0)
    assert res.landed
    assert res.flight_time == pytest.approx(t_exp, abs=1e-9)
    assert res.landing_xy == pytest.approx([x_exp, y_exp], abs=1e-8)
    assert res.terminal_event.position[2] == 0.0


def test_vacuum_conserves_mechanical_energy():
    res = vacuum_model().integrate_to_event(state((0, -3, 1.5), (0, 9, 7)))
    v2 = np.sum(res.states[:, 3:] ** 2, axis=1)
    energy = 0.5 * v2 + G * res.states[:, 2]  # per unit mass
    assert np.max(np.abs(energy - energy[0])) < 1e-9


def test_vertical_drop_with_drag_matches_closed_form():
    params = ShuttleParams.from_terminal_speed(6.8)
    vt = params.terminal_speed
    z0 = 10.0
    res = ShuttleFlightModel(params).integrate_to_event(state((0, -1, z0), (0, 0, 0)))
    t_exp = (vt / G) * math.acosh(math.exp(G * z0 / vt**2))
    assert res.flight_time == pytest.approx(t_exp, abs=1e-6)


def test_reaches_terminal_speed():
    params = ShuttleParams.from_terminal_speed(6.8)
    model = ShuttleFlightModel(params, dt=1 / 100, max_time=60.0)
    res = model.integrate_to_event(state((0, -1, 100.0), (0, 0, 0)))
    speed_at_floor = np.linalg.norm(res.terminal_event.velocity)
    assert speed_at_floor == pytest.approx(params.terminal_speed, rel=5e-3)
    # and speed never overshoots terminal speed when dropped from rest
    speeds = np.linalg.norm(res.states[:, 3:], axis=1)
    assert np.all(speeds <= params.terminal_speed * (1 + 1e-9))


# ---------------------------------------------------------------- drag behaviour

def test_drag_dissipates_speed_and_shortens_range():
    launch = state((0, -6.0, 2.0), (0, 30.0, 10.0))
    drag = ShuttleFlightModel().integrate_to_event(launch)
    vac = vacuum_model().integrate_to_event(launch)
    # with no wind the horizontal speed must decrease monotonically
    horiz = np.linalg.norm(drag.states[:, 3:5], axis=1)
    assert np.all(np.diff(horiz) < 0)
    assert drag.landing_xy[1] < vac.landing_xy[1]
    assert drag.flight_time < 2 * vac.flight_time  # sanity: finite and sensible


def test_tailwind_increases_range_headwind_decreases():
    launch = state((0, -6.0, 2.0), (0, 25.0, 12.0))
    calm = ShuttleFlightModel().integrate_to_event(launch).landing_xy[1]
    tail = ShuttleFlightModel(wind=(0, 3.0, 0)).integrate_to_event(launch).landing_xy[1]
    head = ShuttleFlightModel(wind=(0, -3.0, 0)).integrate_to_event(launch).landing_xy[1]
    assert head < calm < tail


def test_rk4_converges_at_fourth_order():
    launch = state((0.3, -6.0, 2.0), (1.0, 3.0, 8.0))  # stays on its own half, so no net event
    ref = ShuttleFlightModel(dt=1 / 4000).integrate_to_event(launch).landing_xy

    def err(dt):
        xy = ShuttleFlightModel(dt=dt).integrate_to_event(launch).landing_xy
        return np.linalg.norm(xy - ref)

    e30, e60, e120 = err(1 / 30), err(1 / 60), err(1 / 120)
    assert e30 > e60 > e120
    assert e30 / e60 > 8      # an order-4 method gives ~16; a generous bound avoids flakiness
    assert e60 / e120 > 8


def test_default_step_is_accurate():
    launch = state((0, -5.5, 2.2), (0, 25.0, 20.0))
    ref = ShuttleFlightModel(dt=1 / 4000).integrate_to_event(launch)
    got = ShuttleFlightModel().integrate_to_event(launch)
    assert np.linalg.norm(got.landing_xy - ref.landing_xy) < 1e-5   # 0.01 mm
    assert abs(got.flight_time - ref.flight_time) < 1e-6


# ---------------------------------------------------------------- floor / landing

@pytest.mark.parametrize(
    "target_xy, expected_in",
    [
        ((0.0, 5.0), True),       # well inside
        ((2.59, 6.70), True),     # exactly on the corner lines counts as in (set just inside below)
        ((0.0, 7.5), False),      # long
        ((3.2, 3.0), False),      # wide
        ((-3.2, -3.0), False),    # wide, other side
    ],
)
def test_landing_in_out_classification(target_xy, expected_in):
    # Aim a vacuum shot so that it lands at target_xy, then check the classification.
    p0 = np.array([0.0, -4.0, 1.5])
    vz = 8.0
    t = (vz + math.sqrt(vz * vz + 2 * G * p0[2])) / G
    v0 = ((target_xy[0] - p0[0]) / t, (target_xy[1] - p0[1]) / t, vz)
    # nudge the on-the-line case a hair inside to avoid depending on the last ulp
    if target_xy == (2.59, 6.70):
        v0 = (v0[0] - 1e-6, v0[1] - 1e-6, vz)
    res = vacuum_model().integrate_to_event(state(p0, v0))
    assert res.landed
    assert res.in_bounds is expected_in
    assert res.terminal_event.info["margin_to_line"] > 0 if expected_in else res.terminal_event.info["margin_to_line"] < 0
    assert res.terminal_event.info["side"] == ("+y" if target_xy[1] > 0 else "-y")


def test_landing_without_crossing_net_has_no_net_event():
    res = ShuttleFlightModel().integrate_to_event(state((0.3, -6.0, 2.0), (1.0, 3.0, 8.0)))
    assert res.landed
    assert not res.events_of(EventType.NET_CROSSING)
    assert not res.events_of(EventType.NET_COLLISION)


# ---------------------------------------------------------------- net

def test_net_clearance_gives_single_crossing_then_landing():
    res = vacuum_model().integrate_to_event(state((0, -3.0, 1.0), (0, 10.0, 6.0)))
    kinds = [e.type for e in res.events]
    assert kinds.count(EventType.NET_CROSSING) == 1
    assert kinds[-1] == EventType.SHUTTLE_LANDING
    crossing = res.events_of(EventType.NET_CROSSING)[0]
    assert crossing.position[1] == pytest.approx(0.0, abs=1e-9)
    assert crossing.time == pytest.approx(0.3, abs=1e-9)           # 3 m at 10 m/s
    assert crossing.info["clearance"] > 0
    expected_z = 1.0 + 6.0 * 0.3 - 0.5 * G * 0.3**2
    assert crossing.position[2] == pytest.approx(expected_z, abs=1e-9)


def test_net_collision_is_terminal():
    res = vacuum_model().integrate_to_event(state((0, -3.0, 1.0), (0, 10.0, 2.0)))
    assert res.terminal_event.type == EventType.NET_COLLISION
    assert not res.landed and res.landing_xy is None
    assert res.terminal_event.info["clearance"] < 0
    assert res.terminal_event.position[1] == pytest.approx(0.0, abs=1e-9)
    assert res.flight_time == pytest.approx(0.3, abs=1e-9)


def test_net_works_in_both_directions():
    res = vacuum_model().integrate_to_event(state((0, 3.0, 1.0), (0, -10.0, 6.0)))
    crossing = res.events_of(EventType.NET_CROSSING)
    assert len(crossing) == 1
    assert crossing[0].info["direction"] == "toward -y"


def test_net_height_profile_post_is_higher_than_center():
    court = CourtGeometry()
    assert court.net_top(0.0) == pytest.approx(1.524)
    assert court.net_top(3.05) == pytest.approx(1.55)
    assert court.net_top(10.0) == pytest.approx(1.55)   # clamped beyond the posts


def test_shuttle_passing_around_the_net_is_not_a_collision():
    # a low shuttle far outside the posts is not stopped by the net
    res = vacuum_model().integrate_to_event(state((4.0, -3.0, 0.8), (0, 10.0, 1.0)))
    crossing = res.events_of(EventType.NET_CROSSING)
    assert len(crossing) == 1 and crossing[0].info["around_net"] is True


# ---------------------------------------------------------------- boundary

def test_boundary_crossing_precedes_landing_for_long_shot():
    res = vacuum_model().integrate_to_event(state((0, -3.0, 1.0), (0, 10.0, 6.0)))
    kinds = [e.type for e in res.events]
    assert kinds == [EventType.NET_CROSSING, EventType.BOUNDARY_CROSSING, EventType.SHUTTLE_LANDING]
    b = res.events_of(EventType.BOUNDARY_CROSSING)[0]
    assert b.info["plane"] == "y_pos"
    assert b.position[1] == pytest.approx(6.70, abs=1e-9)
    assert res.in_bounds is False


def test_wide_shot_reports_sideline_plane():
    res = vacuum_model().integrate_to_event(state((0, -4.0, 1.5), (-2.0, 3.0, 6.0)))
    planes = [e.info["plane"] for e in res.events_of(EventType.BOUNDARY_CROSSING)]
    assert "x_neg" in planes


def test_events_are_time_ordered_and_terminal_is_last():
    res = vacuum_model().integrate_to_event(state((0, -3.0, 1.0), (0, 10.0, 6.0)))
    times = [e.time for e in res.events]
    assert times == sorted(times)
    assert res.events[-1].terminal
    assert all(not e.terminal for e in res.events[:-1])
    assert np.all(np.diff(res.times) > 0)


# ---------------------------------------------------------------- misc invariants

def test_mirror_symmetry_in_x():
    a = ShuttleFlightModel().integrate_to_event(state((1.0, -5.0, 2.0), (2.0, 25.0, 9.0)))
    b = ShuttleFlightModel().integrate_to_event(state((-1.0, -5.0, 2.0), (-2.0, 25.0, 9.0)))
    assert b.landing_xy == pytest.approx(a.landing_xy * np.array([-1, 1]), abs=1e-9)
    assert b.flight_time == pytest.approx(a.flight_time, abs=1e-9)


def test_determinism():
    launch = make_launch_state([0, -5.5, 2.2], speed=32.0, elevation_deg=38.0)
    r1 = ShuttleFlightModel().integrate_to_event(launch)
    r2 = ShuttleFlightModel().integrate_to_event(launch)
    assert np.array_equal(r1.states, r2.states)
    assert [e.type for e in r1.events] == [e.type for e in r2.events]


def test_timeout_event():
    res = ShuttleFlightModel(max_time=0.1).integrate_to_event(state((0, -5, 20.0), (0, 5, 0)))
    assert res.terminal_event.type == EventType.TIMEOUT
    assert res.flight_time == pytest.approx(0.1, abs=1e-9)


@pytest.mark.parametrize(
    "bad",
    [
        state((0, -3, 0.0), (0, 10, 5)),            # on the floor
        state((0, -3, -1.0), (0, 10, 5)),           # below the floor
        state((0, -3, float("nan")), (0, 10, 5)),
        state((0, -3, 1.0), (0, float("inf"), 5)),
    ],
)
def test_invalid_initial_state_raises(bad):
    with pytest.raises(ValueError):
        ShuttleFlightModel().integrate_to_event(bad)


def test_make_launch_state_angles():
    s = make_launch_state([0, 0, 1], speed=10.0, elevation_deg=0.0, azimuth_deg=0.0)
    assert s.velocity_xyz == pytest.approx([0, 10, 0], abs=1e-12)
    s = make_launch_state([0, 0, 1], speed=10.0, elevation_deg=0.0, azimuth_deg=90.0)
    assert s.velocity_xyz == pytest.approx([10, 0, 0], abs=1e-12)
    s = make_launch_state([0, 0, 1], speed=10.0, elevation_deg=90.0)
    assert s.velocity_xyz == pytest.approx([0, 0, 10], abs=1e-12)
    assert np.linalg.norm(make_launch_state([0, 0, 1], 33.0, -25.0, 17.0).velocity_xyz) == pytest.approx(33.0)


def test_params_validation_and_terminal_speed_roundtrip():
    with pytest.raises(ValueError):
        ShuttleParams(mass=0.0)
    with pytest.raises(ValueError):
        ShuttleParams(drag_area=-1.0)
    assert ShuttleParams.from_terminal_speed(7.2).terminal_speed == pytest.approx(7.2)
    assert ShuttleParams.no_drag().terminal_speed == math.inf


# ---------------------------------------------------------------- HEIGHT_CROSSING (descend_through)

def test_height_crossing_matches_vacuum_closed_form():
    # launched from z0 = 1.0 with vz = 6: back at z = 1.0 (descending) after t = 2 vz / g
    p0, v0 = (0.0, -3.0, 1.0), (0.0, 10.0, 6.0)
    res = vacuum_model().integrate_to_event(state(p0, v0), descend_through=1.0)
    ev = res.events_of(EventType.HEIGHT_CROSSING)
    assert len(ev) == 1
    t = 2 * 6.0 / G
    assert ev[0].time == pytest.approx(t, abs=1e-9)
    assert ev[0].position == pytest.approx([0.0, -3.0 + 10.0 * t, 1.0], abs=1e-8)
    assert ev[0].velocity[2] < 0 and not ev[0].terminal


def test_height_crossing_ignores_the_ascending_pass_when_starting_below():
    z0, vz, h = 0.5, 6.0, 1.0
    res = vacuum_model().integrate_to_event(state((0, -3.0, z0), (0, 10.0, vz)), descend_through=h)
    ev = res.events_of(EventType.HEIGHT_CROSSING)
    assert len(ev) == 1                                         # only the descending pass counts
    t_down = (vz + math.sqrt(vz * vz - 2 * G * (h - z0))) / G   # the larger root
    assert ev[0].time == pytest.approx(t_down, abs=1e-9)


def test_no_height_event_if_apex_stays_below_the_height():
    res = vacuum_model().integrate_to_event(state((0, -3.0, 1.0), (0, -3.0, 2.0)), descend_through=2.0)   # apex 1.2 m, stays on its own half
    assert not res.events_of(EventType.HEIGHT_CROSSING)
    assert res.landed


def test_height_event_is_nonterminal_ordered_and_does_not_change_the_flight():
    launch = state((0.0, -5.0, 2.6), (0.0, 28.0, 25.0))
    plain = ShuttleFlightModel().integrate_to_event(launch)
    with_h = ShuttleFlightModel().integrate_to_event(launch, descend_through=1.5)
    kinds = [e.type for e in with_h.events]
    assert kinds.count(EventType.HEIGHT_CROSSING) == 1 and kinds[-1] == EventType.SHUTTLE_LANDING
    assert [e.time for e in with_h.events] == sorted(e.time for e in with_h.events)
    assert with_h.landing_xy == pytest.approx(plain.landing_xy, abs=1e-6)
    assert with_h.flight_time == pytest.approx(plain.flight_time, abs=1e-6)


def test_higher_crossing_height_is_reached_earlier_on_the_way_down():
    launch = state((0.0, -5.0, 2.6), (0.0, 28.0, 25.0))
    times = [ShuttleFlightModel().integrate_to_event(launch, descend_through=h).events_of(
        EventType.HEIGHT_CROSSING)[0].time for h in (0.5, 1.0, 1.5, 2.0)]
    assert times == sorted(times, reverse=True)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_descend_through_raises(bad):
    with pytest.raises(ValueError):
        ShuttleFlightModel().integrate_to_event(state((0, -3, 1.0), (0, 10, 5)), descend_through=bad)