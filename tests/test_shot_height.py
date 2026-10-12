"""Height-aware launch solving: aim at (x, y, h) where h is where the opponent contacts the shuttle."""

import math

import numpy as np
import pytest

from badminton_rl.dynamics.shot_execution import ContactState, ShotExecutionModel, ShotType
from badminton_rl.dynamics.shuttle_flight import EventType, ShuttleFlightModel, ShuttleState

EX = ShotExecutionModel()
HEIGHTS = [0.0, 1.0, 1.5, 2.0]

CASES = [
    (ShotType.CLEAR,      (0.0, -5.0, 2.6),  (0.0, 5.5)),
    (ShotType.DROP,       (0.5, -4.5, 2.5),  (-1.0, 1.5)),
    (ShotType.LIFT,       (0.0, -1.5, 0.6),  (0.0, 5.8)),
    (ShotType.NET_SHOT,   (0.0, -1.0, 1.0),  (0.0, 1.0)),
    (ShotType.HALF_SMASH, (0.0, -3.5, 2.6),  (1.0, 4.0)),
]


def refly(res, h):
    return ShuttleFlightModel().integrate_to_event(
        ShuttleState(res.initial_position_xyz.copy(), res.initial_velocity_xyz.copy()),
        descend_through=h if h > 0 else None)


@pytest.mark.parametrize("h", HEIGHTS)
@pytest.mark.parametrize("shot,contact,target", CASES)
@pytest.mark.parametrize("verify", [True, False])
def test_shuttle_passes_over_the_target_at_the_target_height(shot, contact, target, h, verify):
    res = EX.solve_launch(ContactState(contact), shot, target, verify=verify, target_height=h)
    if not res.feasible:
        pytest.skip(f"not feasible here ({res.reason})")     # coverage asserted separately below
    truth = refly(res, h)
    assert truth.landed
    if h > 0:
        ev = truth.events_of(EventType.HEIGHT_CROSSING)[0]
        assert ev.position[2] == pytest.approx(h)
        assert np.linalg.norm(ev.position[:2] - np.array(target)) < 2e-3        # lands on target at height h
        assert ev.time == pytest.approx(res.flight_time_s, abs=1e-3)
    else:
        assert np.linalg.norm(truth.landing_xy - np.array(target)) < 2e-3
        assert truth.flight_time == pytest.approx(res.flight_time_s, abs=1e-3)
    assert res.floor_landing_xyz[:2] == pytest.approx(truth.landing_xy, abs=2e-3)
    assert res.target_height == h and res.predicted_landing_xyz[2] == pytest.approx(h)


def test_the_sweep_is_not_vacuous():
    n = sum(EX.solve_launch(ContactState(c), s, t, verify=False, target_height=h).feasible
            for s, c, t in CASES for h in HEIGHTS)
    assert n >= 15


def test_default_equals_floor_target():
    c = ContactState((0.0, -5.0, 2.6))
    a = EX.solve_launch(c, ShotType.CLEAR, (0.3, 5.0))
    b = EX.solve_launch(c, ShotType.CLEAR, (0.3, 5.0), target_height=0.0)
    assert np.array_equal(a.initial_velocity_xyz, b.initial_velocity_xyz) and a.flight_time_s == b.flight_time_s


def test_physics_of_aiming_higher_at_the_same_ground_point():
    """To pass over the same (x, y) at a greater height the shuttle must be there EARLIER in its
    descent: more range, less flight time to the target, and it lands further beyond."""
    c = ContactState((0.0, -5.0, 2.6))
    res = [EX.solve_launch(c, ShotType.CLEAR, (0.0, 5.5), target_height=h) for h in HEIGHTS]
    assert all(r.feasible for r in res)
    assert len({round(r.elevation_deg, 6) for r in res}) == 1            # same nominal elevation, so a clean comparison
    assert [r.speed for r in res] == sorted(r.speed for r in res)
    assert [r.flight_time_s for r in res] == sorted((r.flight_time_s for r in res), reverse=True)
    beyond = [r.floor_landing_xyz[1] for r in res]
    assert beyond == sorted(beyond) and beyond[0] == pytest.approx(5.5, abs=1e-3) and beyond[-1] > 5.5 + 0.1


def test_smash_that_is_a_net_collision_at_the_floor_is_fine_at_contact_height():
    """The motivating case: a steep floor target forces the smash into the net; the same shot to the
    opponent's 1.0 m contact point is feasible and clears the net."""
    c = ContactState((0.0, -3.5, 2.8))
    floor = EX.solve_launch(c, ShotType.SMASH, (0.0, 3.0))
    contact = EX.solve_launch(c, ShotType.SMASH, (0.0, 3.0), target_height=1.0)
    assert not floor.feasible and floor.reason == "net_collision"
    assert contact.feasible and contact.margin_to_net >= EX.cfg.min_net_margin
    assert contact.floor_landing_xyz[1] > 3.0                  # it would land beyond the contact point


def test_rising_shuttle_ignores_the_ascending_pass():
    """A lift from 0.6 m to a 1.5 m contact point crosses 1.5 m going up first; only the descent counts."""
    res = EX.solve_launch(ContactState((0.0, -1.5, 0.6)), ShotType.LIFT, (0.0, 5.8), target_height=1.5)
    assert res.feasible
    truth = refly(res, 1.5)
    assert len(truth.events_of(EventType.HEIGHT_CROSSING)) == 1
    assert truth.events_of(EventType.HEIGHT_CROSSING)[0].velocity[2] < 0


def test_unreachable_and_invalid_heights_are_reported():
    nothing = EX.solve_launch(ContactState((0.0, -1.0, 1.0)), ShotType.NET_SHOT, (0.0, 1.0), target_height=5.0)
    assert not nothing.feasible and nothing.reason == "target_height_not_reached"
    assert np.all(np.isnan(nothing.initial_velocity_xyz)) and np.all(np.isnan(nothing.floor_landing_xyz))
    for bad in (-0.5, float("nan"), float("inf")):
        r = EX.solve_launch(ContactState((0.0, -5.0, 2.6)), ShotType.CLEAR, (0.0, 5.5), target_height=bad)
        assert not r.feasible and r.reason == "target_height_invalid"


@pytest.mark.parametrize("h", [1.0, 2.0])
def test_mirror_symmetry_with_height(h):
    a = EX.solve_launch(ContactState((1.0, -4.0, 2.6)), ShotType.CLEAR, (-0.5, 5.0), target_height=h)
    b = EX.solve_launch(ContactState((1.0, 4.0, 2.6)), ShotType.CLEAR, (-0.5, -5.0), target_height=h)
    assert a.feasible and b.feasible
    assert b.speed == pytest.approx(a.speed, abs=1e-6) and b.flight_time_s == pytest.approx(a.flight_time_s, abs=1e-6)
    assert b.initial_velocity_xyz[1] == pytest.approx(-a.initial_velocity_xyz[1], abs=1e-6)


def test_fast_and_verified_paths_agree():
    c = ContactState((0.0, -5.0, 2.6))
    for h in (1.0, 1.5, 2.0):
        f = EX.solve_launch(c, ShotType.CLEAR, (0.4, 5.0), verify=False, target_height=h)
        v = EX.solve_launch(c, ShotType.CLEAR, (0.4, 5.0), verify=True, target_height=h)
        assert f.speed == pytest.approx(v.speed) and f.flight_time_s == pytest.approx(v.flight_time_s, abs=2e-3)
        assert f.floor_landing_xyz == pytest.approx(v.floor_landing_xyz, abs=2e-3)