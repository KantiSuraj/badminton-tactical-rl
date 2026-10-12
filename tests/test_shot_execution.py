"""Launch solver tests. The authoritative check is always the full 3-D ShuttleFlightModel."""

import math

import numpy as np
import pytest

from badminton_rl.dynamics.shot_execution import (
    DEFAULT_FAMILIES,
    ContactState,
    ShotExecutionConfig,
    ShotExecutionModel,
    ShotType,
)
from badminton_rl.dynamics.shuttle_flight import (
    EventType,
    ShuttleFlightModel,
    ShuttleParams,
    make_launch_state,
)

EX = ShotExecutionModel()

# (shot, contact xyz, target xy) combinations that should be playable
FEASIBLE_CASES = [
    (ShotType.CLEAR,      (0.0, -5.0, 2.6),  (0.0, 5.5)),
    (ShotType.DROP,       (0.5, -4.5, 2.5),  (-1.0, 1.5)),
    (ShotType.SMASH,      (0.0, -3.5, 2.8),  (1.0, 3.5)),
    (ShotType.LIFT,       (0.0, -1.5, 0.6),  (0.0, 5.8)),
    (ShotType.NET_SHOT,   (0.0, -1.0, 1.0),  (0.0, 1.0)),
]


def refly(res):
    """Fly a LaunchResult through the authoritative model from its own launch state."""
    v = res.initial_velocity_xyz
    from badminton_rl.dynamics.shuttle_flight import ShuttleState
    return ShuttleFlightModel().integrate_to_event(ShuttleState(res.initial_position_xyz.copy(), v.copy()))


@pytest.mark.parametrize("shot,contact,target", FEASIBLE_CASES)
@pytest.mark.parametrize("verify", [True, False])
def test_solution_lands_on_target_in_full_model(shot, contact, target, verify):
    res = EX.solve_launch(ContactState(contact), shot, target, verify=verify)
    assert res.feasible and res.reason == "ok"
    truth = refly(res)
    assert truth.landed
    assert np.linalg.norm(truth.landing_xy - np.array(target)) < 2e-3      # 2 mm
    assert truth.flight_time == pytest.approx(res.flight_time_s, abs=1e-3)
    crossing = truth.events_of(EventType.NET_CROSSING)
    assert len(crossing) == 1 and crossing[0].info["clearance"] >= EX.cfg.min_net_margin - 1e-3
    assert res.margin_to_net == pytest.approx(crossing[0].info["clearance"], abs=2e-3)
    assert res.residual < 2e-3


@pytest.mark.parametrize("shot,contact,target", FEASIBLE_CASES)
def test_solution_respects_family_ranges(shot, contact, target):
    fam = DEFAULT_FAMILIES[shot]
    res = EX.solve_launch(ContactState(contact), shot, target)
    assert fam.speed_range[0] - 1e-9 <= res.speed <= fam.speed_range[1] + 1e-9
    assert fam.elevation_range_deg[0] - 1e-9 <= res.elevation_deg <= fam.elevation_range_deg[1] + 1e-9
    assert np.linalg.norm(res.initial_velocity_xyz) == pytest.approx(res.speed)


def test_nominal_elevation_is_preferred_when_feasible():
    res = EX.solve_launch(ContactState((0.0, -5.0, 2.6)), ShotType.CLEAR, (0.0, 5.5))
    assert res.elevation_deg == pytest.approx(DEFAULT_FAMILIES[ShotType.CLEAR].nominal_elevation_deg)


def test_deterministic_and_rng_ignored_without_jitter():
    c = ContactState((0.0, -5.0, 2.6))
    a = EX.solve_launch(c, ShotType.CLEAR, (0.5, 5.0), rng=np.random.default_rng(1))
    b = EX.solve_launch(c, ShotType.CLEAR, (0.5, 5.0), rng=np.random.default_rng(2))
    assert np.array_equal(a.initial_velocity_xyz, b.initial_velocity_xyz)


def test_jitter_is_reproducible_given_seed_and_varies_across_seeds():
    ex = ShotExecutionModel(config=ShotExecutionConfig(elevation_jitter_deg=4.0))
    c = ContactState((0.0, -5.0, 2.6))
    r1 = ex.solve_launch(c, ShotType.CLEAR, (0.0, 5.5), rng=np.random.default_rng(7))
    r2 = ex.solve_launch(c, ShotType.CLEAR, (0.0, 5.5), rng=np.random.default_rng(7))
    r3 = ex.solve_launch(c, ShotType.CLEAR, (0.0, 5.5), rng=np.random.default_rng(8))
    assert np.array_equal(r1.initial_velocity_xyz, r2.initial_velocity_xyz)
    assert r1.elevation_deg != r3.elevation_deg


@pytest.mark.parametrize(
    "shot,contact,target,reason",
    [
        (ShotType.SMASH,    (0.0, -3.0, 2.8), (0.0, 1.5),  "target_too_close_for_family_speed"),
        (ShotType.NET_SHOT, (0.0, -5.0, 1.0), (0.0, 6.0),  "target_too_far_for_family_speed"),
        (ShotType.DRIVE,    (1.0, -3.0, 1.4), (-1.5, 4.0), "net_collision"),
        (ShotType.SMASH,    (0.0, -3.5, 1.0), (0.0, 3.5),  "contact_height_outside_family_range"),
        (ShotType.CLEAR,    (0.0, -5.0, 2.6), (0.0, -3.0), "target_on_own_side"),
        (ShotType.CLEAR,    (0.0, -5.0, 2.6), (4.0, 5.0),  "target_out_of_bounds"),
        (ShotType.CLEAR,    (0.0, -5.0, 2.6), (0.0, 8.0),  "target_out_of_bounds"),
        (ShotType.CLEAR,    (0.0, -5.0, 0.0), (0.0, 5.0),  "contact_invalid"),
        (ShotType.CLEAR,    (0.0, 0.0, 2.6),  (0.0, 5.0),  "contact_invalid"),
    ],
)
def test_infeasible_shots_report_reason_and_no_fabricated_launch(shot, contact, target, reason):
    res = EX.solve_launch(ContactState(contact), shot, target)
    assert not res.feasible
    assert res.reason == reason
    assert np.all(np.isnan(res.initial_velocity_xyz)) and np.all(np.isnan(res.predicted_landing_xyz))
    assert math.isnan(res.flight_time_s)


def test_every_feasible_result_clears_net_margin_over_a_sweep():
    c = ContactState((0.0, -4.0, 2.6))
    n_ok = 0
    for shot in (ShotType.CLEAR, ShotType.DROP, ShotType.HALF_SMASH, ShotType.CUT):
        for ty in (1.0, 2.5, 4.0, 5.5):
            for tx in (-2.0, 0.0, 2.0):
                res = EX.solve_launch(c, shot, (tx, ty))
                if res.feasible:
                    n_ok += 1
                    assert res.margin_to_net >= EX.cfg.min_net_margin - 1e-6
                    assert refly(res).landed
    assert n_ok >= 15      # the sweep is not vacuous


def test_mirror_symmetry_between_halves():
    a = EX.solve_launch(ContactState((1.0, -4.0, 2.6)), ShotType.CLEAR, (-0.5, 5.0))
    b = EX.solve_launch(ContactState((1.0, 4.0, 2.6)), ShotType.CLEAR, (-0.5, -5.0))
    assert a.feasible and b.feasible
    assert b.speed == pytest.approx(a.speed, abs=1e-6)
    assert b.elevation_deg == pytest.approx(a.elevation_deg)
    assert b.flight_time_s == pytest.approx(a.flight_time_s, abs=1e-6)
    assert b.initial_velocity_xyz[1] == pytest.approx(-a.initial_velocity_xyz[1], abs=1e-6)
    assert b.initial_velocity_xyz[0] == pytest.approx(a.initial_velocity_xyz[0], abs=1e-6)


def test_more_drag_needs_more_speed_for_same_target():
    c = ContactState((0.0, -5.0, 2.6))
    light = ShotExecutionModel(ShuttleFlightModel(ShuttleParams.from_terminal_speed(8.0)))
    heavy = ShotExecutionModel(ShuttleFlightModel(ShuttleParams.from_terminal_speed(6.0)))
    a = light.solve_launch(c, ShotType.CLEAR, (0.0, 5.0))
    b = heavy.solve_launch(c, ShotType.CLEAR, (0.0, 5.0))
    assert a.feasible and b.feasible
    assert b.speed > a.speed


def test_wind_is_rejected():
    with pytest.raises(NotImplementedError):
        ShotExecutionModel(ShuttleFlightModel(wind=(0.0, 2.0, 0.0)))