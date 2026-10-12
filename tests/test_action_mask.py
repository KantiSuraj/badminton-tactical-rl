"""Test E from the blueprint: every action declared valid meets the feasibility checks,
no valid target exceeds the court, and invalid solver results are reported with reasons."""

import logging

import numpy as np
import pytest

from badminton_rl.dynamics.action_feasibility import (
    N_ACTIONS,
    N_LANDING,
    N_RECOVERY,
    N_SHOTS,
    ActionFeasibility,
    CellGrid,
    FeasibilityParams,
    NoFeasibleActionError,
    decode_action,
    encode_action,
)
from badminton_rl.dynamics.player_motion import PlayerState
from badminton_rl.dynamics.shot_execution import ContactState, ShotType
from badminton_rl.dynamics.shuttle_flight import CourtGeometry, ShuttleFlightModel, ShuttleState

COURT = CourtGeometry()


@pytest.fixture(scope="module")
def feas():
    return ActionFeasibility()


@pytest.fixture(scope="module")
def scenario(feas):
    player = PlayerState((0.2, -4.4), (0.5, -1.0), side=-1)
    contact = ContactState((0.2, -4.4, 2.6))
    return player, contact, feas.valid_action_mask(player, contact)


# ------------------------------------------------------------------ codec / grid

def test_codec_roundtrip_and_ranges():
    assert N_ACTIONS == 1350 == N_SHOTS * N_LANDING * N_RECOVERY
    seen = set()
    for i in range(N_ACTIONS):
        s, l, r = decode_action(i)
        assert encode_action(int(s), l, r) == i
        seen.add((int(s), l, r))
    assert len(seen) == N_ACTIONS
    for bad in (-1, N_ACTIONS):
        with pytest.raises(ValueError):
            decode_action(bad)
    with pytest.raises(ValueError):
        encode_action(0, N_LANDING, 0)


@pytest.mark.parametrize("side", [-1, 1])
def test_cell_centres_are_inside_the_correct_halves(side):
    g = CellGrid()
    land = np.array([g.landing_centre(i, side) for i in range(N_LANDING)])
    rec = np.array([g.recovery_centre(i, side) for i in range(N_RECOVERY)])
    assert len({tuple(c) for c in land}) == N_LANDING and len({tuple(c) for c in rec}) == N_RECOVERY
    assert all(COURT.is_in_singles(x, y) for x, y in land) and all(COURT.is_in_singles(x, y) for x, y in rec)
    assert np.all(land[:, 1] * side < 0)      # landing cells are on the opponent's half
    assert np.all(rec[:, 1] * side > 0)       # recovery cells are on the player's own half


def test_jittered_target_stays_inside_its_cell():
    g = CellGrid()
    rng = np.random.default_rng(0)
    for l in range(N_LANDING):
        c = g.landing_centre(l, -1)
        for _ in range(20):
            t = g.sample_in_landing_cell(l, -1, rng)
            assert abs(t[0] - c[0]) <= COURT.half_width_singles / 3 + 1e-9
            assert abs(t[1] - c[1]) <= 0.5 * COURT.half_length / 5 + 1e-9   # half the cell depth
            assert COURT.is_in_singles(*t)


# ------------------------------------------------------------------ mask contents

def test_mask_shape_dtype_and_nonempty(scenario):
    _, _, res = scenario
    assert res.mask.shape == (N_ACTIONS,) and res.mask.dtype == bool
    assert 0 < res.n_valid < N_ACTIONS


def test_every_masked_action_has_a_reason_and_valid_ones_are_ok(scenario):
    _, _, res = scenario
    for i in range(N_ACTIONS):
        r = res.reason(i)
        assert r != ""
        assert (r == "ok") == bool(res.mask[i])
    assert sum(res.masked_summary().values()) == N_ACTIONS - res.n_valid


def test_valid_actions_really_are_feasible_in_the_full_model(scenario):
    """Independent re-check of every valid action with the authoritative flight model."""
    player, contact, res = scenario
    checked = set()
    for i in np.flatnonzero(res.mask):
        shot, landing, _ = decode_action(int(i))
        if (int(shot), landing) in checked:
            continue
        checked.add((int(shot), landing))
        launch = res.launches[(int(shot), landing)]
        assert launch.feasible
        target = CellGrid().landing_centre(landing, player.side)
        assert COURT.is_in_singles(*target)                       # no target outside the court
        truth = ShuttleFlightModel().integrate_to_event(
            ShuttleState(launch.initial_position_xyz.copy(), launch.initial_velocity_xyz.copy()))
        assert truth.landed and truth.in_bounds
        assert np.linalg.norm(truth.landing_xy - target) < 5e-3
        assert launch.margin_to_net >= 0.05 - 1e-6
    assert len(checked) >= 10


def test_shot_type_must_suit_contact_height(scenario):
    """From a high overhead contact, low-contact shots are masked; from a low contact, smashes are."""
    _, _, res = scenario
    m = res.mask.reshape(N_SHOTS, N_LANDING, N_RECOVERY)
    assert not m[int(ShotType.LIFT)].any() and not m[int(ShotType.NET_SHOT)].any()
    assert m[int(ShotType.CLEAR)].any() or m[int(ShotType.DROP)].any()


def test_low_contact_masks_overhead_shots(feas):
    player = PlayerState((0.0, -3.0), (0.0, 0.0), side=-1)
    res = feas.valid_action_mask(player, ContactState((0.0, -2.8, 0.7)))
    m = res.mask.reshape(N_SHOTS, N_LANDING, N_RECOVERY)
    assert not m[int(ShotType.SMASH)].any() and not m[int(ShotType.CLEAR)].any()
    assert m[int(ShotType.LIFT)].any()


def test_recovery_mask_depends_on_player_speed_and_time(scenario, feas):
    player, contact, base = scenario
    # more time before the opponent's return -> every previously valid action stays valid
    slow_opp = ActionFeasibility(
        executor=feas.executor, params=FeasibilityParams(opponent_turnaround_s=2.0))
    longer = slow_opp.valid_action_mask(player, contact)
    assert np.all(longer.mask[base.mask])
    assert longer.n_valid >= base.n_valid
    # almost no time -> strictly fewer valid actions
    rushed = ActionFeasibility(
        executor=feas.executor, params=FeasibilityParams(opponent_turnaround_s=0.0))
    tight = rushed.valid_action_mask(player, contact)
    assert tight.n_valid < base.n_valid
    assert np.all(base.mask[tight.mask])
    assert "recovery_unreachable_in_time" in tight.masked_summary()


def test_recovery_cells_far_from_the_player_are_masked_first(feas):
    """Player standing in the front court with a short time budget cannot reach deep cells."""
    quick = ActionFeasibility(executor=feas.executor,
                              params=FeasibilityParams(opponent_turnaround_s=0.0))
    player = PlayerState((0.0, -1.0), (0.0, 0.0), side=-1)
    res = quick.valid_action_mask(player, ContactState((0.0, -1.2, 1.0)))
    m = res.mask.reshape(N_SHOTS, N_LANDING, N_RECOVERY)
    any_ok = m.any(axis=(0, 1))                       # per recovery cell
    near = [r for r in range(N_RECOVERY) if r // 3 == 0]    # row 0 = closest to the net
    deep = [r for r in range(N_RECOVERY) if r // 3 == 2]
    assert any_ok[near].sum() >= any_ok[deep].sum()
    assert any_ok[near].any()


def test_mirror_symmetry_for_the_other_half(feas):
    a = feas.valid_action_mask(PlayerState((0.2, -4.4), (0.5, -1.0), -1), ContactState((0.2, -4.4, 2.6)))
    b = feas.valid_action_mask(PlayerState((0.2, 4.4), (0.5, 1.0), 1), ContactState((0.2, 4.4, 2.6)))
    assert np.array_equal(a.mask, b.mask)


def test_wrong_side_contact_is_masked_with_reason(feas):
    player = PlayerState((0.0, -4.0), (0.0, 0.0), side=-1)
    with pytest.raises(NoFeasibleActionError) as e:
        feas.valid_action_mask(player, ContactState((0.0, 3.0, 2.6)))
    assert "contact_not_on_player_side" in str(e.value)


def test_no_feasible_action_raises_and_logs(feas, caplog):
    player = PlayerState((0.0, -4.0), (0.0, 0.0), side=-1)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(NoFeasibleActionError):
            feas.valid_action_mask(player, ContactState((0.0, -4.0, 0.1)))   # below every family's range
    assert any("no feasible action" in r.message for r in caplog.records)


def test_cache_returns_identical_results(scenario, feas):
    player, contact, base = scenario
    again = feas.valid_action_mask(player, contact)
    assert np.array_equal(again.mask, base.mask)
    assert len(feas._cache) > 0