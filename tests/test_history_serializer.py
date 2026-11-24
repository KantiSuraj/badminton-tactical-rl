"""Unit tests for the stateless b2_event_v1 serializer (pure Python; no NumPy/pandas)."""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.b2_trace_fixtures import A_ROWS, C_ROWS, C_STATES, a_states_list
from badminton_rl.adapters.coachai_history_serializer import (
    EVENT_SCHEMA_VERSION,
    NativeTransitionCapture,
    serialize_rally_history,
)
from badminton_rl.adapters.coachai_worker_utils import (
    WorkerFault,
    _native_to_metres,
    decode_action,
    encode_action,
)

X, Y = 2.59 / 127.5, 6.70 / 330.0
POLICY_ACTION = encode_action(0, 3, 4)  # CLEAR, landing cell 3, recovery cell 4


def cap(index, source, decision, start, end, **kw):
    return NativeTransitionCapture(index, source, decision, start, end, **kw)


def serve_events():
    """Reset phase of TRACE_C: one B serve; A's pending contact state is the boundary."""
    return serialize_rally_history(
        output_rows=C_ROWS[:1], first_output_position=0, history_states=C_STATES[:2],
        transitions=[cap(0, "SERVE", 0, 0, 1)], boundary_state=C_STATES[2],
        terminal=False, prior_ball_round=0, event_index_start=0,
    )


def step_events():
    """Step 1 of TRACE_C: A's policy Clear, then B misses; the rally ends."""
    return serialize_rally_history(
        output_rows=C_ROWS[1:], first_output_position=1, history_states=C_STATES,
        transitions=[cap(0, "POLICY", 1, 1, 3, policy_action_id=POLICY_ACTION)],
        boundary_state=None, terminal=True, prior_ball_round=1, event_index_start=1,
    )


def a_kwargs(**over):
    kw = dict(
        output_rows=A_ROWS, first_output_position=0, history_states=a_states_list(),
        transitions=[cap(0, "POLICY", 8, 0, 2, policy_action_id=POLICY_ACTION),
                     cap(1, "POLICY", 9, 2, 4, policy_action_id=POLICY_ACTION)],
        boundary_state=None, terminal=True, prior_ball_round=15, event_index_start=100,
    )
    kw.update(over)
    return kw


def c_kwargs(**over):
    kw = dict(
        output_rows=copy.deepcopy(C_ROWS[1:]), first_output_position=1,
        history_states=copy.deepcopy(C_STATES),
        transitions=[cap(0, "POLICY", 1, 1, 3, policy_action_id=POLICY_ACTION)],
        boundary_state=None, terminal=True, prior_ball_round=1, event_index_start=1,
    )
    kw.update(over)
    return kw


# ------------------------------------------------------------------ happy paths
def test_native_to_metres_matches_the_documented_example():
    x, y = _native_to_metres((127.5, 165.0))
    assert math.isclose(x, 2.59) and math.isclose(y, 3.35)


def test_serve_event_is_a_rallynet_stroke_with_placeholder_hit_as_none():
    (event,) = serve_events()
    assert event["schema_version"] == EVENT_SCHEMA_VERSION
    assert (event["event_index"], event["decision_index"], event["within_decision_index"]) == (0, 0, 0)
    assert (event["actor"], event["hitter_side"], event["controller"], event["event_type"]) == (
        "B", "B", "RALLYNET", "STROKE")
    assert event["action_performed"] is True
    assert event["contact_xy"] is None and event["field_validity"]["hit"] is False
    assert event["native_shot_id"] == 10 and event["model_shot_id"] == 2 and event["shot_family_id"] is None
    assert event["native_event_code"] is None
    assert event["policy_action_id"] is None and event["fallback_event_id"] is None
    assert math.isclose(event["landing_xy"][0], 101.36628050462721 * X)
    assert math.isclose(event["landing_xy"][1], 242.19119581952137 * Y)
    # Native serve-state shuttle (0, 0) is kept as the actual native contact-state value.
    assert event["contact_state"]["shuttle_xy"] == [0.0, 0.0]
    assert event["coordinate_frame"] == "HITTER_LOCAL" and event["coordinate_units"] == "meters"


def test_policy_stroke_then_rallynet_missed_shot():
    stroke, miss = step_events()
    assert (stroke["actor"], stroke["controller"], stroke["event_type"]) == ("A", "POLICY", "STROKE")
    assert stroke["policy_action_id"] == POLICY_ACTION
    assert stroke["policy_action_family_id"] == decode_action(POLICY_ACTION)[0] == 0
    assert (stroke["native_shot_id"], stroke["model_shot_id"], stroke["shot_family_id"]) == (2, 3, 0)
    assert stroke["recovery_xy"] == pytest.approx([-85.0 * X, -55.0 * Y], abs=1e-12)
    assert (stroke["event_index"], stroke["decision_index"], stroke["within_decision_index"]) == (1, 1, 0)

    assert (miss["actor"], miss["controller"], miss["event_type"]) == ("B", "RALLYNET", "MISSED_SHOT")
    assert miss["action_performed"] is False
    assert miss["native_event_code"] == 11
    assert miss["native_shot_id"] is None and miss["model_shot_id"] is None and miss["shot_family_id"] is None
    assert miss["contact_xy"] is None and miss["landing_xy"] is None and miss["recovery_xy"] is None
    assert miss["field_validity"] == {"hit": False, "landing": False, "recovery": False, "contact_state": True}
    assert miss["contact_state"]["shuttle_xy"] == pytest.approx([85.0 * X, -165.0 * Y], abs=1e-12)  # where the shuttle arrived
    assert (miss["event_index"], miss["within_decision_index"]) == (2, 1)


def test_out_of_court_landing_is_converted_exactly_not_clipped():
    events = serialize_rally_history(**a_kwargs())
    last = events[-1]
    assert last["landing_xy"][1] > 6.70
    assert math.isclose(last["landing_xy"][1], 372.98964500427246 * 6.70 / 330.0)
    assert math.isclose(last["landing_xy"][1], 7.5734, abs_tol=1e-3)


def test_hitter_local_frame_is_preserved_without_flipping():
    events = serialize_rally_history(**a_kwargs())
    a_event, b_event = events[0], events[1]
    assert a_event["landing_xy"] == pytest.approx([-100.5 * X, 141.0 * Y], abs=1e-12)
    assert b_event["contact_xy"] == pytest.approx([100.5 * X, -141.0 * Y], abs=1e-12)  # negation arises from the native frames
    assert (a_event["hitter_side"], b_event["hitter_side"]) == ("A", "B")


def test_shot_family_only_when_an_audited_mapping_exists():
    events = serialize_rally_history(**a_kwargs())
    # native: 4 SMASH(2), 7 NET_SHOT(7), 6 DRIVE(4), 3 PUSH(5)
    assert [e["native_shot_id"] for e in events] == [4, 7, 6, 3]
    assert [e["shot_family_id"] for e in events] == [2, 7, 4, 5]
    (serve,) = serve_events()
    assert serve["shot_family_id"] is None  # native 10 (serve long) has no canonical family


def test_fallback_stroke_has_environment_controller_and_no_policy_action():
    caps = [cap(0, "FALLBACK", 8, 0, 2, fallback_event_id="episode-0-decision-9-fallback-0"),
            cap(1, "POLICY", 9, 2, 4, policy_action_id=POLICY_ACTION)]
    events = serialize_rally_history(**a_kwargs(transitions=caps))
    fb = events[0]
    assert (fb["actor"], fb["controller"], fb["event_type"]) == ("A", "ENVIRONMENT", "FALLBACK_STROKE")
    assert fb["policy_action_id"] is None and fb["policy_action_family_id"] is None
    assert fb["fallback_event_id"] == "episode-0-decision-9-fallback-0"
    assert fb["action_performed"] is True
    assert events[1]["controller"] == "RALLYNET"
    assert [e["within_decision_index"] for e in events] == [0, 1, 0, 1]
    assert [e["decision_index"] for e in events] == [8, 8, 9, 9]
    assert [e["event_index"] for e in events] == [100, 101, 102, 103]


def test_unplayable_transition_is_an_unperformed_environment_event():
    # Synthetic (no native trace exists yet): A's contact state followed by a type-11 record.
    row = dict(C_ROWS[1])
    row.update(type=11.0, player_hit_x=0.0, player_hit_y=0.0, landing_x=0.0, landing_y=0.0,
               player_movement_x=0.0, player_movement_y=0.0)
    (event,) = serialize_rally_history(
        output_rows=[row], first_output_position=1, history_states=C_STATES[:3],
        transitions=[cap(0, "UNPLAYABLE", 1, 1, 2)], boundary_state=None, terminal=True,
        prior_ball_round=1, event_index_start=1,
    )
    assert (event["actor"], event["controller"], event["event_type"]) == ("A", "ENVIRONMENT", "UNPLAYABLE_TRANSITION")
    assert event["action_performed"] is False and event["native_event_code"] == 11
    assert event["native_shot_id"] is None and event["shot_family_id"] is None
    assert event["policy_action_id"] is None


def test_outcome_and_diagnostic_fields_never_reach_events():
    forbidden = {"violation", "getpoint_player", "constraint", "winner", "reward"}
    for event in step_events() + serialize_rally_history(**a_kwargs()):
        assert not forbidden & set(event)
        assert not any("fallback_reason" in key or "empty_mask" in key for key in event)


def test_serializer_does_not_mutate_its_inputs():
    kwargs = c_kwargs()
    snapshot = copy.deepcopy(kwargs)
    serialize_rally_history(**kwargs)
    assert kwargs == snapshot


def test_empty_delta_returns_no_events():
    assert serialize_rally_history(
        output_rows=[], first_output_position=0, history_states=[], transitions=[],
        boundary_state=None, terminal=False, prior_ball_round=0, event_index_start=0,
    ) == []


# ------------------------------------------------------------------ fail-closed
def _mut_row(index, **changes):
    def mutate(kw):
        kw["output_rows"][index].update(changes)
    return mutate


def _mut_state(index, position, value):
    def mutate(kw):
        kw["history_states"][index][position] = value
    return mutate


FAILURES = {
    "state hitter != player_location": _mut_state(2, 1, [57.0, -214.0]),
    "state opponent != opponent_location": _mut_state(2, 2, [0.0, 0.0]),
    "state previous shot != receive_type": _mut_state(2, 0, 9),
    "state shuttle != player_hit": _mut_state(2, 3, [1.0, 1.0]),
    "next shuttle != -landing": _mut_state(3, 3, [85.0, -164.0]),
    "next opponent != -movement": _mut_state(3, 2, [84.0, 55.0]),
    "next previous shot != type": _mut_state(3, 0, 5),
    "non-contiguous ball_round": _mut_row(0, ball_round=3.0),
    "NaN coordinate": _mut_row(0, landing_x=float("nan")),
    "infinite coordinate": _mut_row(0, player_hit_y=float("inf")),
    "stroke actor breaks alternation": _mut_row(0, player="B"),
    "unknown actor": _mut_row(0, player="C"),
    "non-integer shot code": _mut_row(0, type=2.5),
    "miss without zero sentinel": _mut_row(1, player_hit_x=1.0),
    "unmapped native shot id": _mut_row(0, type=42.0),
}


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_integrity_violations_are_fatal(name):
    kwargs = c_kwargs()
    FAILURES[name](kwargs)
    with pytest.raises(WorkerFault) as info:
        serialize_rally_history(**kwargs)
    assert info.value.fatal is True and info.value.code == "native_state_error"


def test_missing_output_column_is_fatal():
    kwargs = c_kwargs()
    del kwargs["output_rows"][0]["landing_x"]
    with pytest.raises(WorkerFault):
        serialize_rally_history(**kwargs)


def test_history_length_must_match_the_output_rows():
    with pytest.raises(WorkerFault):
        serialize_rally_history(**c_kwargs(history_states=C_STATES[:3]))


def test_state_zero_must_duplicate_the_serve_state():
    states = copy.deepcopy(C_STATES[:2])
    states[0][0] = 7
    with pytest.raises(WorkerFault):
        serialize_rally_history(
            output_rows=C_ROWS[:1], first_output_position=0, history_states=states,
            transitions=[cap(0, "SERVE", 0, 0, 1)], boundary_state=C_STATES[2],
            terminal=False, prior_ball_round=0, event_index_start=0,
        )


def test_unclaimed_or_gapped_output_rows_are_fatal():
    with pytest.raises(WorkerFault):  # a row no transition claims
        serialize_rally_history(**c_kwargs(transitions=[cap(0, "POLICY", 1, 1, 2, policy_action_id=1)]))
    with pytest.raises(WorkerFault):  # range does not start at the cursor
        serialize_rally_history(**c_kwargs(transitions=[cap(0, "POLICY", 1, 2, 3, policy_action_id=1)]))
    with pytest.raises(WorkerFault):  # rows but no transitions at all
        serialize_rally_history(**c_kwargs(transitions=[]))


def test_provenance_rules_are_enforced():
    with pytest.raises(WorkerFault):  # policy stroke recorded as a missed shot
        rows = copy.deepcopy(C_ROWS[1:])
        rows[0].update(type=11.0, player_hit_x=0.0, player_hit_y=0.0, landing_x=0.0, landing_y=0.0,
                       player_movement_x=0.0, player_movement_y=0.0)
        serialize_rally_history(**c_kwargs(output_rows=rows))
    with pytest.raises(WorkerFault):  # unplayable transition that is a real stroke
        serialize_rally_history(**c_kwargs(transitions=[cap(0, "UNPLAYABLE", 1, 1, 3)]))
    with pytest.raises(WorkerFault):  # fallback without an event id
        serialize_rally_history(**c_kwargs(transitions=[cap(0, "FALLBACK", 1, 1, 3)]))
    with pytest.raises(WorkerFault):  # out-of-range policy action
        serialize_rally_history(**c_kwargs(transitions=[cap(0, "POLICY", 1, 1, 3, policy_action_id=9999)]))


def test_missed_shot_in_a_non_terminal_operation_is_fatal():
    with pytest.raises(WorkerFault):
        serialize_rally_history(**c_kwargs(terminal=False, boundary_state=C_STATES[3]))


def test_non_terminal_operation_requires_a_boundary_state():
    with pytest.raises(WorkerFault):
        serialize_rally_history(
            output_rows=C_ROWS[:1], first_output_position=0, history_states=C_STATES[:2],
            transitions=[cap(0, "SERVE", 0, 0, 1)], boundary_state=None,
            terminal=False, prior_ball_round=0, event_index_start=0,
        )


def test_serializer_import_does_not_load_the_heavy_stack():
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; import badminton_rl.adapters.coachai_history_serializer; "
        "bad=[m for m in ('numpy','pandas','torch') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    env = dict(os.environ, PYTHONPATH=str(src))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0