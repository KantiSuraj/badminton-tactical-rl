"""Stateless conversion of verified CoachAI+ native records into b2_event_v1 events.

Design contract (docs: B2 event-history decisions 1-8):

* Executed-stroke values come from ``env.output`` rows (``type``, ``player_hit_*``,
  ``landing_*``, ``player_movement_*``).  ``info["action"]`` is never used: it is
  neither complete (no B strokes) nor reliable (the serve entry disagrees with
  what executed).
* The contact state of a row comes from ``info["state"][ball_round]``.  Index 0 is a
  duplicate of index 1 (the serve) and never becomes an event.
* Every row is joined to its state and cross-checked against its neighbours.  Any
  unexplained mismatch is a fatal integrity error; nothing is repaired or skipped.
* Coordinates stay in the native hitter-local frame (signs untouched) and are only
  converted to metres.  No clipping: finite out-of-court values are legitimate.
* Missing values are ``None`` (never a silent zero).  Sentinel handling is keyed on
  the verified native record type, never on a global ``(0, 0) -> None`` rule.
* Outcome / diagnostic columns (violation, getpoint_player, constraint) are never read.

This module owns no cursors and has no side effects.  It must not import NumPy,
pandas, PyTorch or the worker core.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, Sequence

from badminton_rl.adapters.coachai_worker_utils import (
    FAMILY_MAP,
    N_ACTIONS,
    NATIVE_TO_MODEL,
    WorkerFault,
    _native_to_metres,
    decode_action,
)

EVENT_SCHEMA_VERSION = "b2_event_v1"
COORDINATE_FRAME = "HITTER_LOCAL"
COORDINATE_UNITS = "meters"

# Same absolute tolerance the worker already uses for native float comparisons.
HISTORY_TOLERANCE = 1e-8

SERVE_BALL_ROUND = 1
MISSED_SHOT_NATIVE_CODE = 11  # native record type for "Missed shot"

SOURCE_SERVE = "SERVE"
SOURCE_POLICY = "POLICY"
SOURCE_FALLBACK = "FALLBACK"
SOURCE_UNPLAYABLE = "UNPLAYABLE"
_SOURCES = {SOURCE_SERVE, SOURCE_POLICY, SOURCE_FALLBACK, SOURCE_UNPLAYABLE}

CONTROLLER_POLICY = "POLICY"
CONTROLLER_ENVIRONMENT = "ENVIRONMENT"
CONTROLLER_RALLYNET = "RALLYNET"

EVENT_STROKE = "STROKE"
EVENT_FALLBACK_STROKE = "FALLBACK_STROKE"
EVENT_UNPLAYABLE_TRANSITION = "UNPLAYABLE_TRANSITION"
EVENT_MISSED_SHOT = "MISSED_SHOT"

# Columns of CoachAI's native output frame that the serializer reads.  Note the
# native header swaps names: ``type`` / ``receive_type`` hold the numeric IDs.
REQUIRED_OUTPUT_COLUMNS = (
    "ball_round", "player",
    "player_location_x", "player_location_y",
    "opponent_location_x", "opponent_location_y",
    "receive_type", "type",
    "player_hit_x", "player_hit_y",
    "landing_x", "landing_y",
    "player_movement_x", "player_movement_y",
)

NATIVE_TO_FAMILY: dict[int, int] = {
    int(item["native_id"]): family_id for family_id, item in FAMILY_MAP.items()
}


@dataclass(frozen=True)
class NativeTransitionCapture:
    """Provenance of one native transition: which output rows it produced and who controlled it.

    ``output_start`` / ``output_end`` are absolute positions in the (per-attempt) native
    output frame, half-open.  ``decision_index`` is the event-history epoch: 0 for the
    reset phase, n for everything caused by the n-th policy action.
    """

    transition_index: int
    source: str
    decision_index: int
    output_start: int
    output_end: int
    policy_action_id: int | None = None
    fallback_event_id: str | None = None


@dataclass(frozen=True)
class _ParsedState:
    previous_shot_id: int
    hitter: tuple[float, float]
    opponent: tuple[float, float]
    shuttle: tuple[float, float]


# ---------------------------------------------------------------- helpers
def _fail(message: str, **details: Any) -> NoReturn:
    raise WorkerFault(
        "native_state_error",
        f"B2 history integrity error: {message}",
        fatal=True,
        details=details or None,
    )


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(f"{name} is not a real number", value_type=type(value).__name__)
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} is not finite", value=repr(value))
    return result


def _integer(value: Any, name: str) -> int:
    number = _real(value, name)
    if not number.is_integer():
        _fail(f"{name} is not integer-valued", value=repr(value))
    return int(number)


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        items = tuple(value)
    except TypeError:
        _fail(f"{name} is not a 2-sequence", value_type=type(value).__name__)
    if len(items) != 2:
        _fail(f"{name} must have exactly 2 coordinates", length=len(items))
    return (_real(items[0], f"{name}[0]"), _real(items[1], f"{name}[1]"))


def _get(row: Mapping[str, Any], key: str) -> Any:
    if key not in row:
        _fail(f"output row is missing column {key!r}")
    return row[key]


def _row_pair(row: Mapping[str, Any], key_x: str, key_y: str) -> tuple[float, float]:
    return (_real(_get(row, key_x), key_x), _real(_get(row, key_y), key_y))


def _close(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=HISTORY_TOLERANCE) for a, b in zip(left, right)
    )


def _neg(point: tuple[float, float]) -> tuple[float, float]:
    return (-point[0], -point[1])


def _metres(point: tuple[float, float]) -> list[float]:
    x, y = _native_to_metres(point)
    return [float(x), float(y)]


def _parse_state(state: Any, name: str) -> _ParsedState:
    if not isinstance(state, (tuple, list)) or len(state) != 4:
        _fail(f"{name} must be a 4-field native state", value_type=type(state).__name__)
    return _ParsedState(
        previous_shot_id=_integer(state[0], f"{name}[0]"),
        hitter=_pair(state[1], f"{name}[1]"),
        opponent=_pair(state[2], f"{name}[2]"),
        shuttle=_pair(state[3], f"{name}[3]"),
    )


def _states_equal(left: _ParsedState, right: _ParsedState) -> bool:
    return (
        left.previous_shot_id == right.previous_shot_id
        and _close(left.hitter, right.hitter)
        and _close(left.opponent, right.opponent)
        and _close(left.shuttle, right.shuttle)
    )


def _check_capture_coverage(
    captures: Sequence[NativeTransitionCapture], first_position: int, row_count: int
) -> list[tuple[NativeTransitionCapture, int]]:
    """Every output row must be claimed by exactly one transition, in order."""
    if not captures:
        _fail("output rows exist but no transition claims them", row_count=row_count)
    cursor = first_position
    claimed: list[tuple[NativeTransitionCapture, int]] = []
    for index, capture in enumerate(captures):
        if capture.source not in _SOURCES:
            _fail("unknown transition source", source=capture.source)
        if capture.transition_index != index:
            _fail("transition indices are not sequential", expected=index, got=capture.transition_index)
        if isinstance(capture.decision_index, bool) or not isinstance(capture.decision_index, int) or capture.decision_index < 0:
            _fail("decision_index must be a non-negative integer", value=repr(capture.decision_index))
        if capture.output_start != cursor:
            _fail(
                "transition output range does not continue the previous one",
                expected_start=cursor, got_start=capture.output_start, transition_index=index,
            )
        if capture.output_end <= capture.output_start:
            _fail("transition produced no output rows", transition_index=index)
        if capture.source == SOURCE_POLICY:
            if isinstance(capture.policy_action_id, bool) or not isinstance(capture.policy_action_id, int):
                _fail("policy transition lacks an integer policy_action_id", transition_index=index)
            if capture.fallback_event_id is not None:
                _fail("policy transition must not carry a fallback_event_id", transition_index=index)
        elif capture.source == SOURCE_FALLBACK:
            if capture.policy_action_id is not None:
                _fail("fallback transition must not carry a policy_action_id", transition_index=index)
            if not isinstance(capture.fallback_event_id, str) or not capture.fallback_event_id:
                _fail("fallback transition lacks a fallback_event_id", transition_index=index)
        else:
            if capture.policy_action_id is not None or capture.fallback_event_id is not None:
                _fail("transition carries identifiers it must not have", source=capture.source)
        claimed.extend((capture, offset) for offset in range(capture.output_end - capture.output_start))
        cursor = capture.output_end
    if cursor != first_position + row_count:
        _fail(
            "transitions do not account for every output row",
            claimed_end=cursor, rows_end=first_position + row_count,
        )
    return claimed


# ---------------------------------------------------------------- serializer
def serialize_rally_history(
    *,
    output_rows: Sequence[Mapping[str, Any]],
    first_output_position: int,
    history_states: Sequence[Any],
    transitions: Sequence[NativeTransitionCapture],
    boundary_state: Any,
    terminal: bool,
    prior_ball_round: int,
    event_index_start: int,
) -> list[dict[str, Any]]:
    """Convert one operation's new output rows into ordered, verified b2_event_v1 events.

    ``history_states`` is the full cumulative ``info["state"]``; ``prior_ball_round`` is the
    ball_round of the last already-committed row (0 at the start of an attempt).
    ``boundary_state`` is the native state at the decision boundary (A's pending contact
    state), or ``None`` when ``terminal`` is true.
    """
    rows = list(output_rows)
    captures = list(transitions)
    if not isinstance(history_states, (list, tuple)):
        _fail("native history must be a list of states", value_type=type(history_states).__name__)
    if prior_ball_round < 0 or event_index_start < 0 or first_output_position < 0:
        _fail("negative cursor passed to serializer")
    if not rows:
        if captures:
            _fail("transitions were recorded but produced no output rows")
        return []

    claimed = _check_capture_coverage(captures, int(first_output_position), len(rows))

    ball_rounds: list[int] = []
    for offset, row in enumerate(rows):
        ball_round = _integer(_get(row, "ball_round"), "ball_round")
        expected = prior_ball_round + 1 + offset
        if ball_round != expected:
            _fail("ball_round sequence is not contiguous", expected=expected, got=ball_round,
                  output_position=first_output_position + offset)
        ball_rounds.append(ball_round)
    last_ball_round = ball_rounds[-1]

    # Every output row must have exactly one history entry and vice versa
    # (state[0] is the duplicate of the serve state).
    if len(history_states) != last_ball_round + 1:
        _fail(
            "native history length does not match the output rows",
            history_length=len(history_states), expected=last_ball_round + 1,
        )
    if prior_ball_round == 0:
        first = _parse_state(history_states[0], "history[0]")
        second = _parse_state(history_states[1], "history[1]")
        if not _states_equal(first, second):
            _fail("history[0] is not a duplicate of the serve state history[1]")

    events: list[dict[str, Any]] = []
    within_counters: dict[int, int] = {}
    for offset, row in enumerate(rows):
        ball_round = ball_rounds[offset]
        capture, position_in_capture = claimed[offset]
        context = {"ball_round": ball_round, "output_position": first_output_position + offset}

        actor = _get(row, "player")
        if actor not in ("A", "B"):
            _fail("output row player must be 'A' or 'B'", player=repr(actor), **context)
        expected_actor = "B" if ball_round % 2 == 1 else "A"  # B serves; strokes alternate
        if actor != expected_actor:
            _fail("stroke actor breaks the B-serves/alternating pattern",
                  expected=expected_actor, got=actor, **context)
        if capture.source == SOURCE_SERVE:
            expected_in_capture = "B"
        else:
            expected_in_capture = "A" if position_in_capture == 0 else "B"
        if actor != expected_in_capture:
            _fail("row actor disagrees with its transition's structure",
                  source=capture.source, expected=expected_in_capture, got=actor, **context)

        native_type = _integer(_get(row, "type"), "type")
        receive_type = _integer(_get(row, "receive_type"), "receive_type")
        is_miss = native_type == MISSED_SHOT_NATIVE_CODE
        is_serve = ball_round == SERVE_BALL_ROUND

        hitter_row = _row_pair(row, "player_location_x", "player_location_y")
        opponent_row = _row_pair(row, "opponent_location_x", "opponent_location_y")
        hit_row = _row_pair(row, "player_hit_x", "player_hit_y")
        landing_row = _row_pair(row, "landing_x", "landing_y")
        movement_row = _row_pair(row, "player_movement_x", "player_movement_y")

        state = _parse_state(history_states[ball_round], f"history[{ball_round}]")
        if not _close(state.hitter, hitter_row):
            _fail("state hitter position != output player_location", **context)
        if not _close(state.opponent, opponent_row):
            _fail("state opponent position != output opponent_location", **context)
        if state.previous_shot_id != receive_type:
            _fail("state previous shot id != output receive_type",
                  state_value=state.previous_shot_id, row_value=receive_type, **context)

        # Sentinel / placeholder handling is keyed on the verified native record type.
        hit_xy: list[float] | None
        landing_xy: list[float] | None
        recovery_xy: list[float] | None
        if is_miss:
            if not (_close(hit_row, (0.0, 0.0)) and _close(landing_row, (0.0, 0.0))
                    and _close(movement_row, (0.0, 0.0))):
                _fail("Missed-shot record does not carry the verified all-zero sentinel", **context)
            hit_xy = landing_xy = recovery_xy = None
        else:
            if is_serve:
                # Verified placeholder: the serve's hit and the serve state's shuttle are both (0, 0).
                if not (_close(hit_row, (0.0, 0.0)) and _close(state.shuttle, (0.0, 0.0))):
                    _fail("serve record does not carry the verified (0, 0) placeholder", **context)
                hit_xy = None
            else:
                if not _close(state.shuttle, hit_row):
                    _fail("state shuttle position != output player_hit", **context)
                hit_xy = _metres(hit_row)
            landing_xy = _metres(landing_row)
            recovery_xy = _metres(movement_row)

        # Next-contact relations: the receiver sees this stroke's landing/recovery negated.
        if is_miss:
            if not terminal:
                _fail("Missed-shot record in a non-terminal operation", **context)
        else:
            if ball_round + 1 < len(history_states):
                next_raw = history_states[ball_round + 1]
            elif not terminal:
                if boundary_state is None:
                    _fail("non-terminal operation has no decision-boundary state", **context)
                next_raw = boundary_state
            else:
                next_raw = None
            if next_raw is not None:
                nxt = _parse_state(next_raw, f"next state after ball_round {ball_round}")
                if not _close(nxt.shuttle, _neg(landing_row)):
                    _fail("next contact shuttle != negated landing", **context)
                if not _close(nxt.opponent, _neg(movement_row)):
                    _fail("next contact opponent != negated recovery/movement", **context)
                if nxt.previous_shot_id != native_type:
                    _fail("next state previous shot id != this row's type",
                          next_value=nxt.previous_shot_id, row_value=native_type, **context)

        # Provenance.
        policy_action_id: int | None = None
        policy_action_family_id: int | None = None
        fallback_event_id: str | None = None
        if capture.source == SOURCE_SERVE or position_in_capture > 0:
            controller = CONTROLLER_RALLYNET
            event_type = EVENT_MISSED_SHOT if is_miss else EVENT_STROKE
            action_performed = not is_miss
        elif capture.source == SOURCE_POLICY:
            if is_miss:
                _fail("policy-selected stroke is recorded as a missed shot", **context)
            controller, event_type, action_performed = CONTROLLER_POLICY, EVENT_STROKE, True
            policy_action_id = int(capture.policy_action_id)  # type: ignore[arg-type]
            if not 0 <= policy_action_id < N_ACTIONS:
                _fail("policy_action_id out of range", policy_action_id=policy_action_id, **context)
            policy_action_family_id = decode_action(policy_action_id)[0]
        elif capture.source == SOURCE_FALLBACK:
            if is_miss:
                _fail("fallback stroke is recorded as a missed shot", **context)
            controller, event_type, action_performed = CONTROLLER_ENVIRONMENT, EVENT_FALLBACK_STROKE, True
            fallback_event_id = capture.fallback_event_id
        else:  # SOURCE_UNPLAYABLE
            if not is_miss:
                _fail("unplayable transition is not recorded as a missed shot", **context)
            controller, event_type, action_performed = (
                CONTROLLER_ENVIRONMENT, EVENT_UNPLAYABLE_TRANSITION, False,
            )

        # Shot identity: a missed shot is a record, not a stroke -> never a native_shot_id.
        native_shot_id: int | None
        model_shot_id: int | None
        shot_family_id: int | None
        native_event_code: int | None
        if is_miss:
            native_shot_id = model_shot_id = shot_family_id = None
            native_event_code = MISSED_SHOT_NATIVE_CODE
        else:
            native_shot_id = native_type
            model_shot_id = NATIVE_TO_MODEL.get(native_type)
            if model_shot_id is None:
                _fail("native shot id has no model-id mapping", native_shot_id=native_type, **context)
            shot_family_id = NATIVE_TO_FAMILY.get(native_type)  # None when no audited family exists
            native_event_code = None

        decision_index = capture.decision_index
        within = within_counters.get(decision_index, 0)
        within_counters[decision_index] = within + 1

        events.append({
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_index": int(event_index_start) + offset,
            "decision_index": int(decision_index),
            "within_decision_index": within,
            "ball_round": ball_round,
            "actor": actor,
            "hitter_side": actor,
            "controller": controller,
            "event_type": event_type,
            "action_performed": action_performed,
            "policy_action_id": policy_action_id,
            "policy_action_family_id": policy_action_family_id,
            "fallback_event_id": fallback_event_id,
            "native_shot_id": native_shot_id,
            "model_shot_id": model_shot_id,
            "shot_family_id": shot_family_id,
            "native_event_code": native_event_code,
            "contact_xy": hit_xy,
            "landing_xy": landing_xy,
            "recovery_xy": recovery_xy,
            "contact_state": {
                "hitter_xy": _metres(state.hitter),
                "opponent_xy": _metres(state.opponent),
                "shuttle_xy": _metres(state.shuttle),
            },
            "field_validity": {
                "hit": hit_xy is not None,
                "landing": landing_xy is not None,
                "recovery": recovery_xy is not None,
                "contact_state": True,
            },
            "coordinate_frame": COORDINATE_FRAME,
            "coordinate_units": COORDINATE_UNITS,
        })
    return events