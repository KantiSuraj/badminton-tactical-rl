"""Production-path fallback regression tests using the real CoachAI+/RallyNet worker.

These tests execute in a fresh `.venv-coachai` subprocess because the parent
`.venv` intentionally does not import CoachAI+, NumPy, pandas or PyTorch.

The geometry-empty scenario is explicitly fault-injected: the audited current
3x5/3x3 grid has not produced a natural geometry-empty context. The other two
fixtures are evaluated by the real worker mask evaluator and must classify
naturally as NATIVE_CONSTRAINT_EMPTY and B2_TAXONOMY_GAP.

Run from the project root:
    COACHAI_ROOT="$PWD/../external/CoachAI-Plus" \
    B2_WORKER_PYTHON="$PWD/.venv-coachai/bin/python" \
    PYTHONPATH=src pytest -vv -s tests/test_b2_production_fallback.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_CHILD_HARNESS = r'''
import hashlib
import json
import os
import sys
import traceback
from argparse import Namespace
from pathlib import Path

project_root = Path(os.environ["B2_TEST_PROJECT_ROOT"])
coachai_root = Path(os.environ["COACHAI_ROOT"])
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from badminton_rl.adapters.coachai_worker import (
    CoachAIWorker,
    N_ACTIONS,
    REASON_GEOMETRY_EMPTY,
    REASON_NATIVE_EMPTY,
    REASON_TAXONOMY_GAP,
)


def make_worker():
    args = Namespace(
        project_root=project_root,
        coachai_root=coachai_root,
        expected_coachai_commit="4fd05934f172eec16c74d05c2902983b64c8521e",
        player_name=os.environ.get("B2_PLAYER_NAME", "CHOU Tien Chen"),
        opponent_name=os.environ.get("B2_OPPONENT_NAME", "Kento MOMOTA"),
        device=os.environ.get("B2_DEVICE", "auto"),
        worker_seed=42,
        max_decisions=64,
        max_internal_transitions=64,
        fallback_events=Path(os.environ["B2_TEST_EVENT_PATH"]),
        max_line_bytes=1_000_000,
    )
    worker = CoachAIWorker(args)
    worker.initialize()
    return worker


def make_fixtures(worker):
    # Previously observed reachable state: previous shot 1, zone 11, bin 1.
    native_empty = (
        1,
        (6.01813166428073, -149.70839170242823),
        (-8.972772918641567, 263.10115456581116),
        (43.79595033824444, -214.00402128696442),
    )
    # Audit-confirmed native-only shot ID 5 context: previous 4, zone 5, bin 1.
    gap_hit = worker._native_zone_centre(5, opponent_frame=False)
    taxonomy_gap = (
        4,
        (gap_hit[0], gap_hit[1] + 84.0),
        (0.0, 200.0),
        gap_hit,
    )
    # Audit-confirmed mapped IDs 3 and 7 context: previous 1, zone 11, bin 2.
    # Its real grid mask is nonempty; the empty geometry mask below is injected.
    geometry_hit = worker._native_zone_centre(11, opponent_frame=False)
    geometry_candidate = (
        1,
        (geometry_hit[0], geometry_hit[1] + 30.0),
        (0.0, 200.0),
        geometry_hit,
    )
    return {
        REASON_NATIVE_EMPTY: native_empty,
        REASON_TAXONOMY_GAP: taxonomy_gap,
        REASON_GEOMETRY_EMPTY: geometry_candidate,
    }


def reset_worker_to_fixture(worker, state, *, episode_index, episode_seed):
    """Use the production reset operation to obtain a live, RallyNet-valid state.

    Do not duplicate the reset implementation in this test. The public worker
    reset owns episode fields, native score baselines, seeding, native rally
    increments, reset-time internal fallbacks, and the B-serve lifecycle.

    A B serve may legitimately terminate immediately. Retry with the next
    deterministic seed until the *production worker* reports ready_for_policy.
    Reset-time fallbacks, if any, are retained as setup events and are not
    confused with the one controlled fallback exercised by this test.
    """
    env = worker.env
    max_reset_attempts = 64
    terminal_seeds = []
    setup_fallback_event_ids = []
    accepted = None

    for attempt in range(max_reset_attempts):
        fixture_seed = int(episode_seed) + attempt
        native_rally_before = int(env.rally)
        response = worker.reset({
            "seed": fixture_seed,
            "episode_index": int(episode_index),
            "episode_id": f"production-fallback-{episode_index:04d}",
            "options": {"server": "B"},
        })

        expected_native_rally = native_rally_before + 1
        if int(env.rally) != expected_native_rally:
            raise AssertionError(
                "Production worker.reset() did not advance CoachAI's native rally "
                f"counter exactly once: before={native_rally_before}, "
                f"after={env.rally}, expected={expected_native_rally}, attempt={attempt}"
            )

        setup_events = response.get("fallback_events", [])
        if not isinstance(setup_events, list):
            raise AssertionError("Production reset returned malformed fallback_events")
        setup_fallback_event_ids.extend(
            str(item["fallback_event_id"])
            for item in setup_events
            if isinstance(item, dict) and item.get("fallback_event_id")
        )

        status = response.get("status")
        if status == "terminal_without_policy":
            terminal_seeds.append(fixture_seed)
            continue
        if status != "ready_for_policy":
            raise AssertionError(
                "Production reset returned an unexpected status while constructing "
                f"the fallback fixture: status={status!r}, response_keys={sorted(response)}"
            )
        if worker._terminated or worker._truncated or not worker._ready:
            raise AssertionError(
                "Production reset status says ready_for_policy but worker state flags "
                f"disagree: terminated={worker._terminated}, truncated={worker._truncated}, "
                f"ready={worker._ready}"
            )
        if worker._state is None or worker._info is None:
            raise AssertionError("Production reset did not initialize worker state/info")
        if bool(worker._launch):
            raise AssertionError("B-served production reset left A in launch/service state")

        history_players = list(env.info_dict.get("player", []))
        if len(history_players) < 2 or history_players[0] != history_players[1]:
            raise AssertionError(
                "Production B-serve reset did not produce RallyNet-compatible history; "
                f"player_history={history_players!r}"
            )
        if worker._policy_decisions != 0:
            raise AssertionError(
                f"A fresh production reset unexpectedly has {worker._policy_decisions} policy decisions"
            )

        accepted = {
            "seed": fixture_seed,
            "attempt": attempt,
            "native_rally_number": int(env.rally),
            "history_players": history_players,
            "setup_fallback_event_ids": setup_fallback_event_ids,
            "reset_internal_transitions": int(response.get("internal_native_transitions", 0)),
        }
        break

    if accepted is None:
        raise AssertionError(
            "Production worker.reset() did not produce a live B-served policy-ready "
            f"state after {max_reset_attempts} deterministic seeds; "
            f"terminal_reset_seeds={terminal_seeds}"
        )

    # The only state mutation in this fixture is the physical state under test.
    # Crucially, preserve env.info_dict and worker._info from the successful
    # production reset so RallyNet receives native-generated history.
    env.state = state
    worker._state = state
    worker._launch = False
    worker._ready = False
    worker._mask = None
    worker._mask_evaluation = None
    worker._terminated = False
    worker._truncated = False

    return {
        "fixture_reset_seed": accepted["seed"],
        "fixture_reset_attempt": accepted["attempt"],
        "native_rally_number": accepted["native_rally_number"],
        "terminal_reset_seeds": terminal_seeds,
        "history_players": accepted["history_players"],
        "setup_fallback_event_ids": accepted["setup_fallback_event_ids"],
        "reset_internal_transitions": accepted["reset_internal_transitions"],
    }


def run_one_reason(worker, reason, state, *, sequence_index):
    env = worker.env
    fixture_reset = reset_worker_to_fixture(
        worker,
        state,
        episode_index=70 + sequence_index,
        # These initial seeds produced live B-served starts in the validated
        # two-episode smoke run; any legitimate terminal serve is still retried.
        episode_seed=1001 + sequence_index,
    )

    real_evaluate = worker._evaluate_mask
    initial = real_evaluate(state)
    fault_injection = None
    if reason == REASON_GEOMETRY_EMPTY:
        assert initial["valid_action_count"] > 0, (
            "Expected the audited geometry fixture to have a nonempty natural grid; "
            "the test must inject the geometry-empty condition explicitly"
        )
        assert set(initial["native_allowed_shot_ids"]) & {2, 3, 4, 6, 7, 8, 9}
        injected = dict(initial)
        injected["mask"] = worker.np.zeros((N_ACTIONS,), dtype=worker.np.bool_)
        injected["valid_action_count"] = 0
        injected["mask_sha256"] = hashlib.sha256(
            injected["mask"].astype(worker.np.uint8).tobytes()
        ).hexdigest()
        injected["empty_reason"] = REASON_GEOMETRY_EMPTY
        injected["reason_source"] = "synthetic_fault_injection"
        fault_injection = {
            "kind": "force_empty_joint_action_mask",
            "natural_valid_joint_actions": int(initial["valid_action_count"]),
            "native_allowed_shot_ids": list(initial["native_allowed_shot_ids"]),
            "statement": "The natural mask was nonempty; the geometry-empty branch was deliberately injected.",
        }
        injected["fault_injection"] = fault_injection
        first_evaluation = injected
    else:
        assert initial["empty_reason"] == reason, (
            f"Fixture must classify naturally as {reason}; got {initial['empty_reason']!r} "
            f"with native IDs {initial['native_allowed_shot_ids']}"
        )
        first_evaluation = dict(initial)
        first_evaluation["reason_source"] = "runtime_mask_evaluation"

    evaluation_calls = {"count": 0, "post_fallback_stop_injected": False}

    def controlled_evaluate(current_state):
        if evaluation_calls["count"] == 0:
            evaluation_calls["count"] += 1
            return first_evaluation
        evaluation_calls["count"] += 1
        # After the one production fallback, use the real mask if it is policy-
        # ready. If it is empty, inject a single legal stop bit so this test
        # isolates exactly one fallback transition rather than recursively
        # generating unrelated fallback events.
        result = real_evaluate(current_state)
        if bool(result["mask"].any()):
            return result
        result = dict(result)
        stop_mask = worker.np.zeros((N_ACTIONS,), dtype=worker.np.bool_)
        stop_mask[0] = True
        result["mask"] = stop_mask
        result["valid_action_count"] = 1
        result["empty_reason"] = None
        result["reason_source"] = "test_harness_post_transition_stop"
        evaluation_calls["post_fallback_stop_injected"] = True
        return result

    output_rows_before = int(len(env.output))
    original_b_action = env.agent_b.action
    b_stats = {"calls": 0, "returned_action_valid": None}

    def tracked_b_action(*args, **kwargs):
        b_stats["calls"] += 1
        b_state = args[0] if args else kwargs["state"]
        action = original_b_action(*args, **kwargs)
        b_stats["returned_action_valid"] = bool(
            action is not None and env.check_out_violation(action, b_state, False) is None
        )
        return action

    env.agent_b.action = tracked_b_action
    original_constraint_random = worker.constraint_module.random
    original_validate = worker.constraint.validate_action
    original_advance_evaluate = worker._evaluate_mask
    transition_capture = {}
    original_execute_fallback = worker._execute_fallback

    def capture_execute_fallback(*args, **kwargs):
        event, transition = original_execute_fallback(*args, **kwargs)
        transition_capture["transition"] = transition
        return event, transition

    worker._evaluate_mask = controlled_evaluate
    worker._execute_fallback = capture_execute_fallback
    policy_decisions_before = worker._policy_decisions
    fallback_counter_before = worker._fallback_counter
    fallback_events = []
    native_rewards = []
    try:
        internal_transitions = worker._advance_to_policy_state(
            fallback_events=fallback_events,
            native_rewards=native_rewards,
            decision_index=3,
        )
    finally:
        worker._evaluate_mask = original_advance_evaluate
        worker._execute_fallback = original_execute_fallback
        env.agent_b.action = original_b_action

    assert len(fallback_events) == 1, f"Expected one fallback event, got {len(fallback_events)}"
    assert "transition" in transition_capture
    transition = transition_capture["transition"]
    event = fallback_events[0]
    capture = transition.validation_capture
    new_rows = env.output.iloc[output_rows_before:]
    new_sides = [str(value) for value in new_rows["player"].tolist()]

    assert new_sides.count("A") == 1 and new_sides.count("B") == 1, (
        f"Expected one native output row for A and one for B, got {new_sides}"
    )
    assert capture is not None and capture.captured_result is not None
    assert capture.matching_calls == 1, "A-side validator must be captured exactly once"
    assert b_stats["calls"] == 1, "CoachAI must invoke the real RallyNet B agent exactly once"
    expected_b_validator_calls = 1 if b_stats["returned_action_valid"] else 0
    assert capture.nonmatching_calls == expected_b_validator_calls, (
        "B-side validation must be delegated to the original native validator exactly "
        "when RallyNet returns a natively valid action"
    )
    assert worker.constraint_module.random is original_constraint_random
    assert worker.constraint.validate_action == original_validate

    assert event["event_type"] == "b2_empty_mask_fallback"
    assert event["empty_mask_reason"] == reason
    expected_source = "synthetic_fault_injection" if reason == REASON_GEOMETRY_EMPTY else "runtime_mask_evaluation"
    assert event["reason_source"] == expected_source
    if reason == REASON_GEOMETRY_EMPTY:
        assert event["fault_injection"]["kind"] == "force_empty_joint_action_mask"
        assert event["fault_injection"]["natural_valid_joint_actions"] > 0
    else:
        assert event["fault_injection"] is None

    assert event["captured_from_actual_env_step_call"] is True
    assert event["executed_by_native_environment"] is True
    assert event["native_validator_matching_call_count"] == 1
    assert event["native_validator_nonmatching_call_count"] == expected_b_validator_calls
    output_match_field = "native_output_record_matches_captured_action"
    if output_match_field not in event:
        raise AssertionError(
            "Production fallback event schema is missing the native output-match field; "
            f"expected={output_match_field!r}, actual_keys={sorted(event)}"
        )
    assert event[output_match_field] is True
    assert event["global_rng_isolation"]["all_observed_global_rngs_unchanged"] is True
    assert event["all_false_mask_passed_to_policy"] is False
    assert event["policy_action_id"] is None
    assert event["policy_action_log_probability"] is None
    assert event["policy_selected"] is False
    assert event["policy_invoked"] is False
    assert worker._policy_decisions == policy_decisions_before
    assert worker._fallback_counter == fallback_counter_before + 1
    assert internal_transitions == 1

    delta = event["score_delta"]
    if transition.done:
        assert delta in ({"A": 1, "B": 0}, {"A": 0, "B": 1})
        expected_reward = 1.0 if delta["A"] == 1 else -1.0
        assert event["rally_winner_from_score_delta"] == ("A" if delta["A"] else "B")
    else:
        assert delta == {"A": 0, "B": 0}
        expected_reward = 0.0
        assert event["rally_winner_from_score_delta"] is None
    assert event["b2_rally_reward_from_score_delta"] == expected_reward
    assert worker._terminated is bool(transition.done)
    if not transition.done:
        assert worker._ready is True and bool(worker._mask.any())
    else:
        assert worker._ready is False and not bool(worker._mask.any())

    return {
        "reason": reason,
        "fixture_reset_seed": fixture_reset["fixture_reset_seed"],
        "fixture_reset_attempt": fixture_reset["fixture_reset_attempt"],
        "native_rally_number": fixture_reset["native_rally_number"],
        "terminal_reset_seeds": fixture_reset["terminal_reset_seeds"],
        "setup_fallback_event_ids": fixture_reset["setup_fallback_event_ids"],
        "reset_internal_transitions": fixture_reset["reset_internal_transitions"],
        "initial_rallynet_history_players": fixture_reset["history_players"],
        "initial_rallynet_history_players": fixture_reset["history_players"],
        "reason_source": event["reason_source"],
        "fault_injection": event["fault_injection"],
        "event_id": event["fallback_event_id"],
        "fallback_seed": event["fallback_seed"],
        "state_signature_sha256": event["state_signature_sha256"],
        "fallback_selection": event["fallback_selection"],
        "corrected_native_action": event["corrected_native_action"],
        "native_done": event["native_done"],
        "score_delta": delta,
        "b2_reward": event["b2_rally_reward_from_score_delta"],
        "a_validator_matching_calls": capture.matching_calls,
        "b_agent_action_calls": b_stats["calls"],
        "b_action_returned_natively_valid": b_stats["returned_action_valid"],
        "b_validator_delegated_calls": capture.nonmatching_calls,
        "native_output_record_matches": bool(event[output_match_field]),
        "native_output_sides": new_sides,
        "rng_isolated": event["global_rng_isolation"]["all_observed_global_rngs_unchanged"],
        "policy_fields_separated": (
            event["policy_action_id"] is None
            and event["policy_action_log_probability"] is None
            and not event["policy_selected"]
            and not event["policy_invoked"]
            and worker._policy_decisions == policy_decisions_before
        ),
        "internal_transitions": internal_transitions,
        "post_fallback_stop_injected": evaluation_calls["post_fallback_stop_injected"],
    }


def main():
    worker = make_worker()
    try:
        health = worker.health_payload()
        assert health["ready"] is True
        assert health["coachai_commit"] == "4fd05934f172eec16c74d05c2902983b64c8521e"
        assert health["rallynet_checkpoint_sha256"] == "bed1f112d542903a5c6db898f56a0e6baf3d240abe2879041cc092805e985c7d"
        assert worker.rallynet is not None
        assert not bool(getattr(worker.rallynet, "training", True))
        assert worker.env.agent_b is worker.rallynet

        fixtures = make_fixtures(worker)
        order = [REASON_NATIVE_EMPTY, REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY]
        results = []
        for index, reason in enumerate(order):
            results.append(run_one_reason(worker, reason, fixtures[reason], sequence_index=index))

        all_event_lines = [
            json.loads(line)
            for line in Path(os.environ["B2_TEST_EVENT_PATH"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target_event_ids = {item["event_id"] for item in results}
        event_lines = [
            item for item in all_event_lines
            if item.get("fallback_event_id") in target_event_ids
        ]
        assert len(event_lines) == 3, (
            "Expected exactly the three controlled fallback events in the JSONL log; "
            f"target_ids={sorted(target_event_ids)}, matching_events={len(event_lines)}, "
            f"all_event_count={len(all_event_lines)}"
        )
        assert all(item["event_type"] == "b2_empty_mask_fallback" for item in event_lines)
        assert [item["empty_mask_reason"] for item in event_lines] == order
        assert not any(
            item.get("event_type") == "b2_empty_mask_fallback_failed"
            and item.get("fallback_event_id") in target_event_ids
            for item in all_event_lines
        )

        print(json.dumps({
            "status": "PASS",
            "runtime": {
                "python": sys.version.split()[0],
                "device": str(worker.device),
                "coachai_commit": health["coachai_commit"],
                "rallynet_checkpoint_sha256": health["rallynet_checkpoint_sha256"],
                "real_rallynet_loaded": worker.env.agent_b is worker.rallynet,
            },
            "cases": results,
            "target_event_count": len(event_lines),
            "setup_event_count": len(all_event_lines) - len(event_lines),
            "failed_target_event_count": sum(
                x.get("event_type") == "b2_empty_mask_fallback_failed"
                and x.get("fallback_event_id") in target_event_ids
                for x in all_event_lines
            ),
        }, sort_keys=True))
    finally:
        worker.close()


try:
    main()
except Exception:
    traceback.print_exc(file=sys.stderr)
    raise
'''


def _worker_runtime() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    coachai_value = os.environ.get("COACHAI_ROOT")
    python_value = os.environ.get("B2_WORKER_PYTHON")
    if not coachai_value or not python_value:
        pytest.skip("Set COACHAI_ROOT and B2_WORKER_PYTHON for real CoachAI+/RallyNet fallback integration")
    coachai_root = Path(coachai_value).expanduser().resolve()
    # Do not resolve this path: it is intentionally a venv/bin/python symlink.
    worker_python = Path(os.path.abspath(os.path.expanduser(python_value)))
    if not worker_python.is_file():
        pytest.fail(f"B2_WORKER_PYTHON does not exist: {worker_python}")
    if not coachai_root.is_dir():
        pytest.fail(f"COACHAI_ROOT does not exist: {coachai_root}")
    return project_root, coachai_root, worker_python


def _run_real_worker_harness(
    *, project_root: Path, coachai_root: Path, worker_python: Path, event_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({
        "B2_TEST_PROJECT_ROOT": str(project_root),
        "COACHAI_ROOT": str(coachai_root),
        "B2_TEST_EVENT_PATH": str(event_path),
        "B2_PLAYER_NAME": os.environ.get("B2_PLAYER_NAME", "CHOU Tien Chen"),
        "B2_OPPONENT_NAME": os.environ.get("B2_OPPONENT_NAME", "Kento MOMOTA"),
        "B2_DEVICE": os.environ.get("B2_DEVICE", "auto"),
    })
    completed = subprocess.run(
        [str(worker_python), "-c", _CHILD_HARNESS],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Real CoachAI+/RallyNet production fallback harness failed.\n"
            f"returncode={completed.returncode}\nstdout:\n{completed.stdout[-6000:]}\n"
            f"stderr:\n{completed.stderr[-12000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        pytest.fail(f"Worker harness emitted no JSON. stderr:\n{completed.stderr[-8000:]}")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        pytest.fail(f"Worker harness did not end with JSON: {exc}\nstdout:\n{completed.stdout[-6000:]}")
    assert result["status"] == "PASS"
    return result


def test_real_rallynet_production_fallback_all_reasons_and_fresh_process_replay(tmp_path):
    """Exercise both production methods for all reasons, then replay in a new process.

    Native-empty and taxonomy-gap use the real action-mask evaluator and native
    constraint tables. Each fixture obtains a valid B-served RallyNet history
    through deterministic real-reset retries. Geometry-empty is explicitly synthetic fault injection:
    its unmodified real mask is required to be nonempty before it is overridden.
    The real RallyNet object remains installed as env.agent_b in every case.

    The test normalizes the production event field
    `native_output_record_matches_captured_action` to the concise result field
    `native_output_record_matches`; the production schema itself is unchanged.
    """
    project_root, coachai_root, worker_python = _worker_runtime()

    first = _run_real_worker_harness(
        project_root=project_root,
        coachai_root=coachai_root,
        worker_python=worker_python,
        event_path=tmp_path / "fallback_events_run_1.jsonl",
    )
    second = _run_real_worker_harness(
        project_root=project_root,
        coachai_root=coachai_root,
        worker_python=worker_python,
        event_path=tmp_path / "fallback_events_run_2.jsonl",
    )

    assert first["runtime"] == second["runtime"]
    assert [case["reason"] for case in first["cases"]] == [
        "NATIVE_CONSTRAINT_EMPTY",
        "B2_TAXONOMY_GAP",
        "NO_JOINT_ACTION_AFTER_GEOMETRY",
    ]
    assert [case["reason"] for case in second["cases"]] == [
        "NATIVE_CONSTRAINT_EMPTY",
        "B2_TAXONOMY_GAP",
        "NO_JOINT_ACTION_AFTER_GEOMETRY",
    ]

    for left, right in zip(first["cases"], second["cases"]):
        # Compare event identity and the actual corrected five-field action
        # across two separately initialized worker processes.
        assert left["fixture_reset_seed"] == right["fixture_reset_seed"]
        assert left["fixture_reset_attempt"] == right["fixture_reset_attempt"]
        assert left["native_rally_number"] == right["native_rally_number"]
        assert left["terminal_reset_seeds"] == right["terminal_reset_seeds"]
        assert left["setup_fallback_event_ids"] == right["setup_fallback_event_ids"]
        assert left["reset_internal_transitions"] == right["reset_internal_transitions"]
        assert left["initial_rallynet_history_players"] == right["initial_rallynet_history_players"]
        assert left["fallback_seed"] == right["fallback_seed"]
        assert left["state_signature_sha256"] == right["state_signature_sha256"]
        assert left["fallback_selection"] == right["fallback_selection"]
        assert left["corrected_native_action"] == right["corrected_native_action"]
        assert left["a_validator_matching_calls"] == 1
        assert left["b_agent_action_calls"] == 1
        assert left["b_validator_delegated_calls"] == (
            1 if left["b_action_returned_natively_valid"] else 0
        )
        assert left["native_output_record_matches"] is True
        assert left["native_output_sides"].count("A") == 1
        assert left["native_output_sides"].count("B") == 1
        assert left["rng_isolated"] is True
        assert left["policy_fields_separated"] is True
        assert left["internal_transitions"] == 1
        assert right["a_validator_matching_calls"] == 1
        assert right["b_agent_action_calls"] == 1
        assert right["native_output_record_matches"] is True
        assert right["rng_isolated"] is True
        assert right["policy_fields_separated"] is True

    geometry = first["cases"][2]
    assert geometry["reason_source"] == "synthetic_fault_injection"
    assert geometry["fault_injection"]["natural_valid_joint_actions"] > 0
    assert geometry["fault_injection"]["kind"] == "force_empty_joint_action_mask"
    assert first["target_event_count"] == second["target_event_count"] == 3
    assert first["failed_target_event_count"] == second["failed_target_event_count"] == 0
