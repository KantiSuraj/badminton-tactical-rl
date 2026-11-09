from __future__ import annotations

import pytest

from badminton_rl.env.rewards import (
    RewardIntegrityError,
    shared_terminal_reward,
    validate_worker_step_reward,
)


def terminal_payload(winner: str = "A") -> dict:
    delta = {"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}
    before = {"A": 0, "B": 0}
    after = dict(delta)
    reward = 1.0 if winner == "A" else -1.0
    return {
        "terminated": True,
        "truncated": False,
        "reward": reward,
        "score": after,
        "terminal": {
            "winner_side": winner,
            "score_before": before,
            "score_after": after,
            "score_delta": delta,
            "b2_rally_reward": reward,
        },
        "fallback_events": [],
        "native_reward_trace": [-999.0, 999.0],
    }


def nonterminal_payload() -> dict:
    return {
        "terminated": False,
        "truncated": False,
        "reward": 0.0,
        "terminal": None,
        "score": {"A": 0, "B": 0},
        "fallback_events": [],
        "native_reward_trace": [-1000.0, 1000.0],
    }


def test_shared_terminal_reward_table() -> None:
    assert shared_terminal_reward(terminated=False, rally_winner_side=None) == 0.0
    assert shared_terminal_reward(terminated=True, rally_winner_side="A") == 1.0
    assert shared_terminal_reward(terminated=True, rally_winner_side="B") == -1.0
    assert shared_terminal_reward(
        terminated=True, rally_winner_side="A", agent_side="B"
    ) == -1.0


def test_nonterminal_transition_is_exactly_zero_and_ignores_native_reward() -> None:
    assert validate_worker_step_reward(nonterminal_payload()) == 0.0


def test_terminal_agent_win_is_exactly_plus_one() -> None:
    assert validate_worker_step_reward(terminal_payload("A")) == 1.0


def test_terminal_opponent_win_is_exactly_minus_one() -> None:
    assert validate_worker_step_reward(terminal_payload("B")) == -1.0


def test_terminal_winner_must_match_score_delta() -> None:
    payload = terminal_payload("A")
    payload["terminal"]["score_delta"] = {"A": 0, "B": 1}
    with pytest.raises(RewardIntegrityError, match="score delta"):
        validate_worker_step_reward(payload)


def test_terminal_reward_must_match_winner() -> None:
    payload = terminal_payload("A")
    payload["reward"] = -1.0
    with pytest.raises(RewardIntegrityError, match="Worker reward"):
        validate_worker_step_reward(payload)


def test_terminal_score_after_must_match_top_level_score() -> None:
    payload = terminal_payload("A")
    payload["score"] = {"A": 0, "B": 0}
    with pytest.raises(RewardIntegrityError, match="Top-level worker score"):
        validate_worker_step_reward(payload)


def test_nonterminal_cannot_carry_reward_or_winner() -> None:
    payload = nonterminal_payload()
    payload["reward"] = 1.0
    with pytest.raises(RewardIntegrityError, match="non-terminal"):
        validate_worker_step_reward(payload)

    payload = nonterminal_payload()
    payload["terminal"] = {"winner_side": "A"}
    with pytest.raises(RewardIntegrityError, match="terminal winner"):
        validate_worker_step_reward(payload)


def test_time_limit_truncation_is_rejected_not_given_zero_reward() -> None:
    payload = nonterminal_payload()
    payload["truncated"] = True
    with pytest.raises(RewardIntegrityError, match="rejects time-limit truncation"):
        validate_worker_step_reward(payload)


def test_fallback_is_environment_generated_and_reward_is_emitted_once() -> None:
    payload = terminal_payload("A")
    payload["fallback_events"] = [
        {
            "event_type": "b2_empty_mask_fallback",
            "policy_action_id": None,
            "policy_action_log_probability": None,
            "policy_selected": False,
            "policy_invoked": False,
            "all_false_mask_passed_to_policy": False,
            "b2_rally_reward_from_score_delta": 0.0,
        },
        {
            "event_type": "b2_empty_mask_fallback",
            "policy_action_id": None,
            "policy_action_log_probability": None,
            "policy_selected": False,
            "policy_invoked": False,
            "all_false_mask_passed_to_policy": False,
            "b2_rally_reward_from_score_delta": 1.0,
        },
    ]
    # The worker terminal reward is authoritative.  Event-level reward fields
    # are cross-checks and are not summed into a second reward.
    assert validate_worker_step_reward(payload) == 1.0


def test_duplicate_terminal_fallback_reward_claims_are_rejected() -> None:
    payload = terminal_payload("A")
    event = {
        "event_type": "b2_empty_mask_fallback",
        "policy_action_id": None,
        "policy_action_log_probability": None,
        "policy_selected": False,
        "policy_invoked": False,
        "all_false_mask_passed_to_policy": False,
        "b2_rally_reward_from_score_delta": 1.0,
    }
    payload["fallback_events"] = [dict(event), dict(event)]
    with pytest.raises(RewardIntegrityError, match="More than one fallback event"):
        validate_worker_step_reward(payload)


def test_fallback_must_not_be_logged_as_policy_action() -> None:
    payload = terminal_payload("A")
    payload["fallback_events"] = [{
        "event_type": "b2_empty_mask_fallback",
        "policy_action_id": 17,
        "policy_action_log_probability": None,
        "policy_selected": False,
        "policy_invoked": False,
        "all_false_mask_passed_to_policy": False,
    }]
    with pytest.raises(RewardIntegrityError, match="policy_action_id=null"):
        validate_worker_step_reward(payload)


def test_invalid_winner_and_non_boolean_flags_fail_closed() -> None:
    with pytest.raises(RewardIntegrityError, match="verified winner"):
        shared_terminal_reward(terminated=True, rally_winner_side=None)

    payload = nonterminal_payload()
    payload["terminated"] = 0
    with pytest.raises(RewardIntegrityError, match="Booleans"):
        validate_worker_step_reward(payload)


def test_non_finite_reward_fails_closed() -> None:
    payload = nonterminal_payload()
    payload["reward"] = float("nan")
    with pytest.raises(RewardIntegrityError, match="finite"):
        validate_worker_step_reward(payload)


def test_nonterminal_score_cannot_change_before_rally_resolution() -> None:
    payload = nonterminal_payload()
    payload["score"] = {"A": 1, "B": 0}
    with pytest.raises(RewardIntegrityError, match="Non-terminal rally score changed"):
        validate_worker_step_reward(payload)


def test_terminal_score_must_start_from_neutral_baseline() -> None:
    payload = terminal_payload("A")
    payload["terminal"]["score_before"] = {"A": 2, "B": 3}
    payload["terminal"]["score_after"] = {"A": 3, "B": 3}
    payload["terminal"]["score_delta"] = {"A": 1, "B": 0}
    payload["score"] = {"A": 3, "B": 3}
    with pytest.raises(RewardIntegrityError, match="neutral 0-0 baseline"):
        validate_worker_step_reward(payload)