"""Adversarial tests for the B2 v1 reward contract (pure Python, runs in .venv).

Set B2_REWARD_MODULE to the import path of reward.py in your project, e.g.
    B2_REWARD_MODULE=badminton_rl.reward PYTHONPATH=src pytest tests/test_reward.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os

import pytest

R = importlib.import_module(os.environ.get("B2_REWARD_MODULE", "badminton_rl.env.reward"))
Err = R.RewardIntegrityError
N = R.N_ACTIONS
IDS = {"episode_id": "ep-1", "episode_index": 0, "episode_seed": 7}


def mask_with(ids):
    m = [False] * N
    for i in ids:
        m[i] = True
    return m


def mask_hash(m):
    return hashlib.sha256(bytes(1 if x else 0 for x in m)).hexdigest()


def event(*, decision, counter, done=False, winner=None):
    delta = {"A": 0, "B": 0}
    reward = 0.0
    if done:
        delta = {"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}
        reward = 1.0 if winner == "A" else -1.0
    return {
        "event_type": "b2_empty_mask_fallback", "empty_mask_reason": "NATIVE_CONSTRAINT_EMPTY",
        "reason_source": "runtime_mask_evaluation", "fault_injection": None,
        **IDS, "decision_index": decision, "fallback_counter": counter,
        "policy_action_id": None, "policy_action_log_probability": None,
        "policy_selected": False, "policy_invoked": False,
        "all_false_mask_passed_to_policy": False,
        "executed_by_native_environment": True, "captured_from_actual_env_step_call": True,
        "native_environment_transitions_executed": 1, "native_validator_matching_call_count": 1,
        "native_output_record_matches_captured_action": True, "native_launch_flag": False,
        "global_rng_isolation": {"all_observed_global_rngs_unchanged": True},
        "score_before": {"A": 0, "B": 0}, "score_after": dict(delta), "score_delta": delta,
        "native_done": done, "rally_winner_from_score_delta": winner if done else None,
        "b2_rally_reward_from_score_delta": reward,
    }


def terminal_block(winner):
    delta = {"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}
    return {
        "winner_side": winner, "winner_name": "P" if winner == "A" else "O",
        "recorded_winner_name": "P" if winner == "A" else "O",
        "score_before": {"A": 0, "B": 0}, "score_after": dict(delta), "score_delta": delta,
        "b2_rally_reward": 1.0 if winner == "A" else -1.0,
    }


def step_payload(action=0, *, decision=1, terminated=False, truncated=False, winner=None,
                 events=None, unplayable=0, next_ids=(5, 6)):
    events = events or []
    internal = 1 + len(events) + unplayable
    over = terminated or truncated
    m = [False] * N if over else mask_with(next_ids)
    score = ({"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}) if terminated else {"A": 0, "B": 0}
    reward = (1.0 if winner == "A" else -1.0) if terminated else 0.0
    return {
        **IDS, "policy_action_id": action,
        "observation": {"score_ab": [score["A"], score["B"]]},
        "action_mask": m, "action_mask_sha256": mask_hash(m),
        "ready_for_policy": not over, "terminated": terminated, "truncated": truncated,
        "reward": reward, "native_reward_trace": [0.0] * internal,
        "internal_native_transitions": internal, "fallback_events": events,
        "normal_action": {
            "policy_action_id": action, "canonical_action": list(R.decode_action(action)),
            "native_validator_corrected": False, "native_output_record_matches": True,
        },
        "terminal": terminal_block(winner) if terminated else None,
        "output_rows_added": 2, "score": score,
    }


def reset_payload(*, terminated=False, winner=None, events=None, unplayable=0, ids=(0, 1, 2)):
    events = events or []
    m = [False] * N if terminated else mask_with(ids)
    score = ({"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}) if terminated else {"A": 0, "B": 0}
    internal = len(events) + unplayable
    return {
        "status": "terminal_without_policy" if terminated else "ready_for_policy", **IDS,
        "observation": {"score_ab": [score["A"], score["B"]]},
        "action_mask": m, "action_mask_sha256": mask_hash(m),
        "ready_for_policy": not terminated, "terminated": terminated, "truncated": False,
        "internal_native_transitions": internal, "native_reward_trace": [0.0] * internal,
        "fallback_events": events, "terminal": terminal_block(winner) if terminated else None,
        "no_policy_rally": terminated, "score": score,
    }


def mutated(payload, fn):
    p = copy.deepcopy(payload)
    fn(p)
    return p


def ledger(**kw):
    led = R.EpisodeRewardLedger(R.RewardConfig(**kw))
    led.begin_episode(reset_payload())
    return led


# ---------------- happy paths ----------------
def test_nonterminal_zero():
    v = ledger().record_step(0, step_payload(0))
    assert v.reward == 0.0 and not v.terminated and v.reward_source == "none"


@pytest.mark.parametrize("winner,reward", [("A", 1.0), ("B", -1.0)])
def test_terminal_rewards(winner, reward):
    v = ledger().record_step(0, step_payload(0, terminated=True, winner=winner))
    assert v.reward == reward and v.winner_side == winner and not v.terminal_via_internal_transition


def test_terminal_via_fallback_is_flagged():
    ev = [event(decision=2, counter=0, done=True, winner="B")]
    v = ledger().record_step(0, step_payload(0, terminated=True, winner="B", events=ev))
    assert v.reward == -1.0 and v.terminal_via_internal_transition and v.fallback_event_count == 1


def test_terminal_via_unplayable_state_is_flagged():
    v = ledger().record_step(0, step_payload(0, terminated=True, winner="B", unplayable=1))
    assert v.reward == -1.0 and v.unplayable_transition_count == 1


def test_nonterminal_with_fallback_event_ok():
    ev = [event(decision=2, counter=0)]
    assert ledger().record_step(0, step_payload(0, events=ev)).reward == 0.0


def test_stateless_helper_returns_float():
    assert R.validate_worker_step_reward(step_payload(0, terminated=True, winner="A")) == 1.0


# ---------------- reward forgery ----------------
@pytest.mark.parametrize("fn", [
    lambda p: p.__setitem__("reward", 1.0),
    lambda p: p.__setitem__("reward", float("nan")),
    lambda p: p.__setitem__("reward", True),
    lambda p: p.__setitem__("reward", "0"),
])
def test_forged_nonterminal_reward(fn):
    with pytest.raises(Err):
        ledger().record_step(0, mutated(step_payload(0), fn))


@pytest.mark.parametrize("fn", [
    lambda p: p.__setitem__("reward", -1.0),
    lambda p: p["terminal"].__setitem__("winner_side", "B"),
    lambda p: p["terminal"].__setitem__("b2_rally_reward", -1.0),
    lambda p: p["terminal"].__setitem__("score_delta", {"A": 1, "B": 1}),
    lambda p: p["terminal"].__setitem__("score_after", {"A": 2, "B": 0}),
    lambda p: p["terminal"].__setitem__("score_before", {"A": 1, "B": 0}),
    lambda p: p["terminal"].__setitem__("extra_bonus", 5.0),
    lambda p: p["terminal"].__setitem__("recorded_winner_name", "O"),
    lambda p: p.__setitem__("score", {"A": 0, "B": 1}),
    lambda p: p.__setitem__("observation", {"score_ab": [0, 1]}),
    lambda p: p.__setitem__("terminal", None),
])
def test_forged_terminal(fn):
    with pytest.raises(Err):
        ledger().record_step(0, mutated(step_payload(0, terminated=True, winner="A"), fn))


def test_winner_name_checked_against_config():
    led = R.EpisodeRewardLedger(R.RewardConfig(player_name="P", opponent_name="O"))
    led.begin_episode(reset_payload())
    bad = mutated(step_payload(0, terminated=True, winner="A"),
                  lambda p: p["terminal"].update(winner_name="O", recorded_winner_name="O"))
    with pytest.raises(Err):
        led.record_step(0, bad)


def test_native_rewards_never_leak():
    p = step_payload(0)
    p["native_reward_trace"] = [123.0]
    assert ledger().record_step(0, p).reward == 0.0


def test_native_trace_length_mismatch():
    p = mutated(step_payload(0), lambda q: q.__setitem__("native_reward_trace", [0.0, 0.0]))
    with pytest.raises(Err):
        ledger().record_step(0, p)


def test_terminal_flag_on_nonterminal_with_terminal_block():
    p = mutated(step_payload(0), lambda q: q.__setitem__("terminal", terminal_block("A")))
    with pytest.raises(Err):
        ledger().record_step(0, p)


def test_score_changed_without_terminal():
    p = mutated(step_payload(0), lambda q: q.__setitem__("score", {"A": 1, "B": 0}))
    with pytest.raises(Err):
        ledger().record_step(0, p)


# ---------------- episode-level ----------------
def test_step_after_terminal():
    led = ledger()
    led.record_step(0, step_payload(0, terminated=True, winner="A"))
    with pytest.raises(Err):
        led.record_step(0, step_payload(0, terminated=True, winner="A"))


def test_ledger_poisoned_after_failure():
    led = ledger()
    with pytest.raises(Err):
        led.record_step(0, mutated(step_payload(0), lambda p: p.__setitem__("reward", 1.0)))
    with pytest.raises(Err):
        led.record_step(0, step_payload(0))
    led.begin_episode(reset_payload())  # recovers on a clean reset
    assert led.record_step(0, step_payload(0)).reward == 0.0


def test_reset_terminal_gives_no_reward_and_no_steps():
    led = R.EpisodeRewardLedger()
    v = led.begin_episode(reset_payload(terminated=True, winner="A"))
    assert v.no_policy_rally and not v.ready_for_policy
    with pytest.raises(Err):
        led.record_step(0, step_payload(0, terminated=True, winner="A"))


def test_reset_must_not_carry_reward():
    with pytest.raises(Err):
        R.EpisodeRewardLedger().begin_episode(mutated(reset_payload(), lambda p: p.__setitem__("reward", 1.0)))


def test_step_before_begin():
    with pytest.raises(Err):
        R.EpisodeRewardLedger().record_step(0, step_payload(0))


def test_different_episode_payload_rejected():
    p = mutated(step_payload(0), lambda q: q.__setitem__("episode_seed", 99))
    with pytest.raises(Err):
        ledger().record_step(0, p)


def test_masked_action_rejected():
    with pytest.raises(Err):
        ledger().record_step(500, step_payload(500))  # reset mask only enables 0,1,2


def test_echoed_action_mismatch():
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(1))


# ---------------- action attribution ----------------
@pytest.mark.parametrize("fn", [
    lambda p: p["normal_action"].__setitem__("native_validator_corrected", True),
    lambda p: p["normal_action"].__setitem__("native_output_record_matches", False),
    lambda p: p["normal_action"].__setitem__("canonical_action", [1, 0, 0]),
])
def test_action_attribution(fn):
    with pytest.raises(Err):
        ledger().record_step(0, mutated(step_payload(0), fn))


@pytest.mark.parametrize("family", [3, 8, 9])
def test_unsupported_family_never_legal(family):
    a = family * 135
    led = R.EpisodeRewardLedger()
    with pytest.raises(Err):
        led.begin_episode(reset_payload(ids=(a,)))


# ---------------- masks ----------------
@pytest.mark.parametrize("fn", [
    lambda p: p.__setitem__("action_mask_sha256", "0" * 64),
    lambda p: p.__setitem__("action_mask", p["action_mask"][:-1]),
    lambda p: p["action_mask"].__setitem__(0, 1),
    lambda p: p.__setitem__("ready_for_policy", False),
])
def test_mask_corruption(fn):
    with pytest.raises(Err):
        ledger().record_step(0, mutated(step_payload(0), fn))


def test_all_false_mask_while_ready():
    p = step_payload(0)
    p["action_mask"] = [False] * N
    p["action_mask_sha256"] = mask_hash(p["action_mask"])
    with pytest.raises(Err):
        ledger().record_step(0, p)


def test_nonempty_mask_after_terminal():
    p = step_payload(0, terminated=True, winner="A")
    p["action_mask"] = mask_with([1])
    p["action_mask_sha256"] = mask_hash(p["action_mask"])
    with pytest.raises(Err):
        ledger().record_step(0, p)


# ---------------- truncation / stalling ----------------
def _run_to_decision(led, n):
    for d in range(1, n + 1):
        led.record_step(5 if d > 1 else 0, step_payload(5 if d > 1 else 0, decision=d, next_ids=(5,)))


def test_truncation_rejected_by_default():
    led = ledger(max_decisions=3)
    _run_to_decision(led, 2)
    with pytest.raises(R.TruncationRejectedError):
        led.record_step(5, step_payload(5, truncated=True))


@pytest.mark.parametrize("policy,reward", [("ZERO", 0.0), ("LOSS", -1.0)])
def test_truncation_policies(policy, reward):
    led = ledger(max_decisions=3, truncation_policy=R.TruncationPolicy[policy])
    _run_to_decision(led, 2)
    v = led.record_step(5, step_payload(5, truncated=True))
    assert v.truncated and v.reward == reward and v.reward_source == "truncation_policy"


def test_early_truncation_rejected():
    led = ledger(max_decisions=3, truncation_policy=R.TruncationPolicy.ZERO)
    with pytest.raises(Err):
        led.record_step(0, step_payload(0, truncated=True))


def test_missing_truncation_at_limit():
    led = ledger(max_decisions=2)
    led.record_step(0, step_payload(0, next_ids=(5,)))
    with pytest.raises(Err):
        led.record_step(5, step_payload(5, next_ids=(5,)))


def test_truncated_and_terminated():
    p = mutated(step_payload(0, terminated=True, winner="A"), lambda q: q.__setitem__("truncated", True))
    with pytest.raises(Err):
        ledger().record_step(0, p)


# ---------------- fallback events ----------------
@pytest.mark.parametrize("fn", [
    lambda e: e.__setitem__("policy_selected", True),
    lambda e: e.__setitem__("policy_action_id", 3),
    lambda e: e.__setitem__("all_false_mask_passed_to_policy", True),
    lambda e: e.__setitem__("reason_source", "synthetic_fault_injection"),
    lambda e: e.__setitem__("fault_injection", {"kind": "x"}),
    lambda e: e.__setitem__("event_type", "b2_empty_mask_fallback_failed"),
    lambda e: e.__setitem__("episode_seed", 99),
    lambda e: e.__setitem__("decision_index", 9),
    lambda e: e.__setitem__("native_output_record_matches_captured_action", False),
    lambda e: e["global_rng_isolation"].__setitem__("all_observed_global_rngs_unchanged", False),
    lambda e: e.__setitem__("empty_mask_reason", "MADE_UP"),
    lambda e: e.__setitem__("native_validator_matching_call_count", 2),
])
def test_bad_fallback_event(fn):
    ev = mutated({"e": event(decision=2, counter=0)}, lambda d: fn(d["e"]))["e"]
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, events=[ev]))


def test_fault_injection_allowed_only_when_configured():
    ev = event(decision=2, counter=0)
    ev["reason_source"] = "synthetic_fault_injection"
    ev["fault_injection"] = {"kind": "force_empty_joint_action_mask"}
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, events=[ev]))
    assert ledger(allow_fault_injection=True).record_step(0, step_payload(0, events=[ev])).reward == 0.0


def test_event_claims_done_but_payload_nonterminal():
    ev = [event(decision=2, counter=0, done=True, winner="A")]
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, events=ev))


def test_event_winner_disagrees_with_terminal():
    ev = [event(decision=2, counter=0, done=True, winner="A")]
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, terminated=True, winner="B", events=ev))


def test_terminated_after_events_without_terminal_event():
    ev = [event(decision=2, counter=0)]
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, terminated=True, winner="A", events=ev))


def test_done_event_not_last():
    ev = [event(decision=2, counter=0, done=True, winner="A"), event(decision=2, counter=1)]
    with pytest.raises(Err):
        ledger().record_step(0, step_payload(0, terminated=True, winner="A", events=ev))


def test_events_exceed_internal_transitions():
    p = step_payload(0, events=[event(decision=2, counter=0)])
    p["internal_native_transitions"] = 1
    p["native_reward_trace"] = [0.0]
    with pytest.raises(Err):
        ledger().record_step(0, p)


def test_fallback_counter_must_increase():
    led = ledger()
    led.record_step(0, step_payload(0, events=[event(decision=2, counter=5)], next_ids=(5,)))
    with pytest.raises(Err):
        led.record_step(5, step_payload(5, events=[event(decision=3, counter=5)]))


def test_reset_time_fallbacks_ok_and_counters_continue():
    led = R.EpisodeRewardLedger()
    led.begin_episode(reset_payload(events=[event(decision=1, counter=0)]))
    with pytest.raises(Err):
        led.record_step(0, step_payload(0, events=[event(decision=2, counter=0)]))