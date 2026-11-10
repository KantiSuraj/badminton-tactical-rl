from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "RewardIntegrityError",
    "TruncationRejectedError",
    "TruncationPolicy",
    "RewardConfig",
    "RewardVerdict",
    "ResetVerdict",
    "EpisodeRewardLedger",
    "shared_terminal_reward",
    "validate_worker_step_reward",
    "validate_worker_reset_payload",
    "decode_action",
]

N_FAMILIES, N_LANDING, N_RECOVERY = 10, 15, 9
N_ACTIONS = N_FAMILIES * N_LANDING * N_RECOVERY
_FAMILY_BLOCK = N_LANDING * N_RECOVERY
SUPPORTED_FAMILY_IDS = frozenset({0, 1, 2, 4, 5, 6, 7})
UNSUPPORTED_FAMILY_IDS = frozenset({3, 8, 9})
AGENT_SIDE = "A"
FALLBACK_EVENT_TYPE = "b2_empty_mask_fallback"
RUNTIME_REASON_SOURCE = "runtime_mask_evaluation"
EMPTY_MASK_REASONS = frozenset(
    {"NATIVE_CONSTRAINT_EMPTY", "B2_TAXONOMY_GAP", "NO_JOINT_ACTION_AFTER_GEOMETRY"}
)
_ZERO_SCORE = {"A": 0, "B": 0}
_TERMINAL_CORE_KEYS = frozenset(
    {"winner_side", "score_before", "score_after", "score_delta", "b2_rally_reward"}
)
_TERMINAL_KEYS = _TERMINAL_CORE_KEYS | frozenset({"winner_name", "recorded_winner_name"})


class RewardIntegrityError(RuntimeError):
    """A worker result violates the B2 v1 reward contract."""


class TruncationRejectedError(RewardIntegrityError):
    """The worker truncated a rally and the active policy rejects truncation."""


class TruncationPolicy(Enum):
    REJECT = "reject"
    ZERO = "zero"
    LOSS = "loss"


@dataclass(frozen=True)
class RewardConfig:
    truncation_policy: TruncationPolicy = TruncationPolicy.REJECT
    max_decisions: int = 64
    player_name: str | None = None
    opponent_name: str | None = None
    require_recorded_winner: bool = False
    allow_fault_injection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.truncation_policy, TruncationPolicy):
            raise ValueError("truncation_policy must be a TruncationPolicy")
        if not _is_int(self.max_decisions) or self.max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        if (self.player_name is None) != (self.opponent_name is None):
            raise ValueError("Provide both player_name and opponent_name, or neither")
        for name in (self.player_name, self.opponent_name):
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError("player/opponent names must be non-empty strings")
        if type(self.require_recorded_winner) is not bool or type(self.allow_fault_injection) is not bool:
            raise ValueError("require_recorded_winner/allow_fault_injection must be bool")


@dataclass(frozen=True)
class RewardVerdict:
    """The only object from which an environment may take a step reward."""

    reward: float
    terminated: bool
    truncated: bool
    winner_side: str | None
    reward_source: str  # "none" | "terminal_winner" | "truncation_policy"
    terminal_via_internal_transition: bool
    fallback_event_count: int
    unplayable_transition_count: int
    internal_native_transitions: int

    def info(self) -> dict[str, Any]:
        return {
            "rally_winner_side": self.winner_side,
            "reward_source": self.reward_source,
            "terminal_via_internal_transition": self.terminal_via_internal_transition,
            "fallback_event_count": self.fallback_event_count,
            "unplayable_transition_count": self.unplayable_transition_count,
            "internal_native_transitions": self.internal_native_transitions,
        }


@dataclass(frozen=True)
class ResetVerdict:
    ready_for_policy: bool
    no_policy_rally: bool
    winner_side: str | None
    fallback_event_count: int
    unplayable_transition_count: int
    internal_native_transitions: int


# --------------------------------------------------------------------------
# Primitive validators
# --------------------------------------------------------------------------
def _fail(message: str) -> None:
    raise RewardIntegrityError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _int(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if not _is_int(value):
        _fail(f"{field} must be an integer, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        _fail(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        _fail(f"{field} must be <= {maximum}, got {result}")
    return result


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be a Boolean (flags must be Booleans), got {type(value).__name__}")
    return value


def _finite_real(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(f"{field} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _score_pair(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"A", "B"}:
        _fail(f"{field} must be a mapping with exactly A and B")
    return {side: _int(value[side], f"{field}[{side!r}]", minimum=0) for side in ("A", "B")}


def _episode_ids(payload: Mapping[str, Any]) -> tuple[str, int, int]:
    episode_id = payload.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        _fail("episode_id must be a non-empty string")
    return (
        episode_id,
        _int(payload.get("episode_index"), "episode_index", minimum=0),
        _int(payload.get("episode_seed"), "episode_seed"),
    )


def decode_action(action_id: int) -> tuple[int, int, int]:
    action_id = _int(action_id, "action_id", minimum=0, maximum=N_ACTIONS - 1)
    family, remainder = divmod(action_id, _FAMILY_BLOCK)
    landing, recovery = divmod(remainder, N_RECOVERY)
    return family, landing, recovery


def shared_terminal_reward(
    *, terminated: bool, rally_winner_side: str | None, agent_side: str = "A"
) -> float:
    """Sparse zero-sum rally reward for the controlled side; no shaping."""
    if type(terminated) is not bool:
        _fail("terminated must be a Boolean (flags must be Booleans)")
    if not isinstance(agent_side, str) or agent_side not in ("A", "B"):
        _fail(f"Unsupported agent_side: {agent_side!r}")
    if not terminated:
        if rally_winner_side is not None:
            _fail("A non-terminal transition must not declare a rally winner")
        return 0.0
    if not isinstance(rally_winner_side, str) or rally_winner_side not in ("A", "B"):
        _fail("A terminal rally requires a verified winner side ('A' or 'B')")
    return 1.0 if rally_winner_side == agent_side else -1.0


# --------------------------------------------------------------------------
# Mask / observation / action checks
# --------------------------------------------------------------------------
def _validate_mask(payload: Mapping[str, Any], *, ready: bool) -> tuple[bool, ...]:
    mask = payload.get("action_mask")
    if not isinstance(mask, list) or len(mask) != N_ACTIONS:
        _fail(f"action_mask must be a list of length {N_ACTIONS}")
    if any(type(item) is not bool for item in mask):
        _fail("action_mask must contain only Booleans")
    digest = hashlib.sha256(bytes(1 if item else 0 for item in mask)).hexdigest()
    if payload.get("action_mask_sha256") != digest:
        _fail("action_mask_sha256 does not match action_mask")
    any_true = any(mask)
    if ready and not any_true:
        _fail("Policy-ready state has an all-false action mask")
    if not ready and any_true:
        _fail("Non-policy state has a non-empty action mask")
    for family in UNSUPPORTED_FAMILY_IDS:
        if any(mask[family * _FAMILY_BLOCK:(family + 1) * _FAMILY_BLOCK]):
            _fail(f"Mask enables unsupported action family {family}")
    return tuple(mask)


def _check_observation_score(observation: Any, score: Mapping[str, int]) -> None:
    if not isinstance(observation, Mapping):
        _fail("observation must be a mapping")
    pair = observation.get("score_ab")
    if (
        not isinstance(pair, list) or len(pair) != 2
        or not all(_is_int(v) for v in pair)
        or [int(pair[0]), int(pair[1])] != [score["A"], score["B"]]
    ):
        _fail("observation.score_ab disagrees with the top-level worker score")


def _validate_normal_action(normal: Any, action_id: int) -> None:
    if not isinstance(normal, Mapping):
        _fail("normal_action must be a mapping")
    if _int(normal.get("policy_action_id"), "normal_action.policy_action_id") != action_id:
        _fail("normal_action.policy_action_id disagrees with the action taken")
    canonical = normal.get("canonical_action")
    expected = list(decode_action(action_id))
    if (
        not isinstance(canonical, list) or len(canonical) != 3
        or not all(_is_int(v) for v in canonical)
        or [int(v) for v in canonical] != expected
    ):
        _fail("normal_action.canonical_action does not decode from policy_action_id")
    if expected[0] not in SUPPORTED_FAMILY_IDS:
        _fail(f"Policy played unsupported action family {expected[0]}")
    if normal.get("native_validator_corrected") is not False:
        _fail("Native validator modified the policy action; credit would be misattributed")
    if normal.get("native_output_record_matches") is not True:
        _fail("Native output record does not match the policy action")


def _check_native_trace(payload: Mapping[str, Any], internal: int | None) -> None:
    trace = payload.get("native_reward_trace")
    if not isinstance(trace, list) or (internal is not None and len(trace) != internal):
        _fail("native_reward_trace length must equal internal_native_transitions")
    for index, value in enumerate(trace):
        _finite_real(value, field=f"native_reward_trace[{index}]")


# --------------------------------------------------------------------------
# Terminal and fallback-event validation
# --------------------------------------------------------------------------
def _validate_terminal(
    terminal: Any, config: RewardConfig, *, require_envelope: bool = True
) -> tuple[str, dict[str, int], float]:
    if not isinstance(terminal, Mapping):
        _fail("A terminal result requires terminal metadata")
    keys = set(terminal)
    if require_envelope:
        if keys != _TERMINAL_KEYS:
            _fail(f"terminal metadata keys must be exactly {sorted(_TERMINAL_KEYS)}")
    else:
        if not _TERMINAL_CORE_KEYS.issubset(keys) or not keys.issubset(_TERMINAL_KEYS):
            _fail(f"terminal metadata keys must be within {sorted(_TERMINAL_KEYS)}")
    winner = terminal.get("winner_side")
    reward = shared_terminal_reward(terminated=True, rally_winner_side=winner)
    expected_delta = {"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}
    before = _score_pair(terminal.get("score_before"), field="terminal.score_before")
    after = _score_pair(terminal.get("score_after"), field="terminal.score_after")
    delta = _score_pair(terminal.get("score_delta"), field="terminal.score_delta")
    if before != _ZERO_SCORE:
        _fail("B2 v1 rally score must start from the neutral 0-0 baseline")
    actual = {side: after[side] - before[side] for side in ("A", "B")}
    if actual != expected_delta or delta != expected_delta:
        _fail(
            "Terminal score delta must award exactly one point to the declared winner; "
            f"winner={winner!r}, actual={actual}, reported={delta}"
        )
    if _finite_real(terminal.get("b2_rally_reward"), field="terminal.b2_rally_reward") != reward:
        _fail("terminal.b2_rally_reward disagrees with the verified winner")

    if require_envelope or "winner_name" in terminal or config.player_name is not None or config.require_recorded_winner:
        winner_name = terminal.get("winner_name")
        if not isinstance(winner_name, str) or not winner_name.strip():
            _fail("terminal.winner_name must be a non-empty string")
        if config.player_name is not None:
            expected_name = config.player_name if winner == "A" else config.opponent_name
            if winner_name != expected_name:
                _fail(f"terminal.winner_name {winner_name!r} disagrees with winner side {winner!r}")
        recorded = terminal.get("recorded_winner_name")
        if recorded is None:
            if config.require_recorded_winner:
                _fail("terminal.recorded_winner_name is required but missing")
        elif recorded != winner_name:
            _fail("terminal.recorded_winner_name disagrees with winner_name")
    return winner, after, reward


def _check_event(
    index: int, event: Any, ids: tuple[str, int, int] | None, config: RewardConfig,
    expected_decision: int | None, *, require_envelope: bool = True,
) -> Mapping[str, Any]:
    p = f"fallback_events[{index}]"
    if not isinstance(event, Mapping):
        _fail(f"{p} must be an object")
    if event.get("event_type") != FALLBACK_EVENT_TYPE:
        _fail(f"{p} has unexpected event_type {event.get('event_type')!r}")

    core_required = (
        ("policy_action_id", None),
        ("policy_action_log_probability", None),
        ("policy_selected", False),
        ("policy_invoked", False),
        ("all_false_mask_passed_to_policy", False),
    )
    for key, expected in core_required:
        if key not in event or event[key] is not expected:
            suffix = f" ({key}=null)" if expected is None else ""
            _fail(f"{p}.{key} must be {expected!r}{suffix}")

    has_full_event = require_envelope or "native_done" in event or "executed_by_native_environment" in event
    if not has_full_event:
        if "b2_rally_reward_from_score_delta" in event:
            _finite_real(
                event.get("b2_rally_reward_from_score_delta"),
                field=f"{p}.b2_rally_reward_from_score_delta",
            )
        return event

    envelope_required = (
        ("executed_by_native_environment", True),
        ("captured_from_actual_env_step_call", True),
        ("native_output_record_matches_captured_action", True),
        ("native_launch_flag", False),
    )
    for key, expected in envelope_required:
        if key not in event or event[key] is not expected:
            _fail(f"{p}.{key} must be {expected!r}")
    for key in ("native_environment_transitions_executed", "native_validator_matching_call_count"):
        if _int(event.get(key), f"{p}.{key}") != 1:
            _fail(f"{p}.{key} must equal 1")
    isolation = event.get("global_rng_isolation")
    if not isinstance(isolation, Mapping) or isolation.get("all_observed_global_rngs_unchanged") is not True:
        _fail(f"{p} did not prove process-global RNG isolation")
    if event.get("empty_mask_reason") not in EMPTY_MASK_REASONS:
        _fail(f"{p} has unknown empty_mask_reason {event.get('empty_mask_reason')!r}")
    source = event.get("reason_source")
    injection = event.get("fault_injection")
    if source == RUNTIME_REASON_SOURCE:
        if injection is not None:
            _fail(f"{p} claims runtime evaluation but carries fault_injection")
    elif not config.allow_fault_injection:
        _fail(f"{p} is synthetic (reason_source={source!r}); not allowed in training")
    elif not isinstance(injection, Mapping):
        _fail(f"{p} is synthetic but has no fault_injection record")
    if ids is not None and (event.get("episode_id"), event.get("episode_index"), event.get("episode_seed")) != ids:
        _fail(f"{p} belongs to a different episode (stale or cross-episode event)")
    decision = _int(event.get("decision_index"), f"{p}.decision_index", minimum=1)
    if expected_decision is not None and decision != expected_decision:
        _fail(f"{p}.decision_index {decision} != expected {expected_decision}")
    _int(event.get("fallback_counter"), f"{p}.fallback_counter", minimum=0)

    before = _score_pair(event.get("score_before"), field=f"{p}.score_before")
    after = _score_pair(event.get("score_after"), field=f"{p}.score_after")
    delta = _score_pair(event.get("score_delta"), field=f"{p}.score_delta")
    if before != _ZERO_SCORE:
        _fail(f"{p}.score_before must be 0-0")
    if {s: after[s] - before[s] for s in ("A", "B")} != delta:
        _fail(f"{p}.score_delta disagrees with score_before/score_after")
    done = _bool(event.get("native_done"), f"{p}.native_done")
    winner = event.get("rally_winner_from_score_delta")
    reward = _finite_real(
        event.get("b2_rally_reward_from_score_delta"), field=f"{p}.b2_rally_reward_from_score_delta"
    )
    if done:
        if delta not in ({"A": 1, "B": 0}, {"A": 0, "B": 1}):
            _fail(f"{p} is terminal but score_delta is not exactly one point")
        expected_winner = "A" if delta["A"] == 1 else "B"
        if winner != expected_winner:
            _fail(f"{p}.rally_winner_from_score_delta disagrees with score_delta")
        if reward != shared_terminal_reward(terminated=True, rally_winner_side=expected_winner):
            _fail(f"{p} audit reward disagrees with its winner")
    else:
        if delta != _ZERO_SCORE or winner is not None or reward != 0.0:
            _fail(f"{p} is non-terminal but carries a point, winner or reward")
    return event


def _validate_events(
    events: Any, ids: tuple[str, int, int] | None, config: RewardConfig,
    expected_decision: int | None, *, require_envelope: bool = True,
) -> list[Mapping[str, Any]]:
    if not isinstance(events, list):
        _fail("fallback_events must be a list")
    return [
        _check_event(i, e, ids, config, expected_decision, require_envelope=require_envelope)
        for i, e in enumerate(events)
    ]


def _check_counters(events: list[Mapping[str, Any]], last: int | None) -> int | None:
    for event in events:
        counter = int(event["fallback_counter"])
        if last is not None and counter <= last:
            _fail("fallback_counter must be strictly increasing within an episode")
        last = counter
    return last


def _event_is_done(event: Mapping[str, Any]) -> bool:
    if "native_done" in event:
        return bool(event["native_done"])
    return float(event.get("b2_rally_reward_from_score_delta", 0.0) or 0.0) != 0.0


def _link_events(
    events: list[Mapping[str, Any]], *, internal: int, policy_transitions: int,
    terminated: bool, winner: str | None, score_after: Mapping[str, int] | None,
) -> int:
    """Reconcile events with the transition count; return unplayable-transition count."""
    unplayable = internal - policy_transitions - len(events)
    if unplayable < 0:
        _fail("internal_native_transitions is smaller than the logged transitions")
    done = [i for i, e in enumerate(events) if _event_is_done(e)]
    if len(done) > 1:
        _fail("More than one fallback event claims to end the rally")
    if done:
        if done[0] != len(events) - 1:
            _fail("A fallback event ended the rally but is not the final event")
        if not terminated:
            _fail("A fallback event ended the rally but the worker reports it non-terminal")
        last = events[-1]
        if "rally_winner_from_score_delta" in last and "score_after" in last:
            if last["rally_winner_from_score_delta"] != winner or dict(last["score_after"]) != dict(score_after or {}):
                _fail("Fallback event terminal outcome disagrees with the worker terminal result")
        elif "b2_rally_reward_from_score_delta" in last and winner is not None:
            expected_reward = shared_terminal_reward(terminated=True, rally_winner_side=winner)
            if float(last["b2_rally_reward_from_score_delta"]) != expected_reward:
                _fail("Fallback event terminal outcome disagrees with the worker terminal result")
    elif terminated and events and unplayable == 0:
        if any("native_done" in e for e in events):
            _fail("Rally terminated after fallback events, but no fallback event ended it")
    return unplayable


# --------------------------------------------------------------------------
# Core payload validators
# --------------------------------------------------------------------------
def _validate_step_core(
    payload: Any, config: RewardConfig, *, expected_ids: tuple[str, int, int] | None = None,
    decisions: int | None = None, action_id: int | None = None,
    expected_event_decision: int | None = None, require_envelope: bool = True,
) -> tuple[RewardVerdict, tuple[bool, ...] | None, list[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        _fail("Worker step payload must be a mapping")
    terminated = _bool(payload.get("terminated"), "terminated")
    truncated = _bool(payload.get("truncated"), "truncated")
    if terminated and truncated:
        _fail("Worker cannot terminate and truncate the same rally")
    reported = _finite_real(payload.get("reward"), field="worker.reward")

    has_ids = require_envelope or any(k in payload for k in ("episode_id", "episode_index", "episode_seed"))
    ids = _episode_ids(payload) if has_ids else None
    if expected_ids is not None and ids != expected_ids:
        _fail("Step payload belongs to a different episode than the active one")

    if require_envelope or "policy_action_id" in payload or "normal_action" in payload:
        policy_action = _int(payload.get("policy_action_id"), "policy_action_id", minimum=0, maximum=N_ACTIONS - 1)
        if action_id is not None and policy_action != action_id:
            _fail("Worker echoed a different action than the one sent")
        _validate_normal_action(payload.get("normal_action"), policy_action)

    events = _validate_events(
        payload.get("fallback_events"), ids, config, expected_event_decision,
        require_envelope=require_envelope,
    )

    if require_envelope or "internal_native_transitions" in payload:
        internal = _int(payload.get("internal_native_transitions"), "internal_native_transitions", minimum=1)
        _check_native_trace(payload, internal)
    else:
        _check_native_trace(payload, None)
        internal = max(1 + len(events), len(payload.get("native_reward_trace", [])))

    score = _score_pair(payload.get("score"), field="worker.score")
    if require_envelope or "observation" in payload:
        _check_observation_score(payload.get("observation"), score)

    mask: tuple[bool, ...] | None = None
    if require_envelope or "ready_for_policy" in payload or "action_mask" in payload:
        ready = _bool(payload.get("ready_for_policy"), "ready_for_policy")
        if ready == (terminated or truncated):
            _fail("ready_for_policy must be exactly the negation of terminated/truncated")
        mask = _validate_mask(payload, ready=ready)

    winner: str | None = None
    score_after: Mapping[str, int] | None = None
    if truncated:
        if payload.get("terminal") is not None:
            _fail("A truncated worker step must not include terminal metadata")
        if score != _ZERO_SCORE or reported != 0.0:
            _fail("A truncated rally must have score 0-0 and worker reward 0.0")
        if config.truncation_policy is TruncationPolicy.REJECT:
            raise TruncationRejectedError(
                "B2 v1 reward contract rejects time-limit truncation (truncation policy is REJECT)"
            )
        reward = 0.0 if config.truncation_policy is TruncationPolicy.ZERO else -1.0
        source = "truncation_policy"
    elif terminated:
        winner, score_after, reward = _validate_terminal(
            payload.get("terminal"), config, require_envelope=require_envelope
        )
        if score != score_after:
            _fail("Top-level worker score disagrees with terminal.score_after")
        if reported != reward:
            _fail("Worker reward disagrees with the independently verified terminal winner")
        source = "terminal_winner"
    else:
        if payload.get("terminal") is not None:
            _fail("A non-terminal worker step must not include terminal winner or metadata")
        if reported != 0.0:
            _fail(f"A non-terminal B2 transition must have reward 0.0, got {reported}")
        if score != _ZERO_SCORE:
            _fail("Non-terminal rally score changed before a verified point was resolved")
        reward, source = 0.0, "none"

    unplayable = _link_events(
        events, internal=internal, policy_transitions=1,
        terminated=terminated, winner=winner, score_after=score_after,
    )
    if decisions is not None:
        if decisions > config.max_decisions:
            _fail("Decision count exceeds max_decisions")
        if truncated and decisions != config.max_decisions:
            _fail("Truncation is only legal at exactly max_decisions decisions")
        if not terminated and not truncated and decisions >= config.max_decisions:
            _fail("Worker failed to truncate at max_decisions")
    verdict = RewardVerdict(
        reward=float(reward), terminated=terminated, truncated=truncated,
        winner_side=winner, reward_source=source,
        terminal_via_internal_transition=bool(terminated and internal > 1),
        fallback_event_count=len(events), unplayable_transition_count=unplayable,
        internal_native_transitions=internal,
    )
    return verdict, mask, events


def _validate_reset_core(
    payload: Any, config: RewardConfig
) -> tuple[ResetVerdict, tuple[str, int, int], tuple[bool, ...], list[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        _fail("Worker reset payload must be a mapping")
    if "reward" in payload:
        _fail("A reset payload must not carry a reward")
    status = payload.get("status")
    ready = _bool(payload.get("ready_for_policy"), "ready_for_policy")
    terminated = _bool(payload.get("terminated"), "terminated")
    if _bool(payload.get("truncated"), "truncated"):
        _fail("A reset payload cannot be truncated")
    no_policy = _bool(payload.get("no_policy_rally"), "no_policy_rally")
    if ready == terminated:
        _fail("Reset must be exactly one of policy-ready or terminal")
    ids = _episode_ids(payload)
    internal = _int(payload.get("internal_native_transitions"), "internal_native_transitions", minimum=0)
    _check_native_trace(payload, internal)
    score = _score_pair(payload.get("score"), field="worker.score")
    _check_observation_score(payload.get("observation"), score)
    events = _validate_events(payload.get("fallback_events"), ids, config, 1, require_envelope=True)
    mask = _validate_mask(payload, ready=ready)
    winner: str | None = None
    score_after: Mapping[str, int] | None = None
    if terminated:
        if status != "terminal_without_policy" or not no_policy:
            _fail("Terminal reset must have status terminal_without_policy and no_policy_rally=True")
        winner, score_after, _ = _validate_terminal(payload.get("terminal"), config, require_envelope=True)
        if score != score_after:
            _fail("Top-level reset score disagrees with terminal.score_after")
    else:
        if status != "ready_for_policy" or no_policy:
            _fail("Ready reset must have status ready_for_policy and no_policy_rally=False")
        if payload.get("terminal") is not None:
            _fail("A policy-ready reset must not include terminal metadata")
        if score != _ZERO_SCORE:
            _fail("A policy-ready reset must have score 0-0")
    unplayable = _link_events(
        events, internal=internal, policy_transitions=0,
        terminated=terminated, winner=winner, score_after=score_after,
    )
    verdict = ResetVerdict(
        ready_for_policy=ready, no_policy_rally=no_policy, winner_side=winner,
        fallback_event_count=len(events), unplayable_transition_count=unplayable,
        internal_native_transitions=internal,
    )
    return verdict, ids, mask, events


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def validate_worker_step_reward(payload: Mapping[str, Any], config: RewardConfig | None = None) -> float:
    """Stateless validation of one step payload; returns the B2 reward."""
    verdict, _, _ = _validate_step_core(
        payload, config or RewardConfig(), require_envelope=False
    )
    return verdict.reward


def validate_worker_reset_payload(payload: Mapping[str, Any], config: RewardConfig | None = None) -> ResetVerdict:
    return _validate_reset_core(payload, config or RewardConfig())[0]


class EpisodeRewardLedger:
    """Per-episode stateful guard. One instance per environment instance."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._clear()

    def _clear(self) -> None:
        self._active = False
        self._finished = False
        self._poisoned = False
        self._ids: tuple[str, int, int] | None = None
        self._decisions = 0
        self._return = 0.0
        self._nonzero_rewards = 0
        self._mask: tuple[bool, ...] | None = None
        self._last_counter: int | None = None

    @property
    def decisions(self) -> int:
        return self._decisions

    @property
    def episode_return(self) -> float:
        return self._return

    @property
    def finished(self) -> bool:
        return self._finished

    def begin_episode(self, reset_payload: Mapping[str, Any]) -> ResetVerdict:
        self._clear()
        try:
            verdict, ids, mask, events = _validate_reset_core(reset_payload, self.config)
            self._last_counter = _check_counters(events, None)
        except RewardIntegrityError:
            self._poisoned = True
            raise
        self._ids = ids
        self._active = True
        if verdict.ready_for_policy:
            self._mask = mask
        else:
            self._finished = True
            self._mask = None
        return verdict

    def record_step(self, action_id: int, payload: Mapping[str, Any]) -> RewardVerdict:
        if self._poisoned:
            _fail("Ledger is poisoned by an earlier integrity failure; reset required")
        if not self._active:
            _fail("record_step called before begin_episode")
        if self._finished or self._mask is None:
            _fail("record_step called after the episode ended (reward after termination)")
        try:
            action = _int(action_id, "action_id", minimum=0, maximum=N_ACTIONS - 1)
            if not self._mask[action]:
                _fail(f"Action {action} was masked in the state it was taken from")
            decisions = self._decisions + 1
            verdict, mask, events = _validate_step_core(
                payload, self.config, expected_ids=self._ids, decisions=decisions,
                action_id=action, expected_event_decision=decisions + 1,
                require_envelope=True,
            )
            last_counter = _check_counters(events, self._last_counter)
            new_return = self._return + verdict.reward
            nonzero = self._nonzero_rewards + (1 if verdict.reward != 0.0 else 0)
            if nonzero > 1 or abs(new_return) > 1.0:
                _fail("Episode return would leave {-1, 0, +1}: duplicate reward")
        except RewardIntegrityError:
            self._poisoned = True
            raise
        self._decisions = decisions
        self._return = new_return
        self._nonzero_rewards = nonzero
        self._last_counter = last_counter
        if verdict.terminated or verdict.truncated:
            self._finished = True
            self._mask = None
        else:
            self._mask = mask
        return verdict