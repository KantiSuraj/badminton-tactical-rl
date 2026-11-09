"""Strict B2 v1 rally reward contract and worker-payload validation.

The policy objective is the verified rally outcome only.  Native CoachAI+
scalar rewards and per-event fallback audit fields are deliberately excluded
from the returned RL reward.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any


class RewardIntegrityError(RuntimeError):
    """Raised when a worker result violates the B2 v1 reward contract."""


def shared_terminal_reward(
    *,
    terminated: bool,
    rally_winner_side: str | None,
    agent_side: str = "A",
) -> float:
    """Return the sparse, zero-sum rally reward for the controlled side.

    Non-terminal transitions must not carry winner metadata.  Terminal
    transitions must carry an unambiguous winner.  No shaping is performed.
    """
    if type(terminated) is not bool:
        raise RewardIntegrityError("terminated must be a Boolean")
    if not isinstance(agent_side, str) or agent_side not in ("A", "B"):
        raise RewardIntegrityError(f"Unsupported agent_side: {agent_side!r}")

    if not terminated:
        if rally_winner_side is not None:
            raise RewardIntegrityError(
                "A non-terminal transition must not declare a rally winner"
            )
        return 0.0

    if not isinstance(rally_winner_side, str) or rally_winner_side not in ("A", "B"):
        raise RewardIntegrityError(
            "A terminal rally requires a verified winner side ('A' or 'B')"
        )
    return 1.0 if rally_winner_side == agent_side else -1.0


def _finite_real(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RewardIntegrityError(f"{field} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise RewardIntegrityError(f"{field} must be finite")
    return result


def _score_pair(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"A", "B"}:
        raise RewardIntegrityError(f"{field} must be a mapping with exactly A and B")
    result: dict[str, int] = {}
    for side in ("A", "B"):
        score = value[side]
        if isinstance(score, bool) or not isinstance(score, numbers.Integral):
            raise RewardIntegrityError(f"{field}[{side!r}] must be an integer")
        if int(score) < 0:
            raise RewardIntegrityError(f"{field}[{side!r}] cannot be negative")
        result[side] = int(score)
    return result


def _validate_fallback_audit_events(
    events: Any,
    *,
    terminal_reward: float,
    terminated: bool,
) -> None:
    if not isinstance(events, list):
        raise RewardIntegrityError("fallback_events must be a list")

    nonzero_event_rewards: list[float] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise RewardIntegrityError(f"fallback_events[{index}] must be an object")

        # A forced fallback is an environment transition, never a sampled
        # policy action.  Check the fields when present, and require the core
        # separation fields for production fallback records.
        if event.get("event_type") == "b2_empty_mask_fallback":
            if event.get("policy_action_id", "__missing__") is not None:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] must have policy_action_id=null"
                )
            if event.get("policy_selected") is not False:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] must have policy_selected=false"
                )
            if event.get("policy_invoked") is not False:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] must have policy_invoked=false"
                )
            if event.get("policy_action_log_probability", "__missing__") is not None:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] must not carry a policy log probability"
                )
            if event.get("all_false_mask_passed_to_policy") is not False:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] must confirm no all-false mask reached the policy"
                )

        # Some worker events repeat a local score-delta reward for audit.  It
        # is checked for consistency but is NEVER added to the Gym reward.
        if "b2_rally_reward_from_score_delta" in event:
            value = _finite_real(
                event["b2_rally_reward_from_score_delta"],
                field=f"fallback_events[{index}].b2_rally_reward_from_score_delta",
            )
            if value not in {-1.0, 0.0, 1.0}:
                raise RewardIntegrityError(
                    f"fallback_events[{index}] contains an invalid B2 audit reward {value}"
                )
            if value != 0.0:
                nonzero_event_rewards.append(value)

    if len(nonzero_event_rewards) > 1:
        raise RewardIntegrityError(
            "More than one fallback event claims a terminal B2 rally reward"
        )
    if nonzero_event_rewards:
        if not terminated or nonzero_event_rewards[0] != terminal_reward:
            raise RewardIntegrityError(
                "Fallback event terminal reward disagrees with the worker terminal result"
            )


def validate_worker_step_reward(payload: Mapping[str, Any]) -> float:
    """Validate a production worker ``step`` payload and return its B2 reward.

    B2 v1 has no ordinary time-limit truncation.  A non-terminal result returns
    exactly zero; a terminal result returns exactly one verified rally reward.
    Native rewards and fallback-event audit rewards are never summed here.
    """
    if not isinstance(payload, Mapping):
        raise RewardIntegrityError("Worker step payload must be a mapping")

    terminated = payload.get("terminated")
    truncated = payload.get("truncated")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise RewardIntegrityError("Worker terminated/truncated fields must be Booleans")
    if terminated and truncated:
        raise RewardIntegrityError("Worker cannot terminate and truncate the same rally")
    if truncated:
        raise RewardIntegrityError(
            "B2 v1 rejects time-limit truncation; decision watchdogs must fail closed"
        )

    reported_reward = _finite_real(payload.get("reward"), field="worker.reward")
    terminal = payload.get("terminal")

    if not terminated:
        if terminal is not None:
            raise RewardIntegrityError(
                "A non-terminal worker step must not include terminal winner metadata"
            )
        if reported_reward != 0.0:
            raise RewardIntegrityError(
                f"A non-terminal B2 transition must have reward 0.0, got {reported_reward}"
            )
        top_level_score = payload.get("score")
        if top_level_score is not None and _score_pair(
            top_level_score, field="worker.score"
        ) != {"A": 0, "B": 0}:
            raise RewardIntegrityError(
                "Non-terminal rally score changed before a verified point was resolved"
            )
        terminal_reward = 0.0
    else:
        if not isinstance(terminal, Mapping):
            raise RewardIntegrityError("A terminal worker step requires terminal metadata")
        winner = terminal.get("winner_side")
        terminal_reward = shared_terminal_reward(
            terminated=True,
            rally_winner_side=winner,
            agent_side="A",
        )

        expected_delta = {"A": 1, "B": 0} if winner == "A" else {"A": 0, "B": 1}
        score_before = _score_pair(terminal.get("score_before"), field="terminal.score_before")
        score_after = _score_pair(terminal.get("score_after"), field="terminal.score_after")
        if score_before != {"A": 0, "B": 0}:
            raise RewardIntegrityError(
                "B2 v1 rally score must start from the neutral 0-0 baseline"
            )
        actual_delta = {
            side: score_after[side] - score_before[side]
            for side in ("A", "B")
        }
        reported_delta = _score_pair(terminal.get("score_delta"), field="terminal.score_delta")
        if actual_delta != expected_delta or reported_delta != expected_delta:
            raise RewardIntegrityError(
                "Terminal score delta must award exactly one point to the declared winner; "
                f"winner={winner!r}, actual_delta={actual_delta}, reported_delta={reported_delta}"
            )

        terminal_reward_field = _finite_real(
            terminal.get("b2_rally_reward"), field="terminal.b2_rally_reward"
        )
        if terminal_reward_field != terminal_reward:
            raise RewardIntegrityError(
                "terminal.b2_rally_reward disagrees with the verified winner"
            )
        top_level_score = payload.get("score")
        if top_level_score is not None:
            if _score_pair(top_level_score, field="worker.score") != score_after:
                raise RewardIntegrityError(
                    "Top-level worker score disagrees with terminal.score_after"
                )

        if reported_reward != terminal_reward:
            raise RewardIntegrityError(
                "Worker reward disagrees with the independently verified terminal winner"
            )

    if "fallback_events" not in payload:
        raise RewardIntegrityError("Worker step payload is missing fallback_events audit data")
    _validate_fallback_audit_events(
        payload["fallback_events"],
        terminal_reward=terminal_reward,
        terminated=terminated,
    )

    # Native rewards are intentionally not read or combined with this value.
    # Keep the terminal-derived reward as the only policy objective.
    return float(terminal_reward)