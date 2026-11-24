import hashlib
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from badminton_rl.env.gym_env import CourtGymEnv, InvalidActionError
from badminton_rl.env.reward import N_ACTIONS, RewardConfig, decode_action


def _mask_and_digest(valid_indices: list[int]) -> tuple[list[bool], str]:
    mask = [False] * N_ACTIONS
    for idx in valid_indices:
        mask[idx] = True
    digest = hashlib.sha256(bytes(1 if b else 0 for b in mask)).hexdigest()
    return mask, digest


def _make_event(
    *,
    event_index: int,
    decision_index: int,
    actor: str,
    controller: str,
    family_id: int | None = None,
    policy_action_id: int | None = None,
) -> dict:
    return {
        "schema_version": "b2_event_v1",
        "event_index": event_index,
        "decision_index": decision_index,
        "within_decision_index": 0,
        "actor": actor,
        "controller": controller,
        "event_type": "STROKE",
        "shot_family_id": family_id if family_id is not None else 0,
        "policy_action_family_id": family_id,
        "native_shot_id": 1 if actor == "B" else 2,
        "model_shot_id": None if actor == "B" else 1,
        "policy_action_id": policy_action_id,
        "contact_xy": [0.0, 100.0] if actor == "B" else [0.0, -150.0],
        "landing_xy": [0.0, -150.0] if actor == "B" else [0.0, 150.0],
        "recovery_xy": [0.0, 150.0] if actor == "B" else [0.0, -150.0],
        "coordinate_frame": "A_side_local_CoachAI_native_units",
        "hitter_side": actor,
        "field_validity": {"contact": True, "landing": True, "recovery": True},
    }


class StatefulMockWorkerClient:
    """Contract-compliant stateful mock satisfying EpisodeRewardLedger and EpisodeHistoryBuffer."""

    def __init__(self, terminate_on_decision: int = 3):
        self.terminate_on_decision = terminate_on_decision
        self._episode_index = -1
        self._episode_id = ""
        self._episode_seed = 0
        self._decisions = 0
        self._event_index = 0

    def reset(self, seed=None, options=None):
        self._episode_index += 1
        self._episode_seed = int(seed) if seed is not None else 42 + self._episode_index
        self._episode_id = f"b2-episode-{self._episode_index:06d}"
        self._decisions = 0
        self._event_index = 1

        # Action 0 (family 0) and 135 (family 1) are both in SUPPORTED_FAMILY_IDS
        mask, digest = _mask_and_digest([0, 135])
        return {
            "status": "ready_for_policy",
            "episode_id": self._episode_id,
            "episode_index": self._episode_index,
            "episode_seed": self._episode_seed,
            "observation": {
                "schema_version": "b2_obs_v1",
                "coordinate_frame": "A_side_local_CoachAI_native_units",
                "coordinate_units": "CoachAI_native_units",
                "players_xy": [[0.0, -150.0], [0.0, 150.0]],
                "player_order": ["A", "B"],
                "shuttle_xy": [0.0, -150.0],
                "previous_native_shot_id": 1,
                "score_ab": [0, 0],
                "round": 1,
                "launch": False,
                "rally_history_length": 1,
            },
            "action_mask": mask,
            "action_mask_sha256": digest,
            "ready_for_policy": True,
            "terminated": False,
            "truncated": False,
            "internal_native_transitions": 0,
            "native_reward_trace": [],
            "fallback_events": [],
            "terminal": None,
            "no_policy_rally": False,
            "score": {"A": 0, "B": 0},
            "rally_events_delta": [
                _make_event(event_index=0, decision_index=0, actor="B", controller="RALLYNET")
            ],
        }

    def step(self, payload):
        action_id = int(payload["action_id"])
        family, landing, recovery = decode_action(action_id)
        self._decisions += 1
        terminated = self._decisions >= self.terminate_on_decision

        event = _make_event(
            event_index=self._event_index,
            decision_index=self._decisions,
            actor="A",
            controller="POLICY",
            family_id=family,
            policy_action_id=action_id,
        )
        self._event_index += 1

        if terminated:
            mask, digest = _mask_and_digest([])
            score = {"A": 1, "B": 0}
            reward = 1.0
            terminal = {
                "winner_side": "A",
                "winner_name": "A",
                "recorded_winner_name": "A",
                "score_before": {"A": 0, "B": 0},
                "score_after": {"A": 1, "B": 0},
                "score_delta": {"A": 1, "B": 0},
                "raw_rally_reward": 1.0,
                "terminated_after_fallback": False,
                "b2_rally_reward": 1.0,
            }
        else:
            mask, digest = _mask_and_digest([0, 135])
            score = {"A": 0, "B": 0}
            reward = 0.0
            terminal = None

        return {
            "episode_id": self._episode_id,
            "episode_index": self._episode_index,
            "episode_seed": self._episode_seed,
            "policy_action_id": action_id,
            "observation": {
                "schema_version": "b2_obs_v1",
                "coordinate_frame": "A_side_local_CoachAI_native_units",
                "coordinate_units": "CoachAI_native_units",
                "players_xy": [[0.0, -150.0], [0.0, 150.0]],
                "player_order": ["A", "B"],
                "shuttle_xy": [0.0, 150.0],
                "previous_native_shot_id": 2,
                "score_ab": [score["A"], score["B"]],
                "round": 1,
                "launch": False,
                "rally_history_length": self._event_index,
            },
            "action_mask": mask,
            "action_mask_sha256": digest,
            "ready_for_policy": not terminated,
            "terminated": terminated,
            "truncated": False,
            "reward": reward,
            "native_reward_trace": [0.0],
            "internal_native_transitions": 1,
            "fallback_events": [],
            "normal_action": {
                "policy_action_id": action_id,
                "canonical_action": [family, landing, recovery],
                "native_validator_corrected": False,
                "native_output_record_matches": True,
            },
            "terminal": terminal,
            "score": score,
            "rally_events_delta": [event],
        }

    def close(self):
        pass


@pytest.fixture
def env():
    client = StatefulMockWorkerClient(terminate_on_decision=3)
    config = RewardConfig(max_decisions=25, player_name="A", opponent_name="B")
    environment = CourtGymEnv(worker_client=client, reward_config=config, history_capacity=8)
    yield environment
    environment.close()


def test_gym_api_compliance(env):
    original_sample = env.action_space.sample
    env.action_space.sample = lambda *args, **kwargs: int(
        np.random.choice(np.flatnonzero(env.action_masks()))
    )
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.action_space.sample = original_sample


def test_action_mask_rejection(env):
    env.reset(seed=42)
    mask = env.action_masks()
    invalid_action = int(np.argmin(mask))
    assert not mask[invalid_action]
    with pytest.raises(InvalidActionError, match="rejected by the valid-action mask"):
        env.step(invalid_action)


def test_masked_rollout(env):
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    decisions = 0

    while not (terminated or truncated) and decisions < 10:
        mask = env.action_masks()
        valid_actions = np.flatnonzero(mask)
        assert len(valid_actions) > 0
        action = int(np.random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        decisions += 1

    assert terminated is True
    assert decisions == 3
    assert reward == 1.0