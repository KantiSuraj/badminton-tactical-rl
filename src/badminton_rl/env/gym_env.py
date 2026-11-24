"""
Gymnasium compatible adapter for the Badminton Tactical RL environment.
Enforces strict MDP constraints, integrates action masking, and manages state.
"""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from badminton_rl.env.observations import get_observation_space, ObservationBuilder
from badminton_rl.env.reward import EpisodeRewardLedger, RewardConfig, RewardIntegrityError
from badminton_rl.env.history_buffer import EpisodeHistoryBuffer, EventSequenceError
from badminton_rl.adapters.worker_client import WorkerClient

class InvalidActionError(Exception):
    """Raised when the agent selects an action rejected by the valid-action mask."""
    pass

class CourtGymEnv(gym.Env):
    """
    Physics-grounded, state-rich RL environment for badminton tactical decision-making.
    
    Delegates physical transitions to the underlying worker while strictly 
    enforcing Gym contracts, reward logic, and legal action masks.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self, 
        worker_client: WorkerClient, 
        reward_config: RewardConfig, 
        history_capacity: int = 8
    ):
        super().__init__()
        self.worker = worker_client
        
        # Core sub-modules
        self.ledger = EpisodeRewardLedger(config=reward_config)
        self.history = EpisodeHistoryBuffer(capacity=history_capacity)
        
        # 1350 Macro-actions: 10 families * 15 landing cells * 9 recovery cells[cite: 13]
        self.action_space = spaces.Discrete(1350)
        self.observation_space = get_observation_space()
        
        # State tracking
        self._current_mask = np.zeros(1350, dtype=np.bool_)
        self._last_obs_payload: Dict[str, Any] = {}
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment and returns the initial observation.
        """
        # Gymnasium protocol: Seed the environment's PRNG
        super().reset(seed=seed)
        
        # Flush buffers to prevent cross-episode state leakage
        self.history.reset()
        
        # Delegate reset execution to the worker
        try:
            reset_payload = self.worker.reset(seed=seed, options=options)
        except Exception as e:
            raise RuntimeError(f"Worker client failed during reset: {e}") from e
            
        # Ledger strictly validates the reset payload (verifies 0-0 score, etc.)
        verdict = self.ledger.begin_episode(reset_payload)
        
        # Safely extract and apply the action mask
        self._update_mask(reset_payload.get("action_mask"))
            
        # Update sequence buffer with initial serve events
        if "rally_events_delta" in reset_payload:
            self.history.append_deltas(reset_payload["rally_events_delta"], current_decision_index=0)
            
        self._last_obs_payload = reset_payload.get("observation", {})
        
        # Build pure, side-effect-free Gym observation[cite: 12]
        obs = ObservationBuilder.build(self._last_obs_payload, self.history.get_k_recent_events())
        
        info = {
            "worker_status": reset_payload.get("status"),
            "fallback_events_count": verdict.fallback_event_count,
            "unplayable_transition_count": verdict.unplayable_transition_count,
            "ready_for_policy": verdict.ready_for_policy,
            "no_policy_rally": verdict.no_policy_rally,
        }
        
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes a single decision step (semi-Markov decision epoch).
        """
        # Gymnasium can sometimes pass 0-d numpy arrays. Ensure it's a native int.
        action_idx = int(action)
        
        # 1. Action Feasibility Contract: Fail explicitly on invalid shots[cite: 18]
        if not self._current_mask[action_idx]:
            raise InvalidActionError(
                f"Action {action_idx} was rejected by the valid-action mask. "
                "Ensure the RL algorithm uses `env.action_masks()` during sampling to prevent teleporting shots."
            )
            
        # 2. Execute transition
        try:
            step_payload = self.worker.step({"action_id": action_idx})
        except Exception as e:
            raise RuntimeError(f"Worker client crashed during step: {e}") from e
        
        # 3. Ledger evaluates terminal conditions and computes zero-sum reward
        verdict = self.ledger.record_step(action_idx, step_payload)
        
        # 4. Advance history with chronological delta validation
        if "rally_events_delta" in step_payload:
            self.history.append_deltas(step_payload["rally_events_delta"], self.ledger.decisions)
            
        # 5. Update local state caches
        self._update_mask(step_payload.get("action_mask"))
        self._last_obs_payload = step_payload.get("observation", {})
        
        # 6. Build Neural Observation
        obs = ObservationBuilder.build(self._last_obs_payload, self.history.get_k_recent_events())
        
        # 7. Package Info
        info = verdict.info()
        info["action_mask_sha256"] = step_payload.get("action_mask_sha256")
        info["elapsed_decisions"] = self.ledger.decisions
            
        return obs, verdict.reward, verdict.terminated, verdict.truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Required by MaskablePPO (sb3-contrib).
        Returns a boolean array where True indicates a physically and legally feasible action.
        """
        return self._current_mask
        
    def _update_mask(self, mask_list: Optional[list]) -> None:
        """Internal helper to safely update the mask, defaulting to zeros if terminal."""
        if mask_list is not None and len(mask_list) == 1350:
            self._current_mask = np.array(mask_list, dtype=np.bool_)
        else:
            self._current_mask = np.zeros(1350, dtype=np.bool_)

    def close(self):
        """Gracefully release the underlying worker resources."""
        if hasattr(self, 'worker') and hasattr(self.worker, 'close'):
            self.worker.close()