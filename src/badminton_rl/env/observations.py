import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List,Sequence
from badminton_rl.env.event_schema import B2EventSchema

# Constants matching the Methodology Report dimensions
F_PLAYER = 4       # [x, y, vx, vy]
F_SHUTTLE = 4      # [x, y, vx, vy]
F_GAME = 5         # [score_a, score_b, round, previous_shot_id, launch_flag]
F_EVENT = 6        # [hitter_side, shot_id, hit_x, hit_y, land_x, land_y]
K_HISTORY = 5      # Last K strokes to track

def get_observation_space() -> spaces.Dict:
    """
    Defines the structured observation space exposed to the policy.
    Aligns exactly with the dictionary specification in the methodology blueprint.
    """
    return spaces.Dict({
        "players": spaces.Box(low=-2000.0, high=2000.0, shape=(2, F_PLAYER), dtype=np.float32),
        "shuttle": spaces.Box(low=-2000.0, high=2000.0, shape=(F_SHUTTLE,), dtype=np.float32),
        "game_context": spaces.Box(low=-1.0, high=1000.0, shape=(F_GAME,), dtype=np.float32),
        "rally_history": spaces.Box(low=-2000.0, high=2000.0, shape=(K_HISTORY, F_EVENT), dtype=np.float32),
        "history_mask": spaces.Box(low=0, high=1, shape=(K_HISTORY,), dtype=np.bool_),
        # Placeholders for explicit physical constraints (Time-to-intercept, reach margin)
        "intercept": spaces.Box(low=-2000.0, high=2000.0, shape=(4,), dtype=np.float32), 
        "opponent": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
    })

class ObservationBuilder:
    """
    Deterministic, side-effect-free builders for the current state and history tensors.
    """

    @staticmethod
    def build(
        worker_observation: Dict[str, Any], 
        historical_events: List[Any]
    ) -> Dict[str, np.ndarray]:
        """
        Maps a B2 payload observation and history buffer into Gym-compatible tensors.
        Does not mutate the inputs.
        """
        
        # 1. Players Tensor (Shape: [2, 4])
        raw_players = worker_observation.get("players_xy", [[0.0, 0.0], [0.0, 0.0]])
        players = np.zeros((2, F_PLAYER), dtype=np.float32)
        players[0, :2] = raw_players[0]
        players[1, :2] = raw_players[1]
        
        # 2. Shuttle Tensor (Shape: [4,])
        raw_shuttle = worker_observation.get("shuttle_xy", [0.0, 0.0])
        shuttle = np.zeros((F_SHUTTLE,), dtype=np.float32)
        shuttle[:2] = raw_shuttle
        
        # 3. Game Context Tensor (Shape: [5,])
        score_ab = worker_observation.get("score_ab", [0, 0])
        game_context = np.array([
            score_ab[0],
            score_ab[1],
            worker_observation.get("round", 1),
            worker_observation.get("previous_native_shot_id", 0),
            1.0 if worker_observation.get("launch", False) else 0.0
        ], dtype=np.float32)
        
        # 4. History Tensor & Mask (Shape: [K_HISTORY, 6] and [K_HISTORY,])
        history_tensor = np.zeros((K_HISTORY, F_EVENT), dtype=np.float32)
        history_mask = np.zeros((K_HISTORY,), dtype=np.bool_)
        
        # historical_events is pre-padded with None: [None, None, Oldest, ..., Newest]
        for i, event in enumerate(reversed(historical_events[-K_HISTORY:])):
            if event is None:
                continue  # Safely skip the pre-padded empty slots
                
            history_mask[i] = True
            
            # Handle strict B2EventSchema objects (production)
            if hasattr(event, "schema_version"):
                side = event.hitter_side or event.actor
                hitter = 1.0 if side == "A" else 0.0
                
                # Safely fallback to native_shot_id if model_shot_id is None (Opponent B)
                shot_type = float(event.model_shot_id if event.model_shot_id is not None else (event.native_shot_id or 0.0))
                
                hx, hy = event.contact_xy if event.contact_xy is not None else (0.0, 0.0)
                lx, ly = event.landing_xy if event.landing_xy is not None else (0.0, 0.0)
                
            # Handle raw dictionaries (testing fixtures)
            else:
                side = event.get("hitter_side") or event.get("actor") or event.get("player")
                hitter = 1.0 if side == "A" else 0.0
                
                shot_type = float(event.get("model_shot_id") or event.get("native_shot_id") or event.get("type") or 0.0)
                
                contact = event.get("contact_xy")
                if contact is not None:
                    hx, hy = float(contact[0]), float(contact[1])
                else:
                    hx, hy = float(event.get("player_hit_x", 0.0)), float(event.get("player_hit_y", 0.0))
                    
                landing = event.get("landing_xy")
                if landing is not None:
                    lx, ly = float(landing[0]), float(landing[1])
                else:
                    lx, ly = float(event.get("landing_x", 0.0)), float(event.get("landing_y", 0.0))

            history_tensor[i] = np.array([hitter, shot_type, hx, hy, lx, ly], dtype=np.float32)

        # 5. Missing / Placeholder Data
        intercept = np.zeros((4,), dtype=np.float32)
        opponent = np.zeros((1,), dtype=np.float32)
        
        return {
            "players": players,
            "shuttle": shuttle,
            "game_context": game_context,
            "rally_history": history_tensor,
            "history_mask": history_mask,
            "intercept": intercept,
            "opponent": opponent
        }