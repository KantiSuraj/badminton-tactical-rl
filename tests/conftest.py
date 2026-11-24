import pytest
import hashlib
import numpy as np

@pytest.fixture
def golden_payloads():
    """Mocked golden payload to satisfy strict ledger and observation tests."""
    mask = [True] * 1350
    mask_hash = hashlib.sha256(np.array(mask, dtype=np.uint8).tobytes()).hexdigest()
    
    return {
        "reset_payload": {
            "status": "ready_for_policy",
            "observation": {
                "players_xy": [[0.0, 0.0], [0.0, 0.0]],
                "shuttle_xy": [0.0, 0.0],
                "score_ab": [0, 0],
                "round": 1,
                "previous_native_shot_id": 0,
                "launch": False
            },
            "score": {"A": 0, "B": 0},
            "ready_for_policy": True,
            "terminated": False,
            "truncated": False,
            "internal_native_transitions": 0,
            "fallback_events": [],
            "action_mask": mask,
            "action_mask_sha256": mask_hash,
            # Add this field to satisfy the ledger validation
            "no_policy_rally": False,
            "rally_events_delta": [
                {
                    "schema_version": "b2_event_v1",
                    "event_index": 0,
                    "decision_index": 0,
                    "within_decision_index": 0,
                    "actor": "B",
                    "controller": "RALLYNET",
                    "event_type": "STROKE",
                    "shot_family_id": 1,
                    "policy_action_family_id": None,
                    "native_shot_id": 1,
                    "model_shot_id": 1,
                    "policy_action_id": None,
                    "contact_xy": [0.0, 0.0],
                    "landing_xy": [0.0, 0.0],
                    "recovery_xy": [0.0, 0.0],
                    "coordinate_frame": "A_side_local_CoachAI_native_units",
                    "hitter_side": "B",
                    "field_validity": {"contact": True, "landing": True, "recovery": True}
                }
            ]
        }
    }