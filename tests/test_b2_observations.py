import numpy as np
from badminton_rl.env.observations import ObservationBuilder, get_observation_space

def test_observation_builder_shapes(golden_payloads):
    worker_obs = golden_payloads["reset_payload"]["observation"]
    
    # Mocking 8 empty events for structure
    mock_events = [{"player": "A", "type": 1.0, "player_hit_x": 0.0, "player_hit_y": 0.0, "landing_x": 0.0, "landing_y": 0.0}] * 8
    
    obs = ObservationBuilder.build(worker_obs, mock_events)
    space = get_observation_space()
    
    assert space.contains(obs), "Generated observation violates the Gym space boundaries or types."
    
    assert obs["players"].shape == (2, 4)
    assert obs["shuttle"].shape == (4,)
    assert obs["game_context"].shape == (5,)
    assert obs["rally_history"].shape == (5, 6)
    assert obs["history_mask"].shape == (5,)
    
    assert obs["players"].dtype == np.float32
    assert obs["history_mask"].dtype == np.bool_