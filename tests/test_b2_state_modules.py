import pytest
import json
from pathlib import Path
from badminton_rl.env.event_schema import B2EventSchema, EventSchemaError
from badminton_rl.env.history_buffer import EpisodeHistoryBuffer, EventSequenceError



def test_schema_golden_roundtrip(golden_payloads):
    events_delta = golden_payloads["reset_payload"].get("rally_events_delta", [])
    assert len(events_delta) > 0
    
    for raw_event in events_delta:
        event = B2EventSchema.from_dict(raw_event)
        assert event.schema_version == "b2_event_v1"
        assert event.event_index >= 0

def test_schema_mutation_rejection(golden_payloads):
    valid_event = golden_payloads["reset_payload"]["rally_events_delta"][0].copy()
    
    # Mutation 1: Unknown key injection
    mutated_keys = valid_event.copy()
    mutated_keys["rogue_debug_flag"] = True
    with pytest.raises(EventSchemaError, match="unknown fields"):
        B2EventSchema.from_dict(mutated_keys)
        
    # Mutation 2: Coordinate type corruption
    mutated_coord = valid_event.copy()
    mutated_coord["contact_xy"] = ["not", "a_float"]
    with pytest.raises(EventSchemaError):
        B2EventSchema.from_dict(mutated_coord)
        
    # Mutation 3: Validity flag mismatch
    mutated_flag = valid_event.copy()
    mutated_flag["field_validity"]["contact"] = not mutated_flag["field_validity"]["contact"]
    with pytest.raises(EventSchemaError, match="nullability mismatches field_validity"):
        B2EventSchema.from_dict(mutated_flag)

def test_history_buffer_ordering_and_padding(golden_payloads):
    buffer = EpisodeHistoryBuffer(capacity=8)
    events_delta = golden_payloads["reset_payload"]["rally_events_delta"]
    
    buffer.append_deltas(events_delta, current_decision_index=0)
    recent = buffer.get_k_recent_events()
    
    assert len(recent) == 8
    
    valid_events_count = len([e for e in recent if e is not None])
    assert valid_events_count == len(events_delta)
    
    if valid_events_count < 8:
        assert recent[0] is None
        assert recent[-1] is not None

def test_history_buffer_rejects_time_travel():
    buffer = EpisodeHistoryBuffer(capacity=8)
    
    event_1 = {"schema_version": "b2_event_v1", "event_index": 0, "decision_index": 0, "within_decision_index": 0, "field_validity": {}}
    event_2 = {"schema_version": "b2_event_v1", "event_index": 1, "decision_index": 1, "within_decision_index": 0, "field_validity": {}}
    
    buffer.append_deltas([event_1], current_decision_index=0)
    
    # Attempting to append an event with a lower index than expected
    with pytest.raises(EventSequenceError, match="Out of order"):
        buffer.append_deltas([event_1], current_decision_index=1)