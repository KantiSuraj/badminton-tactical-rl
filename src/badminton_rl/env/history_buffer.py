"""Sequential event buffer providing a fixed-length window to the neural network."""
from typing import Any
from collections import deque
from .event_schema import B2EventSchema, EventSchemaError

class EventSequenceError(Exception):
    """Raised when events arrive out of order, or episode boundaries are violated."""
    pass

class EpisodeHistoryBuffer:
    """Manages the rolling K-event window for the policy observation."""
    
    def __init__(self, capacity: int = 8):
        if capacity <= 0:
            raise ValueError("Buffer capacity must be strictly positive.")
        self.capacity = capacity
        self.events: deque[B2EventSchema] = deque(maxlen=capacity)
        self.full_log: list[B2EventSchema] = []
        self._next_expected_index = 0
        self._decision_index = 0

    def reset(self) -> None:
        """Clears the buffer for a new episode."""
        self.events.clear()
        self.full_log.clear()
        self._next_expected_index = 0
        self._decision_index = 0

    def append_deltas(self, rally_events_delta: list[dict[str, Any]], current_decision_index: int) -> None:
        """Appends and validates a chronologically sorted list of event dictionaries."""
        if not isinstance(rally_events_delta, list):
            raise EventSequenceError(f"rally_events_delta must be a list, got {type(rally_events_delta).__name__}")
            
        if current_decision_index < self._decision_index:
            raise EventSequenceError(
                f"Decision index moved backward: {current_decision_index} < {self._decision_index}"
            )
        self._decision_index = current_decision_index

        for raw_event in rally_events_delta:
            try:
                event = B2EventSchema.from_dict(raw_event)
            except EventSchemaError as e:
                raise EventSequenceError(f"Failed to parse event delta: {e}") from e
            
            # Explicit duplicate/gap check via deterministic indices
            if event.event_index != self._next_expected_index:
                raise EventSequenceError(
                    f"Out of order or duplicate event! Expected index {self._next_expected_index}, "
                    f"got {event.event_index}"
                )
            
            # Ensure the event's internal decision clock does not exceed the environment's clock
            if event.decision_index > current_decision_index:
                raise EventSequenceError(
                    f"Event decision index ({event.decision_index}) exceeds current environment "
                    f"decision index ({current_decision_index})"
                )
                
            self.events.append(event)
            self.full_log.append(event)
            self._next_expected_index += 1

    def get_k_recent_events(self) -> list[B2EventSchema | None]:
        """
        Returns exactly `capacity` events.
        If fewer events exist, the list is pre-padded at the BEGINNING with None.
        """
        result: list[B2EventSchema | None] = list(self.events)
        padding_needed = self.capacity - len(result)
        if padding_needed > 0:
            result = [None] * padding_needed + result
        return result