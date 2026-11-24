"""Strict schema validation for B2 tactical events."""
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

class EventSchemaError(Exception):
    """Raised when an event payload violates the B2 schema contract."""
    pass

_VALID_ACTORS = {None, "A", "B"}
_VALID_CONTROLLERS = {None, "RALLYNET", "POLICY", "ENVIRONMENT", "UNKNOWN"}
_VALID_EVENT_TYPES = {None, "STROKE", "FALLBACK_STROKE", "UNPLAYABLE_TRANSITION", "UNKNOWN", "SERVE", "FAULT"}

@dataclass(frozen=True)
class B2EventSchema:
    """Represents a single tactical stroke or environment transition."""
    schema_version: str
    event_index: int
    decision_index: int
    within_decision_index: int
    actor: str | None
    controller: str | None
    event_type: str | None
    shot_family_id: int | None
    policy_action_family_id: int | None
    native_shot_id: int | None
    model_shot_id: int | None
    policy_action_id: int | None
    contact_xy: tuple[float, float] | None
    landing_xy: tuple[float, float] | None
    recovery_xy: tuple[float, float] | None
    coordinate_frame: str | None
    hitter_side: str | None
    field_validity: Mapping[str, bool]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "B2EventSchema":
        """Constructs and deeply validates an event from a raw worker dictionary."""
        if not isinstance(data, dict):
            raise EventSchemaError(f"Event payload must be a dictionary, got {type(data).__name__}.")

        # 1. Strict rejection of unknown fields
        allowed_keys = {f.name for f in fields(cls)}
        unknown_keys = set(data.keys()) - allowed_keys
        if unknown_keys:
            raise EventSchemaError(f"Event payload contains unknown fields: {sorted(unknown_keys)}")

        if data.get("schema_version") != "b2_event_v1":
            raise EventSchemaError(f"Unsupported schema version: {data.get('schema_version')}")

        # 2. Enumerations and allowed combinations
        actor = data.get("actor")
        controller = data.get("controller")
        event_type = data.get("event_type")

        if actor not in _VALID_ACTORS:
            raise EventSchemaError(f"Invalid actor: {actor}")
        if controller not in _VALID_CONTROLLERS:
            raise EventSchemaError(f"Invalid controller: {controller}")
        if event_type not in _VALID_EVENT_TYPES:
            raise EventSchemaError(f"Invalid event_type: {event_type}")

        # 3. Finiteness and exact type checking for coordinates
        def parse_coord(key: str) -> tuple[float, float] | None:
            coord = data.get(key)
            if coord is None:
                return None
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                raise EventSchemaError(f"{key} must be a 2-tuple")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coord):
                raise EventSchemaError(f"{key} contains non-finite or non-numeric coordinates")
            return (float(coord[0]), float(coord[1]))

        contact_xy = parse_coord("contact_xy")
        landing_xy = parse_coord("landing_xy")
        recovery_xy = parse_coord("recovery_xy")

        # 4. Nullability strictly agrees with the validity flags
        validity = data.get("field_validity", {})
        if not isinstance(validity, dict):
            raise EventSchemaError("field_validity must be a dictionary")
            
        if validity.get("contact", False) != (contact_xy is not None):
            raise EventSchemaError("contact_xy nullability mismatches field_validity flag")
        if validity.get("landing", False) != (landing_xy is not None):
            raise EventSchemaError("landing_xy nullability mismatches field_validity flag")
        if validity.get("recovery", False) != (recovery_xy is not None):
            raise EventSchemaError("recovery_xy nullability mismatches field_validity flag")

        # 5. Strict integer parsing for indices and IDs
        def parse_int(key: str, required: bool = False) -> int | None:
            val = data.get(key)
            if val is None:
                if required:
                    raise EventSchemaError(f"Missing required integer field: {key}")
                return None
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not float(val).is_integer():
                raise EventSchemaError(f"{key} must be an integer, got {type(val).__name__}")
            return int(val)

        return cls(
            schema_version=data["schema_version"],
            event_index=parse_int("event_index", required=True),
            decision_index=parse_int("decision_index", required=True),
            within_decision_index=parse_int("within_decision_index", required=True),
            actor=actor,
            controller=controller,
            event_type=event_type,
            shot_family_id=parse_int("shot_family_id"),
            policy_action_family_id=parse_int("policy_action_family_id"),
            native_shot_id=parse_int("native_shot_id"),
            model_shot_id=parse_int("model_shot_id"),
            policy_action_id=parse_int("policy_action_id"),
            contact_xy=contact_xy,
            landing_xy=landing_xy,
            recovery_xy=recovery_xy,
            coordinate_frame=data.get("coordinate_frame"),
            hitter_side=data.get("hitter_side"),
            field_validity=validity
        )