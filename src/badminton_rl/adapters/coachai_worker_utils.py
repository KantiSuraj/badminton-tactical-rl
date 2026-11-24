"""Pure helpers, constants and action mappings for the B2 CoachAI+ worker.

This module must stay free of NumPy / PyTorch / pandas / CoachAI+ imports and
must not import `coachai_worker_core` or `coachai_worker_capture`.  The parent
process can therefore import it without loading the heavyweight native stack.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any

EXPECTED_CONSTRAINT_SHA256 = "779a9756690a1c1b4efcda57cf0b0ab0089a17bf716a9554f76b1e40957ee3c3"
EXPECTED_FILE_SHA256 = {
    "input_data/All_dataset.csv": "3116fd86391e2cec477b513f9e00bc75efc596bf3862779927eee7c923403ba6",
    "input_data/target_players_ids.pkl": "b6c2e4ea5b58923d20c1a7b51a0ee6f8d83a2958d622183995c271222f51e755",
    "BadmintonEnv/Agent/hyperparameters.pkl": "7e5ab372205f5b37db450e519040bdb358136e4c2b66461a1374b50c43588c7d",
    "BadmintonEnv/Agent/RallyNet_weight.trc": "bed1f112d542903a5c6db898f56a0e6baf3d240abe2879041cc092805e985c7d",
    "BadmintonEnv/Agent/INV_weight.pth": "ae9d0e1837c3515c3507172b060e82808e83840938313523509438d56eff3c60",
    "BadmintonEnv/Agent/BC4Serve_weight.pth": "8036be9ccf012174a51ded00689d494b530f01249ea28ce4e7dceb93653be344",
}
EXPECTED_COACHAI_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"
ACTION_MAP_VERSION = "coachai_v1"
OBSERVATION_SCHEMA_VERSION = "b2_native_state_v1"
N_FAMILIES, N_LANDING, N_RECOVERY = 10, 15, 9
N_ACTIONS = N_FAMILIES * N_LANDING * N_RECOVERY
HALF_WIDTH_M, HALF_LENGTH_M = 2.59, 6.70
NATIVE_HALF_WIDTH, NATIVE_HALF_LENGTH = 127.5, 330.0
EXPECTED_POLICY_LANDING_ZONES = [24, 23, 21, 20, 19, 17, 16, 15, 13, 8, 7, 5, 4, 3, 1]
EXPECTED_COMMON_FALLBACK_ZONES = [9, 10, 11, 16]
NATIVE_FALLBACK_IDS = tuple(range(2, 10))

# Canonical family ID -> model output ID -> CoachAI native category ID.
# Unsupported family IDs remain in the fixed 10-family action tensor but have
# no true mask entries.  The family-to-native semantics were approved for B2 v1.
FAMILY_MAP: dict[int, dict[str, Any]] = {
    0: {"family": "CLEAR", "model_id": 3, "native_id": 2, "native_name": "Clear"},
    1: {"family": "DROP", "model_id": 5, "native_id": 9, "native_name": "Drop"},
    2: {"family": "SMASH", "model_id": 4, "native_id": 4, "native_name": "Smash"},
    4: {"family": "DRIVE", "model_id": 7, "native_id": 6, "native_name": "Drive"},
    5: {"family": "PUSH", "model_id": 9, "native_id": 3, "native_name": "Push Shot"},
    6: {"family": "LIFT", "model_id": 6, "native_id": 8, "native_name": "Lob"},
    7: {"family": "NET_SHOT", "model_id": 8, "native_id": 7, "native_name": "Net Shot"},
}
UNSUPPORTED_FAMILIES = {3: "HALF_SMASH", 8: "BLOCK", 9: "CUT"}
MODEL_TO_NATIVE = {
    1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8,
    7: 6, 8: 7, 9: 3, 10: 5, 11: 11,
}
NATIVE_TO_MODEL = {native: model for model, native in MODEL_TO_NATIVE.items()}
REASON_NATIVE_EMPTY = "NATIVE_CONSTRAINT_EMPTY"
REASON_TAXONOMY_GAP = "B2_TAXONOMY_GAP"
REASON_GEOMETRY_EMPTY = "NO_JOINT_ACTION_AFTER_GEOMETRY"


def _describe_history_structure(value, depth=0):
    if depth >= 5:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            str(k): _describe_history_structure(v, depth + 1)
            for k, v in value.items()
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "length": len(value),
            "elements": [
                _describe_history_structure(item, depth + 1)
                for item in value
            ],
        }
    if isinstance(value, list):
        non_none = next((x for x in value if x is not None), None)
        return {
            "type": "list",
            "length": len(value),
            "first_element": (
                _describe_history_structure(value[0], depth + 1)
                if value else None
            ),
            "first_non_none_element": (
                _describe_history_structure(non_none, depth + 1)
                if non_none is not None else None
            ),
        }
    return {"type": type(value).__name__, "repr": repr(value)}


class WorkerFault(RuntimeError):
    """An operation failure with a stable protocol code and fatality marker."""

    def __init__(self, code: str, message: str, *, fatal: bool = False, details: Any = None):
        super().__init__(message)
        self.code = code
        self.fatal = fatal
        self.details = details


class UnexpectedNativeAgentCall(RuntimeError):
    """CoachAI unexpectedly attempted to generate the externally controlled A action."""


def encode_action(family_id: int, landing_id: int, recovery_id: int) -> int:
    for name, value, upper in (
        ("family_id", family_id, N_FAMILIES),
        ("landing_id", landing_id, N_LANDING),
        ("recovery_id", recovery_id, N_RECOVERY),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= int(value) < upper:
            raise ValueError(f"{name} must be an integer in [0, {upper}), got {value!r}")
    family_id, landing_id, recovery_id = int(family_id), int(landing_id), int(recovery_id)
    return (family_id * N_LANDING + landing_id) * N_RECOVERY + recovery_id


def decode_action(action_id: int) -> tuple[int, int, int]:
    if isinstance(action_id, bool) or not isinstance(action_id, numbers.Integral) or not 0 <= int(action_id) < N_ACTIONS:
        raise ValueError(f"action_id must be an integer in [0, {N_ACTIONS}), got {action_id!r}")
    action_id = int(action_id)
    family_id, remainder = divmod(action_id, N_LANDING * N_RECOVERY)
    landing_id, recovery_id = divmod(remainder, N_RECOVERY)
    return family_id, landing_id, recovery_id


def derive_fallback_seed(
    worker_seed: int,
    episode_index: int,
    decision_index: int,
    fallback_counter: int,
    *,
    episode_seed: int | None = None,
) -> int:
    """Stable event RNG seed; independent of Python's randomized hash()."""
    base = (
        f"b2-fallback-v1|{int(worker_seed)}|{int(episode_index)}|"
        f"{int(decision_index)}|{int(fallback_counter)}"
    )
    if episode_seed is not None:
        base = f"{base}|{int(episode_seed)}"
    material = base.encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8, person=b"B2-fallback-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def derive_reset_retry_seed(base_seed: int, attempt: int) -> int:
    """Deterministic seed derivation when resampling a no-policy opening rally."""
    if int(attempt) == 0:
        return int(base_seed)
    material = f"b2-reset-retry-v1|{int(base_seed)}|{int(attempt)}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8, person=b"B2-reset-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def derive_action_jitter_seed(
    worker_seed: int,
    episode_seed: int,
    episode_index: int,
    decision_index: int,
    action_id: int,
) -> int:
    """Deterministic seed for within-cell target coordinate jitter."""
    material = (
        f"b2-jitter-v1|{int(worker_seed)}|{int(episode_seed)}|"
        f"{int(episode_index)}|{int(decision_index)}|{int(action_id)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8, person=b"B2-jitter-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def classify_empty_mask_reason(
    *, contact_reachable: bool, contact_zone_valid: bool,
    native_allowed_ids: list[int], mapped_native_ids: set[int],
) -> str:
    """Classify an empty mask in priority order; no random selection occurs here."""
    if not contact_zone_valid:
        return "CONTACT_OUT_OF_ZONE"
    if not contact_reachable:
        return "CONTACT_UNREACHABLE"
    allowed = set(int(value) for value in native_allowed_ids)
    if not allowed:
        return REASON_NATIVE_EMPTY
    if not (allowed & set(mapped_native_ids)):
        if allowed & set(NATIVE_FALLBACK_IDS):
            return REASON_TAXONOMY_GAP
        return "UNSUPPORTED_NATIVE_CATEGORY_SET"
    return REASON_GEOMETRY_EMPTY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert native/NumPy values recursively, rejecting non-finite numbers."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"Non-primitive dict key type {type(key).__name__}")
            if isinstance(key, float) and not math.isfinite(key):
                raise ValueError("Dict key is a non-finite float")
            key_text = str(key)
            if key_text in converted:
                raise ValueError(f"Duplicate dict key after string conversion: {key_text!r}")
            converted[key_text] = _json_safe(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Native state contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize native value of type {type(value).__name__}")


def _state_signature(state: Any) -> str:
    payload = {
        "previous_native_shot_id": int(state[0]),
        "player_position": [round(float(v), 7) for v in state[1]],
        "opponent_position": [round(float(v), 7) for v in state[2]],
        "shuttle_position": [round(float(v), 7) for v in state[3]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _landing_centre_m(cell_id: int) -> tuple[float, float]:
    row, col = divmod(cell_id, 3)
    x = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
    y = (row + 0.5) * (HALF_LENGTH_M / 5.0)
    return float(x), float(y)


def _recovery_centre_m(cell_id: int) -> tuple[float, float]:
    row, col = divmod(cell_id, 3)
    x = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
    y = -(row + 0.5) * (HALF_LENGTH_M / 3.0)
    return float(x), float(y)


def _metres_to_native(point: tuple[float, float]) -> tuple[float, float]:
    return (
        float(point[0] * NATIVE_HALF_WIDTH / HALF_WIDTH_M),
        float(point[1] * NATIVE_HALF_LENGTH / HALF_LENGTH_M),
    )


def _native_to_metres(point: tuple[float, float]) -> tuple[float, float]:
    """Inverse of `_metres_to_native`: x * 2.59 / 127.5, y * 6.70 / 330.

    Pure unit conversion: no clipping, no sign flip, no [-1, 1] scaling.
    """
    return (
        float(point[0] * HALF_WIDTH_M / NATIVE_HALF_WIDTH),
        float(point[1] * HALF_LENGTH_M / NATIVE_HALF_LENGTH),
    )


def _build_candidate(model_id: int, hit: tuple[float, float], landing: tuple[float, float], recovery: tuple[float, float]) -> tuple[Any, ...]:
    if not 1 <= model_id <= 10:
        raise ValueError(f"Normal/fallback action model ID must be in [1,10], got {model_id}")
    probability = [0.0] * 10
    probability[model_id - 1] = 1.0
    return (int(model_id), tuple(hit), tuple(landing), tuple(recovery), tuple(probability))


def actions_equal(left: Any, right: Any) -> bool:
    """Tolerance-based equality of two native 5-field action tuples.

    Body moved verbatim from `CoachAIWorker._actions_equal` so that the capture
    module can use it without importing the core module.
    """
    if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)) or len(left) != 5 or len(right) != 5:
        return False
    try:
        if int(left[0]) != int(right[0]):
            return False
        for index in (1, 2, 3):
            if not all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-8) for a, b in zip(left[index], right[index])):
                return False
        return bool(
            len(left[4]) == len(right[4])
            and all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-8) for a, b in zip(left[4], right[4]))
        )
    except (TypeError, ValueError, IndexError):
        return False