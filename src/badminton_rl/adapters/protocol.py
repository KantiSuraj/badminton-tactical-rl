"""Versioned JSONL protocol shared by the parent and CoachAI+ worker.

This module deliberately imports only the Python standard library.  It is safe to
import in the parent environment, where CoachAI+, PyTorch and CUDA are absent.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_MAX_LINE_BYTES = 4 * 1024 * 1024
SUPPORTED_OPERATIONS = frozenset({"health", "reset", "step", "close"})
N_FAMILIES = 10
N_LANDING = 15
N_RECOVERY = 9
N_ACTIONS = N_FAMILIES * N_LANDING * N_RECOVERY


class ProtocolError(ValueError):
    """A malformed request, response, or non-JSON-safe payload."""


class RemoteWorkerError(RuntimeError):
    """A well-formed error response returned by the worker."""

    def __init__(self, code: str, message: str, *, fatal: bool = False, details: Any = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.fatal = fatal
        self.details = details


def _json_primitive(value: Any, *, path: str = "$") -> Any:
    """Convert common scalar/container types and reject non-finite numbers.

    NumPy and torch objects are handled by duck typing to keep this module free of
    optional numerical dependencies.  Unsupported objects fail closed rather than
    being stringified into misleading observations.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"Non-finite float at {path}")
        return value
    if isinstance(value, enum.Enum):
        return _json_primitive(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_primitive(dataclasses.asdict(value), path=path)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_primitive(value.detach().cpu().tolist(), path=path)
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_primitive(value.tolist(), path=path)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_primitive(value.item(), path=path)
        except (TypeError, ValueError, RuntimeError):
            pass
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ProtocolError(f"Non-primitive mapping key at {path}: {type(key).__name__}")
            key_text = str(key)
            converted[key_text] = _json_primitive(item, path=f"{path}.{key_text}")
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_json_primitive(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ProtocolError(f"Unsupported JSON value at {path}: {type(value).__name__}")


def encode_json_line(message: Mapping[str, Any], *, max_bytes: int = DEFAULT_MAX_LINE_BYTES) -> bytes:
    """Serialize one message as UTF-8 JSON followed by exactly one newline."""
    if not isinstance(message, Mapping):
        raise ProtocolError("JSONL message must be an object")
    primitive = _json_primitive(message)
    try:
        encoded = json.dumps(
            primitive,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Could not serialize JSONL message: {exc}") from exc
    if len(encoded) > max_bytes:
        raise ProtocolError(f"JSONL message exceeds {max_bytes} bytes ({len(encoded)} bytes)")
    return encoded + b"\n"


def decode_json_line(line: bytes | str, *, max_bytes: int = DEFAULT_MAX_LINE_BYTES) -> dict[str, Any]:
    """Decode one bounded JSON object; reject NaN/Infinity and non-object roots."""
    if isinstance(line, str):
        raw = line.encode("utf-8")
    elif isinstance(line, bytes):
        raw = line
    else:
        raise ProtocolError(f"JSONL line must be bytes or str, got {type(line).__name__}")
    if len(raw) > max_bytes:
        raise ProtocolError(f"JSONL line exceeds {max_bytes} bytes ({len(raw)} bytes)")
    try:
        text = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ProtocolError("JSONL line is not valid UTF-8") from exc
    if not text:
        raise ProtocolError("Empty JSONL line")

    def reject_constant(value: str) -> None:
        raise ProtocolError(f"Non-standard JSON numeric constant: {value}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except ProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON at column {exc.colno}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSONL root must be an object")
    return value


def make_request(request_id: str, op: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProtocolError("request_id must be a non-empty string")
    if op not in SUPPORTED_OPERATIONS:
        raise ProtocolError(f"Unsupported operation: {op!r}")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("request payload must be an object")
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "op": op,
        "payload": dict(payload),
    }
    # Validate serializability immediately, before sending to the child process.
    _json_primitive(request)
    return request


def validate_request(message: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ProtocolError("Request must be an object")
    required = {"protocol_version", "request_id", "op", "payload"}
    if set(message) != required:
        missing = sorted(required - set(message))
        extra = sorted(set(message) - required)
        raise ProtocolError(f"Request fields mismatch; missing={missing}, extra={extra}")
    version = message["protocol_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol_version: {version!r}")
    request_id = message["request_id"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProtocolError("request_id must be a non-empty string")
    op = message["op"]
    if not isinstance(op, str) or op not in SUPPORTED_OPERATIONS:
        raise ProtocolError(f"Unknown operation: {op!r}")
    payload = message["payload"]
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    _json_primitive(payload)
    return {"protocol_version": version, "request_id": request_id, "op": op, "payload": payload}


def success_response(request_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProtocolError("Response payload must be an object")
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": True,
        "payload": dict(payload),
    }
    _json_primitive(response)
    return response


def error_response(
    request_id: str,
    code: str,
    message: str,
    *,
    fatal: bool = False,
    details: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": str(code),
        "message": str(message)[:2000],
        "fatal": bool(fatal),
    }
    if details is not None:
        error["details"] = _json_primitive(details)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": error,
    }


def validate_response(message: Mapping[str, Any], expected_request_id: str) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ProtocolError("Response must be an object")
    version = message.get("protocol_version")
    if isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise ProtocolError(f"Worker protocol version mismatch: {version!r}")
    if message.get("request_id") != expected_request_id:
        raise ProtocolError(
            f"Worker request_id mismatch: expected {expected_request_id!r}, "
            f"received {message.get('request_id')!r}"
        )
    if type(message.get("ok")) is not bool:
        raise ProtocolError("Response 'ok' field must be boolean")
    if message["ok"]:
        if set(message) != {"protocol_version", "request_id", "ok", "payload"}:
            raise ProtocolError("Successful response contains missing or unexpected fields")
        if not isinstance(message["payload"], dict):
            raise ProtocolError("Successful response payload must be an object")
        _json_primitive(message["payload"])
    else:
        if set(message) != {"protocol_version", "request_id", "ok", "error"}:
            raise ProtocolError("Error response contains missing or unexpected fields")
        error = message["error"]
        if not isinstance(error, dict) or not {"code", "message", "fatal"}.issubset(error):
            raise ProtocolError("Malformed worker error object")
        if not isinstance(error["code"], str) or not isinstance(error["message"], str):
            raise ProtocolError("Worker error code/message must be strings")
        if type(error["fatal"]) is not bool:
            raise ProtocolError("Worker error fatal field must be boolean")
        _json_primitive(error)
    return dict(message)


def validate_action_id(action_id: Any) -> int:
    """Validate an integer-like macro-action ID, including NumPy integer scalars."""
    if isinstance(action_id, bool) or not isinstance(action_id, numbers.Integral):
        raise ProtocolError(f"action_id must be an integer, got {type(action_id).__name__}")
    value = int(action_id)
    if not 0 <= value < N_ACTIONS:
        raise ProtocolError(f"action_id must be in [0, {N_ACTIONS}), got {value}")
    return value


def validate_action_mask(mask: Any) -> list[bool]:
    if not isinstance(mask, (list, tuple)) or len(mask) != N_ACTIONS:
        raise ProtocolError(f"action_mask must contain exactly {N_ACTIONS} Boolean entries")
    if any(type(value) is not bool for value in mask):
        raise ProtocolError("action_mask entries must be Boolean")
    return list(mask)
