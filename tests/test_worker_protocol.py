"""Protocol contract tests; these never import CoachAI+, torch or CUDA."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from badminton_rl.adapters.protocol import (
    DEFAULT_MAX_LINE_BYTES,
    N_ACTIONS,
    PROTOCOL_VERSION,
    ProtocolError,
    RemoteWorkerError,
    decode_json_line,
    encode_json_line,
    error_response,
    make_request,
    success_response,
    validate_action_id,
    validate_action_mask,
    validate_request,
    validate_response,
)


class ExampleEnum(Enum):
    ITEM = "item"


@dataclass
class ExampleRecord:
    value: int


def test_jsonl_round_trip_and_standard_types():
    message = {
        "unicode": "羽毛球",
        "path": Path("reports/events.jsonl"),
        "enum": ExampleEnum.ITEM,
        "record": ExampleRecord(7),
        "tuple": (1, 2),
    }
    encoded = encode_json_line(message)
    assert encoded.endswith(b"\n")
    assert decode_json_line(encoded) == {
        "unicode": "羽毛球",
        "path": "reports/events.jsonl",
        "enum": "item",
        "record": {"value": 7},
        "tuple": [1, 2],
    }


def test_jsonl_rejects_invalid_utf8_empty_non_object_and_nan():
    for invalid in (b"\xff", b"", b"[]", b'{"value":NaN}', b'{"value":Infinity}'):
        with pytest.raises(ProtocolError):
            decode_json_line(invalid)


def test_jsonl_line_size_is_bounded():
    with pytest.raises(ProtocolError, match="exceeds"):
        encode_json_line({"value": "x" * 100}, max_bytes=16)
    with pytest.raises(ProtocolError, match="exceeds"):
        decode_json_line(b"{" + b" " * 20 + b"}", max_bytes=16)
    assert DEFAULT_MAX_LINE_BYTES >= 1024 * 1024


def test_make_and_validate_request_contract():
    request = make_request("r1", "reset", {"seed": 42})
    assert validate_request(request) == request
    assert request["protocol_version"] == PROTOCOL_VERSION
    for bad in (
        {**request, "protocol_version": True},
        {**request, "protocol_version": 999},
        {**request, "request_id": " "},
        {**request, "op": "restart"},
        {**request, "payload": []},
        {**request, "extra": 1},
    ):
        with pytest.raises(ProtocolError):
            validate_request(bad)
    with pytest.raises(ProtocolError):
        make_request("", "health")
    with pytest.raises(ProtocolError):
        make_request("r2", "unknown")


def test_success_and_error_response_contracts():
    success = success_response("r1", {"ready": True})
    assert validate_response(success, "r1")["payload"] == {"ready": True}
    error = error_response("r2", "bad_action", "masked", fatal=False, details={"id": 4})
    assert validate_response(error, "r2")["error"]["code"] == "bad_action"
    with pytest.raises(ProtocolError, match="request_id mismatch"):
        validate_response(success, "different")
    malformed = {**success, "protocol_version": True}
    with pytest.raises(ProtocolError):
        validate_response(malformed, "r1")
    with pytest.raises(ProtocolError):
        validate_response({**success, "extra": "no"}, "r1")


def test_remote_worker_error_preserves_fields():
    exc = RemoteWorkerError("native_state_error", "bad transition", fatal=True, details={"side": "A"})
    assert exc.code == "native_state_error"
    assert exc.fatal is True
    assert exc.details == {"side": "A"}


def test_action_id_strictly_rejects_bool_float_and_out_of_range():
    assert validate_action_id(0) == 0
    assert validate_action_id(N_ACTIONS - 1) == N_ACTIONS - 1
    assert validate_action_id(np.int64(17)) == 17
    for value in (True, False, 1.0, "1", -1, N_ACTIONS):
        with pytest.raises(ProtocolError):
            validate_action_id(value)


def test_action_mask_requires_exact_shape_and_true_booleans():
    mask = [False] * N_ACTIONS
    mask[17] = True
    assert validate_action_mask(mask) == mask
    for invalid in (None, [], [False] * (N_ACTIONS - 1), [False] * (N_ACTIONS + 1),
                    [0] * N_ACTIONS, [False] * (N_ACTIONS - 1) + [1]):
        with pytest.raises(ProtocolError):
            validate_action_mask(invalid)


def test_non_finite_float_rejected_at_encode_boundary():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ProtocolError):
            encode_json_line({"value": value})
