#!/usr/bin/env python3
"""Persistent B2 worker hosting the pinned CoachAI+ native environment.

Only this process imports CoachAI+, NumPy, pandas, PyTorch or RallyNet.  The
parent communicates with this process using the versioned JSONL protocol in
`protocol.py`.  One `step` request represents one external A-side decision;
CoachAI+ invokes RallyNet for B's response internally.

Run directly with the dedicated worker interpreter:

    .venv-coachai/bin/python src/badminton_rl/adapters/coachai_worker.py \
      --coachai-root ../external/CoachAI-Plus

The native CoachAI+ checkout is never modified.  Its output CSV is written to
an automatically cleaned temporary directory owned by this process.
"""
from __future__ import annotations

# Module layout (behaviour-preserving split of the former single file):
#   coachai_worker.py          executable facade: CLI + JSONL protocol loop
#   coachai_worker_core.py     CoachAIWorker state machine
#   coachai_worker_capture.py  OneShotValidationCapture (validator interception)
#   coachai_worker_utils.py    constants, mappings, codecs, pure helpers
# Names that used to live in this file are re-exported below so existing
# `from badminton_rl.adapters.coachai_worker import ...` statements keep working.

import argparse
import atexit
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

# Executing this file by path is the production launch mode, so establish the
# src/ import root before importing the standard parent/worker protocol module.
if __package__ in {None, ""}:
    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from badminton_rl.adapters.protocol import (  # noqa: E402
    DEFAULT_MAX_LINE_BYTES,
    ProtocolError,
    decode_json_line,
    encode_json_line,
    error_response,
    success_response,
    validate_request,
)
from badminton_rl.adapters.coachai_worker_utils import (  # noqa: E402,F401
    ACTION_MAP_VERSION,
    EXPECTED_COACHAI_COMMIT,
    EXPECTED_COMMON_FALLBACK_ZONES,
    EXPECTED_CONSTRAINT_SHA256,
    EXPECTED_FILE_SHA256,
    EXPECTED_POLICY_LANDING_ZONES,
    FAMILY_MAP,
    HALF_LENGTH_M,
    HALF_WIDTH_M,
    MODEL_TO_NATIVE,
    NATIVE_FALLBACK_IDS,
    NATIVE_HALF_LENGTH,
    NATIVE_HALF_WIDTH,
    NATIVE_TO_MODEL,
    N_ACTIONS,
    N_FAMILIES,
    N_LANDING,
    N_RECOVERY,
    OBSERVATION_SCHEMA_VERSION,
    REASON_GEOMETRY_EMPTY,
    REASON_NATIVE_EMPTY,
    REASON_TAXONOMY_GAP,
    UNSUPPORTED_FAMILIES,
    UnexpectedNativeAgentCall,
    WorkerFault,
    _build_candidate,
    _describe_history_structure,
    _json_safe,
    _landing_centre_m,
    _metres_to_native,
    _native_to_metres,
    _recovery_centre_m,
    _sha256_file,
    _state_signature,
    actions_equal,
    classify_empty_mask_reason,
    decode_action,
    derive_action_jitter_seed,
    derive_fallback_seed,
    derive_reset_retry_seed,
    encode_action,
)
from badminton_rl.adapters.coachai_history_serializer import (  # noqa: E402,F401
    NativeTransitionCapture,
    serialize_rally_history,
)
from badminton_rl.adapters.coachai_worker_capture import OneShotValidationCapture  # noqa: E402,F401
from badminton_rl.adapters.coachai_worker_core import (  # noqa: E402,F401
    CoachAIWorker,
    ExternalControlledAgent,
    NativeStepResult,
)


def _dispatch(worker: CoachAIWorker, raw_request: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(raw_request)
    request_id = request["request_id"]
    try:
        payload = worker.handle(request["op"], request["payload"])
        return success_response(request_id, payload)
    except WorkerFault as exc:
        return error_response(
            request_id, exc.code, str(exc), fatal=exc.fatal, details=exc.details
        )
    except (ProtocolError, ValueError, TypeError) as exc:
        return error_response(request_id, "protocol_violation", str(exc), fatal=False)
    except Exception as exc:
        print(f"[coachai_worker] unhandled {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return error_response(request_id, "internal", f"Worker internal error: {exc}", fatal=True)


def serve_jsonl(worker: CoachAIWorker, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> int:
    """Serve one request per input line; stdout is reserved exclusively for JSONL."""
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > max_line_bytes:
            response = error_response("invalid", "protocol_violation", "Request line exceeds configured size")
        else:
            try:
                raw = decode_json_line(raw_line, max_bytes=max_line_bytes)
                response = _dispatch(worker, raw)
            except ProtocolError as exc:
                # Malformed JSON has no trustworthy request ID.  The client will fail
                # closed on the sentinel ID rather than accepting a miscorrelated reply.
                response = error_response("invalid", "protocol_violation", str(exc), fatal=True)
        try:
            sys.stdout.buffer.write(encode_json_line(response, max_bytes=max_line_bytes))
            sys.stdout.buffer.flush()
        except Exception as exc:
            print(f"[coachai_worker] failed writing protocol response: {exc}", file=sys.stderr, flush=True)
            return 2
        if response.get("ok") and response.get("payload", {}).get("closed") is True:
            return 0
        if not response.get("ok") and response.get("error", {}).get("fatal") is True:
            return 3
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--coachai-root", type=Path, required=True)
    parser.add_argument("--player-name", default="CHOU Tien Chen")
    parser.add_argument("--opponent-name", default="Kento MOMOTA")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker-seed", type=int, default=42)
    parser.add_argument("--expected-coachai-commit", default=EXPECTED_COACHAI_COMMIT)
    parser.add_argument("--fallback-events", type=Path, default=Path("reports/b2_fallback_events.jsonl"))
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--max-internal-transitions", type=int, default=64)
    parser.add_argument("--max-reset-retries", type=int, default=0)
    parser.add_argument("--cell-jitter-fraction", type=float, default=0.0)
    parser.add_argument("--fallback-win-reward", type=float, default=1.0)
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_decisions <= 0 or args.max_internal_transitions <= 0:
        print("[coachai_worker] max decision/transition limits must be positive", file=sys.stderr, flush=True)
        return 2
    if args.max_reset_retries < 0 or not (0.0 <= args.cell_jitter_fraction <= 0.45):
        print("[coachai_worker] invalid reset retries or cell jitter fraction", file=sys.stderr, flush=True)
        return 2
    worker = CoachAIWorker(args)
    atexit.register(worker.close)

    def _handle_signal(signum: int, _frame: Any) -> None:
        worker.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        worker.initialize()
        print(
            f"[coachai_worker] ready pid={os.getpid()} device={worker.device} "
            f"coachai_commit={worker.checkout_fingerprint['head']}",
            file=sys.stderr,
            flush=True,
        )
        return serve_jsonl(worker, max_line_bytes=args.max_line_bytes)
    except Exception as exc:
        print(f"[coachai_worker] startup failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        try:
            worker.close()
        except Exception:
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())