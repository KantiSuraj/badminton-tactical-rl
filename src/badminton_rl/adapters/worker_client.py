"""Synchronous, fail-closed parent-side client for the B2 CoachAI+ worker.

The client owns one subprocess per environment instance.  It never retries a
state-mutating `step` request after a timeout or ambiguous transport failure.
A broken child may be recreated only on the next explicit `reset` call.
"""
from __future__ import annotations

import math
import numbers
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    DEFAULT_MAX_LINE_BYTES,
    ProtocolError,
    RemoteWorkerError,
    decode_json_line,
    encode_json_line,
    make_request,
    validate_action_id,
    validate_action_mask,
    validate_response,
)


class WorkerClientError(RuntimeError):
    """Base class for parent/worker lifecycle failures."""


class WorkerTimeoutError(WorkerClientError):
    pass


class WorkerCrashedError(WorkerClientError):
    pass


class WorkerNotReadyError(WorkerClientError):
    pass


class MaskedActionError(WorkerClientError, ValueError):
    pass


@dataclass(frozen=True)
class WorkerTimeouts:
    health: float = 120.0
    reset: float = 30.0
    step: float = 15.0
    close: float = 5.0

    def __post_init__(self) -> None:
        for name in ("health", "reset", "step", "close"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"Timeout for {name!r} must be a positive finite number")

    def for_operation(self, op: str) -> float:
        value = getattr(self, op, None)
        if value is None:
            raise ValueError(f"No timeout configured for operation {op!r}")
        return float(value)


@dataclass(frozen=True)
class WorkerConfig:
    project_root: Path
    coachai_root: Path
    worker_python: Path
    worker_script: Path | None = None
    player_name: str = "CHOU Tien Chen"
    opponent_name: str = "Kento MOMOTA"
    device: str = "auto"
    worker_seed: int = 42
    expected_coachai_commit: str = "4fd05934f172eec16c74d05c2902983b64c8521e"
    fallback_events_path: Path | None = None
    stderr_log_path: Path | None = None
    worker_id: str | None = None
    max_decisions: int = 64
    max_internal_transitions: int = 64
    max_reset_retries: int = 0
    cell_jitter_fraction: float = 0.0
    fallback_win_reward: float = 1.0
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    extra_cli_args: tuple[str, ...] = field(default_factory=tuple)

    def resolved(self) -> "WorkerConfig":
        if isinstance(self.worker_seed, bool) or not isinstance(self.worker_seed, numbers.Integral):
            raise ValueError("worker_seed must be an integer")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if not self.player_name.strip() or not self.opponent_name.strip():
            raise ValueError("player_name and opponent_name must be non-empty")
        for name, value in (
            ("max_decisions", self.max_decisions),
            ("max_internal_transitions", self.max_internal_transitions),
            ("max_reset_retries", self.max_reset_retries),
            ("max_line_bytes", self.max_line_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
        if self.max_decisions <= 0 or self.max_internal_transitions <= 0:
            raise ValueError("max_decisions and max_internal_transitions must be positive")
        if self.max_reset_retries < 0:
            raise ValueError("max_reset_retries must be non-negative")
        if not (0.0 <= float(self.cell_jitter_fraction) <= 0.45):
            raise ValueError("cell_jitter_fraction must be in [0.0, 0.45]")
        if self.max_line_bytes < 1024:
            raise ValueError("max_line_bytes must be at least 1024")
        root = Path(self.project_root).expanduser().resolve()
        worker_script = (
            Path(self.worker_script).expanduser().resolve()
            if self.worker_script is not None
            else root / "src" / "badminton_rl" / "adapters" / "coachai_worker.py"
        )
        suffix = f"_{self.worker_id}" if self.worker_id else ""
        fallback_log = (
            Path(self.fallback_events_path).expanduser().resolve()
            if self.fallback_events_path is not None
            else root / "reports" / f"b2_fallback_events{suffix}.jsonl"
        )
        stderr_log = (
            Path(self.stderr_log_path).expanduser().resolve()
            if self.stderr_log_path is not None
            else root / "reports" / f"b2_worker_stderr{suffix}.log"
        )
        return WorkerConfig(
            project_root=root,
            coachai_root=Path(self.coachai_root).expanduser().resolve(),
            # Do NOT call Path.resolve() here.  Virtualenv interpreters are
            # commonly symlinks (e.g. .venv-coachai/bin/python -> python3.12).
            # Resolving that symlink launches /usr/bin/python3.12 directly and
            # bypasses the virtualenv's pyvenv.cfg/site-packages.  abspath keeps
            # the invocation path absolute while preserving the symlink.
            worker_python=Path(os.path.abspath(os.path.expanduser(os.fspath(self.worker_python)))),
            worker_script=worker_script,
            player_name=self.player_name,
            opponent_name=self.opponent_name,
            device=self.device,
            worker_seed=int(self.worker_seed),
            expected_coachai_commit=self.expected_coachai_commit,
            fallback_events_path=fallback_log,
            stderr_log_path=stderr_log,
            worker_id=self.worker_id,
            max_decisions=int(self.max_decisions),
            max_internal_transitions=int(self.max_internal_transitions),
            max_reset_retries=int(self.max_reset_retries),
            cell_jitter_fraction=float(self.cell_jitter_fraction),
            fallback_win_reward=float(self.fallback_win_reward),
            max_line_bytes=int(self.max_line_bytes),
            extra_cli_args=tuple(self.extra_cli_args),
        )

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        *,
        coachai_root: str | Path | None = None,
        worker_python: str | Path | None = None,
        **kwargs: Any,
    ) -> "WorkerConfig":
        root = Path(project_root).expanduser().resolve()
        external = (
            Path(coachai_root).expanduser()
            if coachai_root is not None
            else root.parent / "external" / "CoachAI-Plus"
        )
        python_path = (
            Path(worker_python).expanduser()
            if worker_python is not None
            else root / ".venv-coachai" / "bin" / "python"
        )
        return cls(
            project_root=root,
            coachai_root=external,
            worker_python=python_path,
            **kwargs,
        )


class WorkerClient:
    """One-outstanding-request JSONL client with explicit recovery semantics."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        timeouts: WorkerTimeouts | None = None,
        auto_start: bool = False,
        stderr_tail_lines: int = 120,
        stderr_rotate_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.config = config.resolved()
        self.timeouts = timeouts or WorkerTimeouts()
        self._lock = threading.RLock()
        self._responses: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._request_counter = 0
        self._broken = False
        self._closed = False
        self._started_once = False
        self._ready_for_policy = False
        self._action_mask: list[bool] | None = None
        self._health_payload: dict[str, Any] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=max(10, int(stderr_tail_lines)))
        self._stderr_lock = threading.Lock()
        self._stderr_rotate_bytes = max(1024 * 1024, int(stderr_rotate_bytes))
        self._restart_count = 0
        if auto_start:
            self.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def health_payload(self) -> Mapping[str, Any] | None:
        return self._health_payload

    @property
    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_tail)

    @property
    def action_mask(self) -> tuple[bool, ...] | None:
        return tuple(self._action_mask) if self._action_mask is not None else None

    @property
    def ready_for_policy(self) -> bool:
        return self._ready_for_policy

    def start(self) -> Mapping[str, Any]:
        with self._lock:
            if self._closed:
                raise WorkerClientError("WorkerClient is closed; create a new client instance")
            if self._process is not None and not self._broken and self._process.poll() is None:
                return self._health_payload or self._request("health", {})["payload"]
            if self._started_once:
                raise WorkerCrashedError(
                    "Worker has failed; call reset() to explicitly start a fresh rally worker"
                )
            self._spawn()
            response = self._request("health", {}, timeout=self.timeouts.health)
            payload = response["payload"]
            if payload.get("ready") is not True:
                self._invalidate("worker health response did not report ready", terminate=True)
                raise WorkerNotReadyError(f"Worker did not become ready: {payload!r}")
            self._health_payload = payload
            self._started_once = True
            return payload

    def health(self) -> Mapping[str, Any]:
        with self._lock:
            self._ensure_started_for_read()
            response = self._request("health", {}, timeout=self.timeouts.health)
            self._health_payload = response["payload"]
            return self._health_payload

    def reset(self, seed: int, *, episode_index: int | None = None, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Start a fresh rally.  This is the only operation that may respawn a broken worker."""
        with self._lock:
            if self._closed:
                raise WorkerClientError("Cannot reset a closed WorkerClient")
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                raise ValueError("reset seed must be an integer")
            seed = int(seed)
            if episode_index is not None:
                if isinstance(episode_index, bool) or not isinstance(episode_index, numbers.Integral) or episode_index < 0:
                    raise ValueError("episode_index must be a non-negative integer")
                episode_index = int(episode_index)
            if options is not None and not isinstance(options, Mapping):
                raise ValueError("reset options must be a mapping")
            if self._process is None or self._broken or self._process.poll() is not None:
                self._spawn()
                health = self._request("health", {}, timeout=self.timeouts.health)["payload"]
                if health.get("ready") is not True:
                    self._invalidate("worker not ready after restart", terminate=True)
                    raise WorkerNotReadyError(f"Worker not ready after restart: {health!r}")
                self._health_payload = health
                if self._started_once:
                    self._restart_count += 1
                self._started_once = True
            payload: dict[str, Any] = {"seed": seed}
            if episode_index is not None:
                payload["episode_index"] = int(episode_index)
            if options is not None:
                payload["options"] = dict(options)
            response = self._request("reset", payload, timeout=self.timeouts.reset)
            result = response["payload"]
            self._accept_state_payload(result)
            return result

    def step(self, action_id: int) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise WorkerClientError("Cannot step a closed WorkerClient")
            if self._process is None or self._broken or self._process.poll() is not None:
                raise WorkerCrashedError("Worker is not alive; call reset() to start a fresh rally worker")
            action = validate_action_id(action_id)
            if not self._ready_for_policy or self._action_mask is None:
                raise WorkerNotReadyError("Worker is not waiting for a policy action; reset or terminal state")
            if not self._action_mask[action]:
                raise MaskedActionError(f"Action {action} is masked in the current worker state")
            try:
                response = self._request("step", {"action_id": action}, timeout=self.timeouts.step)
            except (WorkerTimeoutError, WorkerCrashedError, ProtocolError):
                # `step` mutates native state; its outcome is ambiguous. Never retry it.
                self._invalidate("step transport failure; episode aborted", terminate=True)
                self._ready_for_policy = False
                self._action_mask = None
                raise
            result = response["payload"]
            self._accept_state_payload(result)
            return result

    def _accept_state_payload(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise ProtocolError("Worker state payload must be an object")
        ready = payload.get("ready_for_policy")
        terminated = payload.get("terminated")
        truncated = payload.get("truncated")
        if type(ready) is not bool:
            raise ProtocolError("Worker state payload needs Boolean ready_for_policy")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ProtocolError("Worker state payload needs Boolean terminated and truncated fields")
        if terminated and truncated:
            raise ProtocolError("A worker state cannot be both terminated and truncated")
        mask = validate_action_mask(payload.get("action_mask"))
        if ready and (terminated or truncated):
            raise ProtocolError("Worker cannot be policy-ready after termination/truncation")
        if ready and not any(mask):
            raise ProtocolError("Worker marked a state ready while its action mask is all false")
        if not ready and any(mask):
            raise ProtocolError("Worker returned a non-empty action mask for a non-policy state")
        if not ready and not (terminated or truncated):
            raise ProtocolError("Worker returned no policy decision without terminal/truncation status")
        self._ready_for_policy = ready
        self._action_mask = mask

    def _ensure_started_for_read(self) -> None:
        if self._closed:
            raise WorkerClientError("WorkerClient is closed")
        if self._process is None or self._broken or self._process.poll() is not None:
            raise WorkerCrashedError("Worker is not alive; call reset() to start a fresh worker")

    def _spawn(self) -> None:
        if self._closed:
            raise WorkerClientError("Cannot spawn a closed WorkerClient")
        if self._process is not None:
            self._stop_process(force=True)
        cfg = self.config
        if not cfg.worker_python.is_file():
            raise FileNotFoundError(f"Worker Python interpreter not found: {cfg.worker_python}")
        if cfg.worker_script is None or not cfg.worker_script.is_file():
            raise FileNotFoundError(f"Worker script not found: {cfg.worker_script}")
        if not cfg.project_root.is_dir():
            raise FileNotFoundError(f"Project root not found: {cfg.project_root}")
        if not cfg.coachai_root.is_dir():
            raise FileNotFoundError(f"CoachAI+ checkout not found: {cfg.coachai_root}")
        if cfg.max_line_bytes < 1024:
            raise ValueError("max_line_bytes must be at least 1024")

        args = [
            str(cfg.worker_python), str(cfg.worker_script),
            "--project-root", str(cfg.project_root),
            "--coachai-root", str(cfg.coachai_root),
            "--player-name", cfg.player_name,
            "--opponent-name", cfg.opponent_name,
            "--device", cfg.device,
            "--worker-seed", str(cfg.worker_seed),
            "--expected-coachai-commit", cfg.expected_coachai_commit,
            "--fallback-events", str(cfg.fallback_events_path),
            "--max-decisions", str(cfg.max_decisions),
            "--max-internal-transitions", str(cfg.max_internal_transitions),
            "--max-reset-retries", str(cfg.max_reset_retries),
            "--cell-jitter-fraction", str(cfg.cell_jitter_fraction),
            "--fallback-win-reward", str(cfg.fallback_win_reward),
            "--max-line-bytes", str(cfg.max_line_bytes),
            *cfg.extra_cli_args,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        src = str(cfg.project_root / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        assert cfg.stderr_log_path is not None
        cfg.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.stderr_log_path.exists() and cfg.stderr_log_path.stat().st_size > self._stderr_rotate_bytes:
            rotated = cfg.stderr_log_path.with_suffix(cfg.stderr_log_path.suffix + ".1")
            try:
                rotated.unlink(missing_ok=True)
                cfg.stderr_log_path.replace(rotated)
            except OSError:
                pass
        self._responses = queue.Queue()
        with self._stderr_lock:
            self._stderr_tail.clear()
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(cfg.project_root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=-1,
            )
        except OSError as exc:
            raise WorkerClientError(f"Could not start CoachAI worker: {exc}") from exc
        self._process = proc
        self._broken = False
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, name=f"b2-worker-stdout-{proc.pid}", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, name=f"b2-worker-stderr-{proc.pid}", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            self._responses.put(None)
            return
        try:
            while True:
                line = proc.stdout.readline(self.config.max_line_bytes + 1)
                if not line:
                    break
                if len(line) > self.config.max_line_bytes:
                    # Drain the rest of this physical line before reporting the violation.
                    while line and not line.endswith(b"\n"):
                        line = proc.stdout.readline(self.config.max_line_bytes + 1)
                    self._responses.put(ProtocolError("Worker stdout line exceeded configured size"))
                    continue
                self._responses.put(line)
        except BaseException as exc:  # reader thread must surface transport errors to caller
            self._responses.put(exc)
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        path = self.config.stderr_log_path
        assert path is not None
        try:
            with path.open("ab") as handle:
                while True:
                    chunk = proc.stderr.readline(8192)
                    if not chunk:
                        break
                    handle.write(chunk)
                    handle.flush()
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                    except Exception:
                        text = repr(chunk)
                    with self._stderr_lock:
                        self._stderr_tail.append(text)
                    try:
                        if handle.tell() >= self._stderr_rotate_bytes:
                            handle.close()
                            rotated = path.with_suffix(path.suffix + ".1")
                            rotated.unlink(missing_ok=True)
                            path.replace(rotated)
                            handle = path.open("ab")
                    except OSError:
                        # Diagnostics must never block protocol handling.
                        pass
        except OSError:
            pass

    def _request(self, op: str, payload: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        proc = self._process
        if proc is None or self._broken:
            raise WorkerCrashedError("Worker process is unavailable")
        if proc.poll() is not None:
            raise WorkerCrashedError(self._crash_message("Worker exited before request"))
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                break
        self._request_counter += 1
        request_id = f"w{proc.pid}-r{self._request_counter:08d}"
        request = make_request(request_id, op, payload)
        encoded = encode_json_line(request, max_bytes=self.config.max_line_bytes)
        if proc.stdin is None:
            raise WorkerCrashedError("Worker stdin is closed")
        try:
            proc.stdin.write(encoded)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._invalidate("worker stdin failed", terminate=True)
            raise WorkerCrashedError(self._crash_message(f"Could not write {op} request: {exc}")) from exc

        wait_for = timeout if timeout is not None else self.timeouts.for_operation(op)
        try:
            item = self._responses.get(timeout=wait_for)
        except queue.Empty as exc:
            self._invalidate(f"{op} request timed out after {wait_for:.3f}s", terminate=True)
            raise WorkerTimeoutError(
                self._crash_message(f"Worker {op} timed out after {wait_for:.3f}s; request was not retried")
            ) from exc
        if item is None:
            self._invalidate("worker stdout closed", terminate=True)
            raise WorkerCrashedError(self._crash_message(f"Worker exited while handling {op}"))
        if isinstance(item, BaseException):
            self._invalidate("worker stdout reader failed", terminate=True)
            raise WorkerCrashedError(self._crash_message(f"Worker stdout read failed: {item}")) from item
        try:
            decoded = decode_json_line(item, max_bytes=self.config.max_line_bytes)
            response = validate_response(decoded, request_id)
        except ProtocolError:
            self._invalidate("worker protocol violation", terminate=True)
            raise
        if not response["ok"]:
            error = response["error"]
            if error.get("fatal"):
                self._invalidate(f"fatal worker error: {error['code']}", terminate=True)
            raise RemoteWorkerError(
                error["code"], error["message"],
                fatal=bool(error.get("fatal")), details=error.get("details"),
            )
        return response

    def _crash_message(self, prefix: str) -> str:
        proc = self._process
        status = proc.poll() if proc is not None else None
        stderr = self.stderr_tail[-6000:]
        return f"{prefix}; worker_return_code={status}; stderr_tail={stderr!r}"

    def _invalidate(self, reason: str, *, terminate: bool) -> None:
        self._broken = True
        self._ready_for_policy = False
        self._action_mask = None
        if terminate:
            self._stop_process(force=True)

    def _stop_process(self, *, force: bool) -> None:
        proc = self._process
        self._process = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            if proc.poll() is None and force:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except OSError:
            pass
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=0.5)
        self._stdout_thread = None
        self._stderr_thread = None
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            proc = self._process
            if proc is not None and not self._broken and proc.poll() is None:
                try:
                    self._request("close", {}, timeout=self.timeouts.close)
                except (WorkerClientError, ProtocolError, RemoteWorkerError):
                    pass
            self._stop_process(force=True)
            self._closed = True
            self._ready_for_policy = False
            self._action_mask = None

    def __enter__(self) -> "WorkerClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort only; callers should use close/context manager
        try:
            if hasattr(self, "_closed") and not self._closed:
                self.close()
        except Exception:
            pass
