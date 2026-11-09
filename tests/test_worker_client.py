"""Subprocess lifecycle tests using a deterministic fake JSONL worker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from badminton_rl.adapters.protocol import N_ACTIONS, RemoteWorkerError
from badminton_rl.adapters.worker_client import (
    MaskedActionError,
    WorkerClient,
    WorkerConfig,
    WorkerCrashedError,
    WorkerTimeoutError,
    WorkerTimeouts,
)


FAKE_WORKER = r'''#!/usr/bin/env python3
import json
import os
import sys
import time

args = sys.argv[1:]
def arg(name, default=None):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return default
mode = arg("--mode", "normal")
counter = arg("--counter")
mask = [False] * 1350
mask[0] = True
terminal_mask = [False] * 1350

def send(response):
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    if op == "health":
        if mode == "bad_health":
            sys.stdout.write("not-json\n"); sys.stdout.flush(); continue
        payload = {"ready": True, "fake": True}
        response = {"protocol_version": 1, "request_id": request["request_id"], "ok": True, "payload": payload}
    elif op == "reset":
        payload = {
            "status": "ready_for_policy", "ready_for_policy": True,
            "terminated": False, "truncated": False,
            "action_mask": mask, "observation": {"fake": True},
            "episode_index": request["payload"].get("episode_index"),
            "episode_seed": request["payload"]["seed"],
        }
        response = {"protocol_version": 1, "request_id": request["request_id"], "ok": True, "payload": payload}
    elif op == "step":
        if counter:
            with open(counter, "a", encoding="utf-8") as out:
                out.write("step\n")
        if mode == "timeout_step":
            time.sleep(30)
        if mode == "crash_step":
            os._exit(9)
        if mode == "remote_error":
            response = {
                "protocol_version": 1, "request_id": request["request_id"],
                "ok": False, "error": {"code": "masked_action_selected", "message": "no", "fatal": False},
            }
        elif mode == "bad_correlation":
            response = {
                "protocol_version": 1, "request_id": "wrong-id", "ok": True,
                "payload": {"ready_for_policy": True, "terminated": False, "truncated": False, "action_mask": mask},
            }
        else:
            payload = {
                "ready_for_policy": False, "terminated": True, "truncated": False,
                "action_mask": terminal_mask, "observation": {"fake": True},
                "reward": -1.0, "terminal": {"winner_side": "B"},
            }
            response = {"protocol_version": 1, "request_id": request["request_id"], "ok": True, "payload": payload}
    elif op == "close":
        response = {
            "protocol_version": 1, "request_id": request["request_id"], "ok": True,
            "payload": {"closed": True},
        }
        send(response)
        break
    else:
        response = {
            "protocol_version": 1, "request_id": request["request_id"], "ok": False,
            "error": {"code": "unknown", "message": "unknown", "fatal": False},
        }
    send(response)
'''


def make_client(tmp_path: Path, *, mode: str = "normal", timeout: float = 1.0) -> tuple[WorkerClient, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir(exist_ok=True)
    coachai_root = tmp_path / "coachai"
    coachai_root.mkdir(exist_ok=True)
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    counter = tmp_path / "worker_calls.txt"
    config = WorkerConfig(
        project_root=project_root,
        coachai_root=coachai_root,
        worker_python=Path(sys.executable),
        worker_script=script,
        fallback_events_path=tmp_path / "events.jsonl",
        stderr_log_path=tmp_path / "stderr.log",
        extra_cli_args=("--mode", mode, "--counter", str(counter)),
    )
    client = WorkerClient(
        config,
        timeouts=WorkerTimeouts(health=max(timeout, 5.0), reset=max(timeout, 5.0), step=timeout, close=max(timeout, 5.0)),
    )
    return client, counter


def test_worker_client_normal_lifecycle_and_action_mask_guard(tmp_path):
    client, _ = make_client(tmp_path)
    health = client.start()
    assert health["fake"] is True
    reset = client.reset(42, episode_index=0)
    assert client.ready_for_policy is True
    assert len(client.action_mask) == N_ACTIONS
    assert client.action_mask[0] is True
    with pytest.raises(MaskedActionError):
        client.step(1)
    assert client.ready_for_policy is True  # local rejection did not mutate worker state
    stepped = client.step(0)
    assert stepped["terminated"] is True
    assert stepped["reward"] == -1.0
    assert client.ready_for_policy is False
    client.close()
    client.close()  # idempotent
    with pytest.raises(Exception):
        client.reset(43)


def test_worker_client_rejects_step_before_reset_and_after_terminal(tmp_path):
    client, _ = make_client(tmp_path)
    client.start()
    with pytest.raises(Exception):
        client.step(0)
    client.reset(1)
    client.step(0)
    with pytest.raises(Exception):
        client.step(0)
    client.close()


def test_step_timeout_is_never_retried_and_next_reset_restarts_child(tmp_path):
    client, counter = make_client(tmp_path, mode="timeout_step", timeout=0.2)
    client.start()
    client.reset(1)
    with pytest.raises(WorkerTimeoutError):
        client.step(0)
    with pytest.raises(WorkerCrashedError, match="call reset"):
        client.start()
    assert counter.read_text(encoding="utf-8").splitlines() == ["step"]
    assert client.ready_for_policy is False
    restarted = client.reset(2, episode_index=1)
    assert restarted["ready_for_policy"] is True
    assert client.restart_count == 1
    client.close()


def test_worker_crash_during_step_is_not_retried(tmp_path):
    client, counter = make_client(tmp_path, mode="crash_step", timeout=1.0)
    client.start()
    client.reset(1)
    with pytest.raises(WorkerCrashedError):
        client.step(0)
    assert counter.read_text(encoding="utf-8").splitlines() == ["step"]
    client.close()


def test_worker_protocol_corruption_fails_closed(tmp_path):
    client, _ = make_client(tmp_path, mode="bad_health")
    with pytest.raises(Exception):
        client.start()
    assert client.pid is None
    client.close()


def test_worker_wrong_request_correlation_fails_closed(tmp_path):
    client, _ = make_client(tmp_path, mode="bad_correlation")
    client.start()
    client.reset(1)
    with pytest.raises(Exception):
        client.step(0)
    assert client.ready_for_policy is False
    client.close()


def test_nonfatal_remote_error_keeps_process_alive_but_does_not_advance_client(tmp_path):
    client, counter = make_client(tmp_path, mode="remote_error")
    client.start()
    client.reset(1)
    with pytest.raises(RemoteWorkerError) as caught:
        client.step(0)
    assert caught.value.code == "masked_action_selected"
    assert client.pid is not None
    # The error was rejected before the client advanced its cached mask/ready state.
    assert client.ready_for_policy is True
    assert counter.read_text(encoding="utf-8").splitlines() == ["step"]
    client.close()


def test_reset_accepts_numpy_integer_seeds_and_episode_indices(tmp_path):
    client, _ = make_client(tmp_path)
    try:
        state = client.reset(np.int64(42), episode_index=np.int64(3))
        assert state["ready_for_policy"] is True
        assert state["episode_index"] == 3
        assert state["episode_seed"] == 42
    finally:
        client.close()


def test_invalid_reset_arguments_fail_before_subprocess_start(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(ValueError):
        client.reset(True)
    with pytest.raises(ValueError):
        client.reset(1, episode_index=-1)
    assert client.pid is None
    client.close()


def test_worker_timeouts_and_config_limits_are_validated():
    with pytest.raises(ValueError):
        WorkerTimeouts(step=0)
    with pytest.raises(ValueError):
        WorkerTimeouts(step=float("nan"))
    with pytest.raises(ValueError):
        WorkerConfig(
            project_root=Path("."), coachai_root=Path("."), worker_python=Path(sys.executable),
            max_decisions=0,
        ).resolved()


def test_worker_config_preserves_virtualenv_interpreter_symlink(tmp_path):
    """Resolving a venv's python symlink bypasses pyvenv.cfg and site-packages."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    coachai_root = tmp_path / "coachai"
    coachai_root.mkdir()
    target_python = Path(sys.executable).resolve()
    venv_bin = tmp_path / ".venv-coachai" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter_link = venv_bin / "python"
    interpreter_link.symlink_to(target_python)

    config = WorkerConfig(
        project_root=project_root,
        coachai_root=coachai_root,
        worker_python=interpreter_link,
    ).resolved()

    assert config.worker_python == interpreter_link.absolute()
    assert config.worker_python.is_symlink()
