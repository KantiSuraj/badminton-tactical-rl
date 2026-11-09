"""Opt-in real CoachAI+/RallyNet worker smoke tests.

Run with COACHAI_ROOT and B2_WORKER_PYTHON set. These tests deliberately do
not substitute a stub opponent: they load the pinned RallyNet checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from badminton_rl.adapters.worker_client import WorkerClient, WorkerConfig, WorkerTimeouts


COACHAI_ROOT = os.environ.get("COACHAI_ROOT")
WORKER_PYTHON = os.environ.get("B2_WORKER_PYTHON")
pytestmark = pytest.mark.integration


@pytest.fixture

def real_worker(tmp_path):
    if not COACHAI_ROOT or not WORKER_PYTHON:
        pytest.skip("Set COACHAI_ROOT and B2_WORKER_PYTHON to run real CoachAI+/RallyNet integration")
    project_root = Path(__file__).resolve().parents[1]
    config = WorkerConfig.from_project(
        project_root,
        coachai_root=COACHAI_ROOT,
        worker_python=WORKER_PYTHON,
        player_name=os.environ.get("B2_PLAYER_NAME", "CHOU Tien Chen"),
        opponent_name=os.environ.get("B2_OPPONENT_NAME", "Kento MOMOTA"),
        device=os.environ.get("B2_DEVICE", "auto"),
        worker_seed=42,
        fallback_events_path=tmp_path / "fallback_events.jsonl",
        stderr_log_path=tmp_path / "worker_stderr.log",
    )
    client = WorkerClient(
        config,
        timeouts=WorkerTimeouts(health=180.0, reset=60.0, step=60.0, close=15.0),
    )
    try:
        health = client.start()
        assert health["ready"] is True
        assert health["mode"] == "B2"
        assert health["controlled_side"] == "A"
        assert health["opponent_side"] == "B"
        assert health["service_policy"] == "B_SERVES_A_RECEIVES"
        assert health["action_count"] == 1350
        yield client
    finally:
        client.close()


def _run_one_rally(client: WorkerClient, seed: int, episode_index: int) -> tuple[int, dict]:
    state = client.reset(seed, episode_index=episode_index)
    assert state["reset_return_length"] == 4
    assert state["observation"]["launch"] is False
    assert state["ready_for_policy"] or state["terminated"]
    steps = 0
    while state["ready_for_policy"]:
        mask = state["action_mask"]
        assert len(mask) == 1350 and any(mask)
        action_id = next(index for index, valid in enumerate(mask) if valid)
        state = client.step(action_id)
        steps += 1
        assert state["internal_native_transitions"] >= 1
        assert state["reward"] in (-1.0, 0.0, 1.0)
        assert state["ready_for_policy"] == (not state["terminated"] and not state["truncated"])
        if state["ready_for_policy"]:
            assert any(state["action_mask"]), "worker exposed an all-false policy mask"
        else:
            assert not any(state["action_mask"]), "terminal/truncated worker exposed legal actions"
    assert state["terminated"] or state["truncated"]
    if state["terminated"]:
        assert state["terminal"]["winner_side"] in ("A", "B")
        expected = 1.0 if state["terminal"]["winner_side"] == "A" else -1.0
        observed_reward = state.get("reward", state["terminal"]["b2_rally_reward"])
        assert observed_reward == expected
    return steps, state


def test_real_rallynet_completes_two_repeated_episodes_in_one_worker(real_worker):
    pid = real_worker.pid
    steps_1, terminal_1 = _run_one_rally(real_worker, 1001, 0)
    assert real_worker.pid == pid
    steps_2, terminal_2 = _run_one_rally(real_worker, 1002, 1)
    assert real_worker.pid == pid  # persistent process, RallyNet and constraint reuse
    assert steps_1 >= 0 and steps_2 >= 0
    assert terminal_1["terminated"] or terminal_1["truncated"]
    assert terminal_2["terminated"] or terminal_2["truncated"]


def test_real_rallynet_fresh_process_reset_replay_starts_from_same_seed(tmp_path):
    if not COACHAI_ROOT or not WORKER_PYTHON:
        pytest.skip("Set COACHAI_ROOT and B2_WORKER_PYTHON to run real CoachAI+/RallyNet integration")
    project_root = Path(__file__).resolve().parents[1]
    start_signatures = []
    for process_index in range(2):
        config = WorkerConfig.from_project(
            project_root,
            coachai_root=COACHAI_ROOT,
            worker_python=WORKER_PYTHON,
            player_name=os.environ.get("B2_PLAYER_NAME", "CHOU Tien Chen"),
            opponent_name=os.environ.get("B2_OPPONENT_NAME", "Kento MOMOTA"),
            device=os.environ.get("B2_DEVICE", "auto"),
            worker_seed=42,
            fallback_events_path=tmp_path / f"fallback_{process_index}.jsonl",
            stderr_log_path=tmp_path / f"stderr_{process_index}.log",
        )
        with WorkerClient(
            config,
            timeouts=WorkerTimeouts(health=180.0, reset=60.0, step=60.0, close=15.0),
        ) as client:
            state = client.reset(4343, episode_index=3)
            start_signatures.append((
                state["observation"],
                state["action_mask_sha256"],
            ))
    assert start_signatures[0] == start_signatures[1]
