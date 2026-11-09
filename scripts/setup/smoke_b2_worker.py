#!/usr/bin/env python3
"""Run real B2 worker rallies through the production subprocess client."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from badminton_rl.adapters.worker_client import WorkerClient, WorkerConfig, WorkerTimeouts  # noqa: E402


def run_episode(client: WorkerClient, seed: int, episode_index: int, max_steps: int) -> dict:
    state = client.reset(seed, episode_index=episode_index)
    steps = 0
    while state["ready_for_policy"] and steps < max_steps:
        valid_ids = [index for index, valid in enumerate(state["action_mask"]) if valid]
        if not valid_ids:
            raise RuntimeError("Worker declared policy-ready with an empty action mask")
        # Deterministic legal-action smoke policy; this is not a tactical baseline.
        state = client.step(valid_ids[0])
        steps += 1
    if state["ready_for_policy"]:
        raise RuntimeError(f"Episode exceeded smoke max_steps={max_steps} without terminal/truncation")
    return {
        "episode_index": episode_index,
        "seed": seed,
        "policy_steps": steps,
        "terminated": state["terminated"],
        "truncated": state["truncated"],
        "reward": state.get("reward", 0.0),
        "terminal": state.get("terminal"),
        "score": state.get("score"),
        "final_mask_true_count": sum(state["action_mask"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coachai-root", type=Path, default=Path(os.environ.get(
        "COACHAI_ROOT", PROJECT_ROOT.parent / "external" / "CoachAI-Plus")))
    parser.add_argument("--worker-python", type=Path, default=Path(os.environ.get(
        "B2_WORKER_PYTHON", PROJECT_ROOT / ".venv-coachai" / "bin" / "python")))
    parser.add_argument("--player-name", default=os.environ.get("B2_PLAYER_NAME", "CHOU Tien Chen"))
    parser.add_argument("--opponent-name", default=os.environ.get("B2_OPPONENT_NAME", "Kento MOMOTA"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=os.environ.get("B2_DEVICE", "auto"))
    parser.add_argument("--worker-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--fallback-events", type=Path, default=PROJECT_ROOT / "reports" / "b2_fallback_events.jsonl")
    parser.add_argument("--stderr-log", type=Path, default=PROJECT_ROOT / "reports" / "b2_worker_stderr.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("--episodes and --max-steps must be positive")
    config = WorkerConfig.from_project(
        PROJECT_ROOT,
        coachai_root=args.coachai_root,
        worker_python=args.worker_python,
        player_name=args.player_name,
        opponent_name=args.opponent_name,
        device=args.device,
        worker_seed=args.worker_seed,
        fallback_events_path=args.fallback_events,
        stderr_log_path=args.stderr_log,
    )
    summaries = []
    with WorkerClient(
        config,
        timeouts=WorkerTimeouts(health=180.0, reset=60.0, step=60.0, close=15.0),
    ) as client:
        health = client.start()
        for episode_index in range(args.episodes):
            summaries.append(run_episode(
                client, args.seed_start + episode_index, episode_index, args.max_steps
            ))
    print(json.dumps({"status": "PASS", "health": health, "episodes": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
