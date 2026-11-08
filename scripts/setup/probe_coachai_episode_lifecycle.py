#!/usr/bin/env python3
"""Probe CoachAI+ reset semantics across consecutive one-rally episodes.

Run from the badminton-tactical-rl project root:

    .venv-coachai/bin/python scripts/setup/probe_coachai_episode_lifecycle.py

This diagnostic compares two rallies in one native BadmintonEnv instance with
one rally in a fresh instance. It does not modify CoachAI+ source. CSV output
is written to a temporary directory and removed automatically.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_episode_lifecycle_probe.json"
PROBE_HELPER = PROJECT_ROOT / "scripts" / "setup" / "probe_coachai_runtime.py"


def load_probe_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("b2_runtime_probe", PROBE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runtime probe helpers from {PROBE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def construct_env(env_class: Any, agent_a: Any, agent_b: Any, player: str,
                  opponent: str, filepath: Path) -> Any:
    env = env_class(
        agent_a,
        agent_b,
        player,
        opponent,
        1,
        is_match=True,
        is_constraint=True,
        is_have_serve_state=False,
        filepath=str(filepath),
    )
    # The canonical action space has no service action. Make B serve so A's
    # first policy decision is a receive/return decision.
    env.serving_player = "B"
    return env


def complete_one_rally(env: Any, agent_a: Any, max_decisions: int) -> dict[str, Any]:
    score_before = {"A": int(env.score_a), "B": int(env.score_b)}
    output_rows_before = int(len(env.output))
    rally_before = int(env.rally)

    env.serving_player = "B"
    reset_started = time.perf_counter()
    reset_result = env.reset()
    reset_ms = (time.perf_counter() - reset_started) * 1000.0
    if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
        raise RuntimeError(
            "Unexpected reset return; expected four values, got "
            f"{type(reset_result).__name__} length="
            f"{len(reset_result) if isinstance(reset_result, (tuple, list)) else 'n/a'}"
        )

    state, info, done, launch = reset_result
    reset_snapshot = {
        "return_length": len(reset_result),
        "done": bool(done),
        "launch": bool(launch),
        "score_after_reset": {"A": int(env.score_a), "B": int(env.score_b)},
        "rally_after_reset": int(env.rally),
        "history_length": len(info.get("state", [])),
        "player_history": list(info.get("player", [])),
        "reset_ms": round(reset_ms, 3),
    }

    decision_count = 0
    native_rewards: list[float | None] = []
    while not bool(done) and decision_count < max_decisions:
        action = agent_a.action(state, info, launch)
        step_result = env.step(action, launch)
        if not isinstance(step_result, (tuple, list)) or len(step_result) != 5:
            raise RuntimeError(
                "Externally controlled A step must return five values; got "
                f"{type(step_result).__name__} length="
                f"{len(step_result) if isinstance(step_result, (tuple, list)) else 'n/a'}"
            )
        state, native_reward, info, done, launch = step_result
        native_rewards.append(float(native_reward))
        decision_count += 1

    if not bool(done):
        raise RuntimeError(f"Rally did not terminate within {max_decisions} A decisions")

    score_after = {"A": int(env.score_a), "B": int(env.score_b)}
    delta = {side: score_after[side] - score_before[side] for side in ("A", "B")}
    if delta == {"A": 1, "B": 0}:
        winner_side = "A"
    elif delta == {"A": 0, "B": 1}:
        winner_side = "B"
    else:
        winner_side = None

    output_rows_after = int(len(env.output))
    final_record = None
    if output_rows_after:
        final_record = env.output.iloc[-1].to_dict()
        final_record = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in final_record.items()
        }

    return {
        "rally_before_reset": rally_before,
        "score_before_reset": score_before,
        "reset": reset_snapshot,
        "decision_count": decision_count,
        "terminated": True,
        "score_after_rally": score_after,
        "score_delta": delta,
        "winner_side_from_score_delta": winner_side,
        "winner_name_from_final_record": (
            final_record.get("getpoint_player") if final_record else None
        ),
        "output_rows_before": output_rows_before,
        "output_rows_after": output_rows_after,
        "output_rows_added": output_rows_after - output_rows_before,
        "native_rewards": native_rewards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--player", default="CHOU Tien Chen")
    parser.add_argument("--opponent", default="Kento MOMOTA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()

    report: dict[str, Any] = {
        "probe_version": "b2_episode_lifecycle_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "coachai_root": str(root),
        "python_version": platform.python_version(),
        "seed": args.seed,
        "players": {"A": args.player, "B": args.opponent},
    }

    try:
        helper = load_probe_helpers()
        checkout = helper.verify_checkout(root)
        report["coachai_checkout"] = checkout
        sys.path[:0] = [str(root), str(root / "BadmintonEnv")]
        os.chdir(root)

        import torch
        from BadmintonEnv.Environment import BadmintonEnv

        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        helper.install_cpu_torch_load_fallback(torch, device)
        helper.seed_everything(args.seed, torch)
        agent_a, agent_b, target_players, _ = helper.load_rallynet_agents(
            root, torch, args.player, args.opponent, device
        )
        report["runtime"] = {
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "agent_a_training_mode": bool(agent_a.training),
            "agent_b_training_mode": bool(agent_b.training),
            "target_player_count": len(target_players),
        }

        with tempfile.TemporaryDirectory(prefix="coachai_b2_lifecycle_") as temp:
            temp_root = Path(temp)
            same_env_start = time.perf_counter()
            same_env = construct_env(
                BadmintonEnv, agent_a, agent_b, args.player, args.opponent,
                temp_root / "same_env.csv",
            )
            same_env_constructor_ms = (time.perf_counter() - same_env_start) * 1000.0
            try:
                same_env_rallies = []
                for index in range(2):
                    same_env_rallies.append(
                        complete_one_rally(same_env, agent_a, args.max_decisions)
                    )
            finally:
                same_env.close()

            helper.seed_everything(args.seed, torch)
            fresh_env_start = time.perf_counter()
            fresh_env = construct_env(
                BadmintonEnv, agent_a, agent_b, args.player, args.opponent,
                temp_root / "fresh_env.csv",
            )
            fresh_env_constructor_ms = (time.perf_counter() - fresh_env_start) * 1000.0
            try:
                fresh_rally = complete_one_rally(
                    fresh_env, agent_a, args.max_decisions
                )
            finally:
                fresh_env.close()

        report["same_environment"] = {
            "constructor_ms": round(same_env_constructor_ms, 3),
            "rallies": same_env_rallies,
            "score_carries_between_resets": (
                same_env_rallies[1]["score_before_reset"]
                != {"A": 0, "B": 0}
            ),
            "output_accumulates_between_rallies": (
                same_env_rallies[1]["output_rows_before"] > 0
            ),
        }
        report["fresh_environment"] = {
            "constructor_ms": round(fresh_env_constructor_ms, 3),
            "rally": fresh_rally,
        }
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "same_environment_constructor_ms": report.get("same_environment", {}).get("constructor_ms"),
        "same_environment_score_carries": report.get("same_environment", {}).get("score_carries_between_resets"),
        "same_environment_output_accumulates": report.get("same_environment", {}).get("output_accumulates_between_rallies"),
        "fresh_environment_constructor_ms": report.get("fresh_environment", {}).get("constructor_ms"),
        "error": report.get("error"),
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
