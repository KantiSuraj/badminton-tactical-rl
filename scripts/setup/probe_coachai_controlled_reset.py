#!/usr/bin/env python3
"""Test an explicit one-rally reset adapter on one persistent CoachAI+ instance.

Run from the badminton-tactical-rl project root:

    .venv-coachai/bin/python scripts/setup/probe_coachai_controlled_reset.py

The probe keeps the same RallyNet objects, native constraint object and
BadmintonEnv object alive. Before each one-rally episode it resets only the
pinned environment's documented mutable episode/match fields. This is a
diagnostic, not production code; the field reset must be reviewed against the
pinned CoachAI+ commit before being used by the worker.

CoachAI+ source is never modified. CSV output is written to a temporary path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
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
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_controlled_reset_probe.json"
PROBE_HELPER = PROJECT_ROOT / "scripts" / "setup" / "probe_coachai_runtime.py"


def load_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("b2_runtime_probe", PROBE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runtime probe helpers from {PROBE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def construct_env(env_class: Any, agent_a: Any, agent_b: Any,
                  player: str, opponent: str, filepath: Path) -> Any:
    return env_class(
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


def reset_episode_context(env: Any, episode_match_id: int) -> dict[str, Any]:
    """Reset fields required for an isolated 0-0, B-serving rally.

    Keep `output`, `badminton_constraint`, `baseServe`, and both agent objects:
    they are shared run-level resources, not per-rally history.
    """
    required = (
        "score_a", "score_b", "pre_score_a", "pre_score_b", "serving_player",
        "rally", "match", "round", "state", "info_dict", "output",
        "badminton_constraint", "baseServe", "agent_a", "agent_b",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise RuntimeError(
            "Pinned CoachAI+ instance is missing expected reset fields: "
            + ", ".join(missing)
        )

    before = {
        "score_a": int(env.score_a),
        "score_b": int(env.score_b),
        "pre_score_a": int(env.pre_score_a),
        "pre_score_b": int(env.pre_score_b),
        "serving_player": str(env.serving_player),
        "rally": int(env.rally),
        "match": int(env.match),
        "round": int(env.round),
        "output_rows": int(len(env.output)),
    }

    # These fields are initialized in BadmintonEnv.__init__ and/or mutated by
    # reset()/step()/Record(). Reset them explicitly to make each Gym episode
    # represent a fresh 0-0 rally with B serving.
    env.score_a = 0
    env.score_b = 0
    env.pre_score_a = 0
    env.pre_score_b = 0
    env.serving_player = "B"
    env.rally = 0              # reset() increments this to 1
    env.match = int(episode_match_id)  # unique CSV run identifier
    env.round = 1
    env.state = None
    env.info_dict = {}

    return before


def _rounded(value: Any, digits: int = 9) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_rounded(item, digits) for item in list(value)]
    if isinstance(value, dict):
        return {str(k): _rounded(v, digits) for k, v in value.items()}
    return repr(value)


def play_one_episode(env: Any, agent_a: Any, helpers: Any, torch: Any,
                     seed: int, episode_match_id: int, max_decisions: int) -> dict[str, Any]:
    helpers.seed_everything(seed, torch)
    reset_before = reset_episode_context(env, episode_match_id)
    score_before = {"A": int(env.score_a), "B": int(env.score_b)}
    rows_before = int(len(env.output))
    constraint_id_before = id(env.badminton_constraint)
    agent_a_id_before = id(env.agent_a)
    agent_b_id_before = id(env.agent_b)

    started = time.perf_counter()
    reset_result = env.reset()
    reset_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
        raise RuntimeError(
            "Expected native four-value reset; got "
            f"{type(reset_result).__name__} length="
            f"{len(reset_result) if isinstance(reset_result, (tuple, list)) else 'n/a'}"
        )

    state, info, done, launch = reset_result
    reset_snapshot = {
        "return_length": len(reset_result),
        "done": bool(done),
        "launch": bool(launch),
        "state": helpers.summarize_state(state),
        "info": helpers.summarize_info(info),
        "score": {"A": int(env.score_a), "B": int(env.score_b)},
        "rally": int(env.rally),
        "match": int(env.match),
        "reset_ms": round(reset_ms, 3),
    }

    # B's serve is executed inside reset(). Record the latest native action
    # from its returned info history to check the service path is stable.
    b_serve_action = None
    if isinstance(info, dict) and isinstance(info.get("action"), (tuple, list)) and info["action"]:
        b_serve_action = info["action"][-1]

    decisions = []
    native_rewards: list[float] = []
    while not bool(done) and len(decisions) < max_decisions:
        action_started = time.perf_counter()
        action_a = agent_a.action(state, info, launch)
        action_ms = (time.perf_counter() - action_started) * 1000.0
        step_result = env.step(action_a, launch)
        if not isinstance(step_result, (tuple, list)) or len(step_result) != 5:
            raise RuntimeError(
                "Expected native five-value A-controlled step; got "
                f"{type(step_result).__name__} length="
                f"{len(step_result) if isinstance(step_result, (tuple, list)) else 'n/a'}"
            )
        next_state, native_reward, next_info, done, next_launch = step_result
        native_rewards.append(float(native_reward))
        decisions.append({
            "action_a": helpers.summarize_action(action_a),
            "action_a_inference_ms": round(action_ms, 3),
            "native_reward": float(native_reward),
            "done": bool(done),
            "launch": bool(next_launch),
            "next_state": helpers.summarize_state(next_state),
            "history_length": len(next_info.get("state", [])) if isinstance(next_info, dict) else None,
        })
        state, info, launch = next_state, next_info, next_launch

    if not bool(done):
        raise RuntimeError(f"Episode did not terminate within {max_decisions} A decisions")

    score_after = {"A": int(env.score_a), "B": int(env.score_b)}
    delta = {side: score_after[side] - score_before[side] for side in ("A", "B")}
    if delta == {"A": 1, "B": 0}:
        winner_side = "A"
    elif delta == {"A": 0, "B": 1}:
        winner_side = "B"
    else:
        winner_side = None

    new_rows = env.output.iloc[rows_before:]
    final_record = new_rows.iloc[-1].to_dict() if len(new_rows) else None
    if final_record is not None:
        final_record = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in final_record.items()
        }
    record_winner = final_record.get("getpoint_player") if final_record else None
    expected_name = env.agent_a_name if winner_side == "A" else env.agent_b_name if winner_side == "B" else None

    result = {
        "seed": seed,
        "reset_fields_before": reset_before,
        "reset": reset_snapshot,
        "b_serve_action": helpers.summarize_action(b_serve_action),
        "decision_count": len(decisions),
        "decisions": decisions,
        "native_rewards": native_rewards,
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": delta,
        "winner_side_from_score_delta": winner_side,
        "winner_name_from_final_record": record_winner,
        "winner_record_matches_score_delta": (
            winner_side in {"A", "B"} and record_winner == expected_name
        ),
        "output_rows_before": rows_before,
        "output_rows_after": int(len(env.output)),
        "output_rows_added": int(len(env.output)) - rows_before,
        "resources_reused": {
            "constraint_object_same": id(env.badminton_constraint) == constraint_id_before,
            "agent_a_same": id(env.agent_a) == agent_a_id_before,
            "agent_b_same": id(env.agent_b) == agent_b_id_before,
        },
    }
    result["determinism_signature"] = _rounded({
        "reset_state": state if not decisions else reset_result[0],
        "b_serve_action": b_serve_action,
        "decisions": [
            {"action": item["action_a"], "next_state": item["next_state"], "done": item["done"]}
            for item in decisions
        ],
        "score_delta": delta,
        "winner": winner_side,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--player", default="CHOU Tien Chen")
    parser.add_argument("--opponent", default="Kento MOMOTA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
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
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2 for a reuse probe")

    report: dict[str, Any] = {
        "probe_version": "b2_controlled_reset_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "coachai_root": str(root),
        "python_version": platform.python_version(),
        "players": {"A": args.player, "B": args.opponent},
        "base_seed": args.seed,
        "episode_count": args.episodes,
        "reset_strategy": "persistent_env_with_explicit_episode_field_reset",
        "reset_fields": [
            "score_a=0", "score_b=0", "pre_score_a=0", "pre_score_b=0",
            "serving_player='B'", "rally=0", "match=episode_id", "round=1",
            "state=None", "info_dict={}",
        ],
        "preserved_resources": ["agent_a", "agent_b", "badminton_constraint", "baseServe", "output"],
    }

    try:
        helpers = load_helpers()
        report["coachai_checkout"] = helpers.verify_checkout(root)
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
        helpers.install_cpu_torch_load_fallback(torch, device)
        agent_a, agent_b, target_players, hyperparameters = helpers.load_rallynet_agents(
            root, torch, args.player, args.opponent, device
        )

        with tempfile.TemporaryDirectory(prefix="coachai_b2_controlled_reset_") as temp:
            output_path = Path(temp) / "persistent_env.csv"
            constructor_started = time.perf_counter()
            env = construct_env(
                BadmintonEnv, agent_a, agent_b, args.player, args.opponent, output_path
            )
            constructor_ms = (time.perf_counter() - constructor_started) * 1000.0
            resource_ids = {
                "env": id(env),
                "agent_a": id(agent_a),
                "agent_b": id(agent_b),
                "constraint": id(env.badminton_constraint),
                "base_serve": id(env.baseServe),
            }
            try:
                episodes = [
                    play_one_episode(
                        env, agent_a, helpers, torch, args.seed, index + 1, args.max_decisions
                    )
                    for index in range(args.episodes)
                ]
            finally:
                env.close()

            output_exists = output_path.is_file()
            output_size = output_path.stat().st_size if output_exists else None
            output_rows_readback = None
            if output_exists:
                import pandas as pd
                output_rows_readback = int(len(pd.read_csv(output_path)))

        signatures = [item["determinism_signature"] for item in episodes]
        all_resource_reuse = all(
            all(item["resources_reused"].values()) for item in episodes
        )
        all_neutral_starts = all(
            item["score_before"] == {"A": 0, "B": 0}
            and item["reset"]["score"] == {"A": 0, "B": 0}
            and item["reset"]["launch"] is False
            and item["reset"]["rally"] == 1
            for item in episodes
        )
        all_winners_match = all(
            item["winner_side_from_score_delta"] in {"A", "B"}
            and item["winner_record_matches_score_delta"]
            for item in episodes
        )
        report["runtime"] = {
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "target_player_count": len(target_players),
            "agent_a_class": type(agent_a).__name__,
            "agent_b_class": type(agent_b).__name__,
            "agent_a_training_mode": bool(agent_a.training),
            "agent_b_training_mode": bool(agent_b.training),
            "hyperparameter_keys": sorted(hyperparameters.keys()),
        }
        report["persistent_environment"] = {
            "constructor_ms": round(constructor_ms, 3),
            "resource_ids": resource_ids,
            "episodes": episodes,
            "same_seed_signatures_equal": all(sig == signatures[0] for sig in signatures[1:]),
            "all_neutral_starts": all_neutral_starts,
            "all_winner_records_match_score_delta": all_winners_match,
            "all_run_level_resources_reused": all_resource_reuse,
            "output_file_exists_after_close": output_exists,
            "output_file_bytes": output_size,
            "output_file_rows_readback": output_rows_readback,
            "expected_output_rows": sum(item["output_rows_added"] for item in episodes),
        }
        report["status"] = "PASS" if (
            all_neutral_starts and all_winners_match and all_resource_reuse
            and output_exists and output_rows_readback == sum(item["output_rows_added"] for item in episodes)
        ) else "FAILED"
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
        "episodes": report.get("episode_count"),
        "all_neutral_starts": report.get("persistent_environment", {}).get("all_neutral_starts"),
        "same_seed_signatures_equal": report.get("persistent_environment", {}).get("same_seed_signatures_equal"),
        "all_winner_records_match_score_delta": report.get("persistent_environment", {}).get("all_winner_records_match_score_delta"),
        "all_run_level_resources_reused": report.get("persistent_environment", {}).get("all_run_level_resources_reused"),
        "output_file_rows_readback": report.get("persistent_environment", {}).get("output_file_rows_readback"),
        "error": report.get("error"),
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
