"""Probe the pinned CoachAI+ runtime API without modifying the checkout.

Run from the badminton-tactical-rl project root with the isolated worker interpreter:

    .venv-coachai/bin/python scripts/setup/probe_coachai_runtime.py

The probe creates one rally with A serving and one with B serving (unless narrowed with
--serve-modes), records native reset/step tuple shapes, state/action/info summaries,
constraint repairs, score deltas and the final recorded point winner, then writes JSON
under reports/. CoachAI+ output CSVs are written to a temporary directory outside the
third-party checkout and removed automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COACHAI_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_runtime_probe.json"

# These are the active files needed by the pinned runner / RallyNet. Hashes are captured
# in the report. The five model/pickle hashes below were independently recorded in the
# project's third_party_manifest.yaml; All_dataset.csv has no upstream expected hash.
ACTIVE_FILES = (
    "input_data/All_dataset.csv",
    "input_data/target_players_ids.pkl",
    "BadmintonEnv/Agent/hyperparameters.pkl",
    "BadmintonEnv/Agent/RallyNet_weight.trc",
    "BadmintonEnv/Agent/INV_weight.pth",
    "BadmintonEnv/Agent/BC4Serve_weight.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_checkout(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        # A linked worktree may have a .git file rather than a directory.
        if not (root / ".git").is_file():
            raise FileNotFoundError(f"CoachAI+ Git checkout not found: {root}")

    head = git_output(root, "rev-parse", "HEAD")
    pinned = git_output(root, "rev-parse", f"{EXPECTED_COACHAI_COMMIT}^{{commit}}")
    status = git_output(root, "status", "--short")
    if head != EXPECTED_COACHAI_COMMIT or pinned != EXPECTED_COACHAI_COMMIT:
        raise RuntimeError(
            "CoachAI+ commit mismatch: "
            f"HEAD={head}, expected={EXPECTED_COACHAI_COMMIT}, resolved_pin={pinned}"
        )
    if status:
        raise RuntimeError(
            "CoachAI+ working tree is not clean. Preserve generated files, move them "
            "outside the checkout, then rerun. git status --short:\n" + status
        )

    files: dict[str, Any] = {}
    for relative in ACTIVE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required CoachAI+ file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    return {
        "root": str(root),
        "head": head,
        "expected_commit": EXPECTED_COACHAI_COMMIT,
        "working_tree_clean": True,
        "files": files,
    }


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Do not force deterministic-algorithm mode: it can reject valid upstream ops.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_builtin(value: Any) -> Any:
    """Convert common NumPy/PyTorch values into JSON-safe summaries."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def summarize_state(state: Any) -> dict[str, Any]:
    """CoachAI+ state is expected to be (shot_type, A_pos, B_pos, shuttle_pos)."""
    if not isinstance(state, (tuple, list)):
        return {"python_type": type(state).__name__, "value": to_builtin(state)}
    names = ("previous_shot_type", "player_position", "opponent_position", "shuttle_position")
    fields = []
    for index, value in enumerate(state):
        fields.append(
            {
                "index": index,
                "name": names[index] if index < len(names) else f"field_{index}",
                "python_type": type(value).__name__,
                "value": to_builtin(value),
            }
        )
    return {"container_type": type(state).__name__, "length": len(state), "fields": fields}


def summarize_action(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    if not isinstance(action, (tuple, list)):
        return {"python_type": type(action).__name__, "value": to_builtin(action)}
    names = ("shot_type", "hitting_position", "landing_position", "recovery_position", "shot_probability")
    summary: dict[str, Any] = {"length": len(action), "fields": {}}
    for index, value in enumerate(action):
        name = names[index] if index < len(names) else f"field_{index}"
        converted = to_builtin(value)
        field: dict[str, Any] = {"python_type": type(value).__name__}
        if index == 4:
            arr = np.asarray(converted, dtype=np.float64).reshape(-1)
            field.update(
                {
                    "length": int(arr.size),
                    "sum": finite_number(arr.sum()),
                    "min": finite_number(arr.min()) if arr.size else None,
                    "max": finite_number(arr.max()) if arr.size else None,
                    "values": arr.tolist(),
                }
            )
        else:
            field["value"] = converted
        summary["fields"][name] = field
    return summary


def summarize_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"python_type": type(info).__name__, "value": to_builtin(info)}

    summary: dict[str, Any] = {"keys": sorted(str(key) for key in info)}
    for key in ("match", "rally", "round", "score", "env_score", "player", "launch", "constraint"):
        if key in info:
            summary[key] = to_builtin(info[key])
    for key in ("state", "action"):
        if key in info:
            value = info[key]
            summary[f"{key}_history_length"] = len(value) if isinstance(value, (list, tuple)) else None
            if key == "action" and isinstance(value, (list, tuple)) and value:
                summary["latest_action"] = summarize_action(value[-1])
            if key == "state" and isinstance(value, (list, tuple)) and value:
                summary["latest_history_state"] = summarize_state(value[-1])
    return summary


def load_rallynet_agents(root: Path, torch: Any, player_name: str, opponent_name: str, device: Any):
    from BadmintonEnv.Agent.RallyNet import RallyNet

    with (root / "input_data" / "target_players_ids.pkl").open("rb") as stream:
        target_players = pickle.load(stream)
    with (root / "BadmintonEnv" / "Agent" / "hyperparameters.pkl").open("rb") as stream:
        hyperparameters = pickle.load(stream)

    for name in (player_name, opponent_name):
        if name not in target_players:
            raise ValueError(
                f"Player {name!r} is not in input_data/target_players_ids.pkl. "
                f"Known player count: {len(target_players)}"
            )

    def build_agent(name: str):
        player_id = target_players.index(name)
        agent = RallyNet(
            data_size=hyperparameters["data_size"],
            latent_size=hyperparameters["latent_size"],
            context_size=hyperparameters["context_size"],
            hidden_size=hyperparameters["hidden_size"],
            player_ids_len=hyperparameters["player_ids_len"],
            target_players=hyperparameters["target_players"],
            shot_type_len=hyperparameters["shot_type_len"],
            ts=hyperparameters["ts"],
            id=player_id,
            device=device,
        ).to(device)
        checkpoint = root / "BadmintonEnv" / "Agent" / "RallyNet_weight.trc"
        weights = torch.load(checkpoint, map_location=device, weights_only=True)
        agent.load_state_dict(weights)
        agent.eval()
        return agent

    return build_agent(player_name), build_agent(opponent_name), target_players, hyperparameters


def install_cpu_torch_load_fallback(torch: Any, device: Any) -> None:
    """Mirror the established baseline wrapper's CPU-only checkpoint handling."""
    if str(device).startswith("cpu"):
        original_load = torch.load

        def cpu_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return original_load(*args, **kwargs)

        torch.load = cpu_load


def run_one_rally(
    *,
    server: str,
    env_class: Any,
    agent_a: Any,
    agent_b: Any,
    player_name: str,
    opponent_name: str,
    max_decisions: int,
    temporary_output: Path,
) -> dict[str, Any]:
    env = env_class(
        agent_a,
        agent_b,
        player_name,
        opponent_name,
        1,
        is_match=True,
        is_constraint=True,
        is_have_serve_state=False,
        filepath=str(temporary_output),
    )
    env.serving_player = server

    constraint_trace: list[dict[str, Any]] = []
    original_validate = env.badminton_constraint.validate_action

    def traced_validate(state: Any, action: Any):
        before = summarize_action(action)
        result = original_validate(state, action)
        if isinstance(result, (tuple, list)) and len(result) == 3:
            repaired_action, native_reward, constrained = result
            after = summarize_action(repaired_action)
            if len(constraint_trace) < 12:
                before_fields = before.get("fields", {}) if before else {}
                after_fields = after.get("fields", {}) if after else {}
                changed_fields = [
                    name
                    for name in sorted(set(before_fields) | set(after_fields))
                    if before_fields.get(name) != after_fields.get(name)
                ]
                constraint_trace.append(
                    {
                        "before": before,
                        "after": after,
                        "changed_fields": changed_fields,
                        "native_constraint_reward": finite_number(native_reward),
                        "is_constraint": bool(constrained),
                        "action_changed": bool(changed_fields),
                    }
                )
        return result

    # Per-instance instrumentation only; no CoachAI+ source file is modified.
    env.badminton_constraint.validate_action = traced_validate

    score_before = {"A": int(env.score_a), "B": int(env.score_b)}
    start_time = time.perf_counter()
    result: dict[str, Any] = {
        "requested_server": server,
        "score_before": score_before,
        "constructor": {
            "class": f"{env_class.__module__}.{env_class.__name__}",
            "is_match": True,
            "is_constraint": True,
            "is_have_serve_state": False,
            "episodes": 1,
        },
        "steps": [],
        "constraint_validation_samples": constraint_trace,
    }

    try:
        reset_result = env.reset()
        if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
            raise RuntimeError(
                f"Unexpected native reset return: type={type(reset_result).__name__}, "
                f"length={len(reset_result) if isinstance(reset_result, (tuple, list)) else 'n/a'}"
            )

        state, info, done, launch = reset_result
        result["reset"] = {
            "return_length": len(reset_result),
            "tuple_schema": ["state", "info", "done", "launch"],
            "done": bool(done),
            "launch": bool(launch),
            "state": summarize_state(state),
            "info": summarize_info(info),
            "score_after_reset": {"A": int(env.score_a), "B": int(env.score_b)},
        }

        decision_count = 0
        native_rewards: list[float | None] = []
        normal_step_return_lengths: list[int] = []
        while not bool(done) and decision_count < max_decisions:
            action_started = time.perf_counter()
            action = agent_a.action(state, info, launch)
            action_ms = (time.perf_counter() - action_started) * 1000.0
            step_result = env.step(action, launch)
            decision_count += 1

            if not isinstance(step_result, (tuple, list)):
                raise RuntimeError(f"Native step returned {type(step_result).__name__}, not a tuple")
            normal_step_return_lengths.append(len(step_result))
            if len(step_result) == 5:
                next_state, native_reward, next_info, next_done, next_launch = step_result
                tuple_schema = ["state", "native_reward", "info", "done", "launch"]
            elif len(step_result) == 4:
                # The source's B-serve launch branch is normally consumed inside reset().
                # Preserve this schema rather than inventing a missing reward value.
                next_state, next_info, next_done, next_launch = step_result
                native_reward = None
                tuple_schema = ["state", "info", "done", "launch"]
            else:
                raise RuntimeError(f"Native step returned {len(step_result)} values; expected 4 or 5")

            reward_value = finite_number(native_reward) if native_reward is not None else None
            native_rewards.append(reward_value)
            if decision_count <= 12 or bool(next_done):
                result["steps"].append(
                    {
                        "decision_index": decision_count,
                        "step_return_length": len(step_result),
                        "tuple_schema": tuple_schema,
                        "agent_a_action": summarize_action(action),
                        "agent_a_action_inference_ms": round(action_ms, 3),
                        "native_reward": reward_value,
                        "done": bool(next_done),
                        "launch": bool(next_launch),
                        "next_state": summarize_state(next_state),
                        "next_info": summarize_info(next_info),
                        "score": {"A": int(env.score_a), "B": int(env.score_b)},
                    }
                )
            state, info, done, launch = next_state, next_info, next_done, next_launch

        score_after = {"A": int(env.score_a), "B": int(env.score_b)}
        delta_a = score_after["A"] - score_before["A"]
        delta_b = score_after["B"] - score_before["B"]
        if delta_a == 1 and delta_b == 0:
            winner_side = "A"
        elif delta_b == 1 and delta_a == 0:
            winner_side = "B"
        else:
            winner_side = None

        output_records = env.output.to_dict(orient="records") if hasattr(env, "output") else []
        last_record = output_records[-1] if output_records else None
        recorded_winner = last_record.get("getpoint_player") if last_record else None

        result.update(
            {
                "terminated": bool(done),
                "max_decisions_reached": not bool(done) and decision_count >= max_decisions,
                "decision_count": decision_count,
                "score_after": score_after,
                "score_delta": {"A": delta_a, "B": delta_b},
                "winner_side_from_score_delta": winner_side,
                "winner_name_from_final_record": recorded_winner,
                "native_game_winner_check": env.check_winner(),
                "recorded_stroke_rows": len(output_records),
                "final_record": to_builtin(last_record),
                "native_reward_values": native_rewards,
                "normal_step_return_lengths": normal_step_return_lengths,
                "all_normal_steps_return_five_values": all(length == 5 for length in normal_step_return_lengths),
                "info_final": summarize_info(info),
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            }
        )
        if not done:
            result["warning"] = (
                "Rally did not terminate within max_decisions; do not treat this as a completed rally."
            )
        if done and winner_side is None:
            result["warning"] = (
                "Native done=True, but score delta did not identify exactly one rally winner. "
                "Winner extraction must remain unresolved."
            )
        if recorded_winner and winner_side:
            expected_name = player_name if winner_side == "A" else opponent_name
            result["final_record_winner_matches_score_delta"] = recorded_winner == expected_name
        else:
            result["final_record_winner_matches_score_delta"] = None
        return result
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--player", default="CHOU Tien Chen", help="Canonical side A player name")
    parser.add_argument("--opponent", default="Kento MOMOTA", help="Canonical side B player name")
    parser.add_argument("--serve-modes", nargs="+", choices=("A", "B"), default=("A", "B"))
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
        "probe_version": "b2_runtime_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "seed": args.seed,
        "players": {"A": args.player, "B": args.opponent},
        "serve_modes_requested": list(args.serve_modes),
        "max_decisions": args.max_decisions,
        "results": {},
    }

    try:
        checkout = verify_checkout(root)
        report["coachai_checkout"] = checkout
        sys.path[:0] = [str(root), str(root / "BadmintonEnv")]
        os.chdir(root)

        import torch

        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
            device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
        install_cpu_torch_load_fallback(torch, device)
        seed_everything(args.seed, torch)

        from BadmintonEnv.Environment import BadmintonEnv

        agent_a, agent_b, target_players, hyperparameters = load_rallynet_agents(
            root, torch, args.player, args.opponent, device
        )

        report["runtime"] = {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "target_player_count": len(target_players),
            "player_a_id": target_players.index(args.player),
            "player_b_id": target_players.index(args.opponent),
            "hyperparameter_keys": sorted(str(key) for key in hyperparameters),
            "agent_a_class": f"{type(agent_a).__module__}.{type(agent_a).__name__}",
            "agent_b_class": f"{type(agent_b).__module__}.{type(agent_b).__name__}",
            "agent_a_training_mode": bool(agent_a.training),
            "agent_b_training_mode": bool(agent_b.training),
        }

        with tempfile.TemporaryDirectory(prefix="coachai_b2_probe_") as temp_dir:
            temp_root = Path(temp_dir)
            for index, server in enumerate(args.serve_modes):
                # A distinct environment and output file per serve path. The two loaded
                # RallyNet objects are reused; their history is passed through env.info.
                output_path = temp_root / f"probe_server_{server}.csv"
                try:
                    # Reset every RNG before each serve-path probe so A/B runs use the same seed.
                    seed_everything(args.seed, torch)
                    mode_result = run_one_rally(
                        server=server,
                        env_class=BadmintonEnv,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        player_name=args.player,
                        opponent_name=args.opponent,
                        max_decisions=args.max_decisions,
                        temporary_output=output_path,
                    )
                    report["results"][server] = mode_result
                except Exception as exc:  # keep the partial report for diagnosis
                    report["results"][server] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }

        report["status"] = (
            "PASS"
            if report["results"]
            and all(
                result.get("terminated") is True
                and result.get("winner_side_from_score_delta") in {"A", "B"}
                and result.get("all_normal_steps_return_five_values", True)
                for result in report["results"].values()
            )
            else "PARTIAL_OR_FAILED"
        )
    except Exception as exc:
        report["status"] = "FAILED"
        report["fatal_error_type"] = type(exc).__name__
        report["fatal_error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report.get("status", "FAILED"),
        "report": str(report_path),
        "results": {
            mode: {
                "terminated": result.get("terminated"),
                "decision_count": result.get("decision_count"),
                "score_delta": result.get("score_delta"),
                "winner_side": result.get("winner_side_from_score_delta"),
                "error": result.get("error"),
            }
            for mode, result in report.get("results", {}).items()
        },
        "fatal_error": report.get("fatal_error"),
    }, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
