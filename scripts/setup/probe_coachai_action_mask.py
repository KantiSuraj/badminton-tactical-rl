#!/usr/bin/env python3
"""Probe the pinned CoachAI+ native constraints against the proposed 1,350-action B2 grid.

Run from the project root:

    .venv-coachai/bin/python scripts/setup/probe_coachai_action_mask.py

This is a diagnostic, not production code. It does not call validate_action() on illegal
candidates because that method repairs actions (and can use random fallbacks). Instead, it
reconstructs the three native legality checks without mutation, then passes every candidate
marked legal to validate_action() and requires the native validator to leave it unchanged.

The canonical-to-CoachAI mapping below is PROPOSED from the pinned model/native ID conversion.
HALF_SMASH, BLOCK and CUT are deliberately unsupported until the user approves the taxonomy
resolution. No CoachAI+ source files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / "scripts" / "setup" / "probe_coachai_runtime.py"
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_action_mask_probe.json"
EXPECTED_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"

# IDs are canonical ShuttleSet family IDs -> RallyNet output IDs -> CoachAI native IDs.
# The middle/native IDs are grounded in Constraint.py's t2c table and Environment.py's
# type_mapping. The semantic family correspondence is a proposal pending user approval.
PROPOSED_FAMILY_MAP = (
    {"family": "CLEAR", "family_id": 0, "model_shot_id": 3, "native_shot_id": 2,
     "native_name": "Clear", "status": "supported_proposed"},
    {"family": "DROP", "family_id": 1, "model_shot_id": 5, "native_shot_id": 9,
     "native_name": "Drop", "status": "supported_proposed"},
    {"family": "SMASH", "family_id": 2, "model_shot_id": 4, "native_shot_id": 4,
     "native_name": "Smash", "status": "supported_proposed"},
    {"family": "HALF_SMASH", "family_id": 3, "model_shot_id": None, "native_shot_id": None,
     "native_name": None, "status": "unsupported_proposed"},
    {"family": "DRIVE", "family_id": 4, "model_shot_id": 7, "native_shot_id": 6,
     "native_name": "Drive", "status": "supported_proposed"},
    {"family": "PUSH", "family_id": 5, "model_shot_id": 9, "native_shot_id": 3,
     "native_name": "Push Shot", "status": "supported_proposed"},
    {"family": "LIFT", "family_id": 6, "model_shot_id": 6, "native_shot_id": 8,
     "native_name": "Lob", "status": "supported_proposed"},
    {"family": "NET_SHOT", "family_id": 7, "model_shot_id": 8, "native_shot_id": 7,
     "native_name": "Net Shot", "status": "supported_proposed"},
    {"family": "BLOCK", "family_id": 8, "model_shot_id": None, "native_shot_id": None,
     "native_name": None, "status": "unsupported_proposed"},
    {"family": "CUT", "family_id": 9, "model_shot_id": None, "native_shot_id": None,
     "native_name": None, "status": "unsupported_proposed"},
)

N_FAMILIES, N_LANDING, N_RECOVERY = 10, 15, 9
N_ACTIONS = N_FAMILIES * N_LANDING * N_RECOVERY
HALF_WIDTH_M = 2.59
HALF_LENGTH_M = 6.70
NATIVE_HALF_WIDTH = 127.5
NATIVE_HALF_LENGTH = 330.0


def load_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("b2_runtime_probe", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runtime probe helpers from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def landing_centre(cell_id: int) -> tuple[float, float]:
    row, col = divmod(cell_id, 3)
    x = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
    y = (row + 0.5) * (HALF_LENGTH_M / 5.0)
    return float(x), float(y)


def recovery_centre(cell_id: int) -> tuple[float, float]:
    row, col = divmod(cell_id, 3)
    x = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
    y = -(row + 0.5) * (HALF_LENGTH_M / 3.0)
    return float(x), float(y)


def metres_to_native(xy_m: tuple[float, float]) -> tuple[float, float]:
    """Map the B2 centred-metre singles frame to CoachAI+'s action-coordinate frame.

    Scale is derived from the documented playable singles half-width/half-length and the
    native action bounds (x +/-127.5; opponent target y 0..330). It is intentionally explicit.
    """
    sx = NATIVE_HALF_WIDTH / HALF_WIDTH_M
    sy = NATIVE_HALF_LENGTH / HALF_LENGTH_M
    return float(xy_m[0] * sx), float(xy_m[1] * sy)


def y_movement_bin(state: Any) -> int:
    player_y = float(state[1][1])
    shuttle_y = float(state[3][1])
    movement = (shuttle_y - player_y) / 240.0
    if movement < -0.5:
        return 0
    if movement < -0.25:
        return 1
    if movement < 0.0:
        return 2
    if movement < 0.25:
        return 3
    if movement < 0.5:
        return 4
    return 5


def native_contact_reachable(env: Any, state: Any) -> tuple[bool, float, str | None]:
    hit_x, hit_y = float(state[3][0]), float(state[3][1])
    player_x, player_y = float(state[1][0]), float(state[1][1])
    dx = (hit_x - player_x) / 177.5
    dy = (hit_y - player_y) / 240.0
    distance = math.hypot(dx, dy)
    # Probe the native reach/miss condition with an in-bounds target. The candidate shot ID
    # is immaterial to check_out_violation(); only hit point and landing bounds are used.
    probe_action = (3, (hit_x, hit_y), (0.0, 100.0), (0.0, -100.0), tuple([1.0] + [0.0] * 9))
    violation = env.check_out_violation(probe_action, state, False)
    return violation is None, distance, violation


def encode_action(family_id: int, landing_id: int, recovery_id: int) -> int:
    return (family_id * N_LANDING + landing_id) * N_RECOVERY + recovery_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--player", default="CHOU Tien Chen")
    parser.add_argument("--opponent", default="Kento MOMOTA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()

    report: dict[str, Any] = {
        "probe_version": "b2_coachai_action_mask_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "coachai_root": str(root),
        "expected_commit": EXPECTED_COMMIT,
        "seed": args.seed,
        "players": {"A": args.player, "B": args.opponent},
        "action_space": {"shape": [N_ACTIONS], "dtype": "bool", "canonical_factors": [10, 15, 9]},
        "mapping_status": "PROPOSED_ONLY_NOT_FROZEN",
        "family_mapping": list(PROPOSED_FAMILY_MAP),
        "coordinate_conversion": {
            "source_frame": "centred metres; net y=0; A own half negative y",
            "target_frame": "CoachAI+ A-relative action coordinates",
            "native_x_units_per_metre": NATIVE_HALF_WIDTH / HALF_WIDTH_M,
            "native_y_units_per_metre": NATIVE_HALF_LENGTH / HALF_LENGTH_M,
            "native_singles_bounds": {"x": [-NATIVE_HALF_WIDTH, NATIVE_HALF_WIDTH],
                                      "opponent_landing_y": [0.0, NATIVE_HALF_LENGTH]},
        },
    }

    env = None
    old_cwd = Path.cwd()
    try:
        helpers = load_helpers()
        report["coachai_checkout"] = helpers.verify_checkout(root)
        sys.path[:0] = [str(root), str(root / "BadmintonEnv"), str(PROJECT_ROOT / "src")]
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
        helpers.seed_everything(args.seed, torch)
        agent_a, agent_b, target_players, hyperparameters = helpers.load_rallynet_agents(
            root, torch, args.player, args.opponent, device
        )

        with tempfile.TemporaryDirectory(prefix="coachai_b2_action_mask_") as temp_dir:
            output_path = Path(temp_dir) / "native_output.csv"
            env = BadmintonEnv(
                agent_a, agent_b, args.player, args.opponent, 1,
                is_match=True, is_constraint=True, is_have_serve_state=False,
                filepath=str(output_path),
            )
            env.serving_player = "B"
            helpers.seed_everything(args.seed, torch)
            started = time.perf_counter()
            reset_result = env.reset()
            reset_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
                raise RuntimeError(f"Expected 4-value B-serving reset, got {type(reset_result).__name__}")
            state, info, done, launch = reset_result
            if bool(done) or bool(launch):
                raise RuntimeError(f"Expected active A receive state after B serve; done={done}, launch={launch}")

            constraint = env.badminton_constraint
            hit_xy = (float(state[3][0]), float(state[3][1]))
            hit_zone = int(constraint.coordinate_to_block(hit_xy, "self"))
            movement_bin = y_movement_bin(state)
            contact_ok, contact_distance, contact_violation = native_contact_reachable(env, state)

            landing_points = [metres_to_native(landing_centre(i)) for i in range(N_LANDING)]
            recovery_points = [metres_to_native(recovery_centre(i)) for i in range(N_RECOVERY)]
            landing_zones = [int(constraint.coordinate_to_block(p, "opp")) for p in landing_points]
            recovery_zones = [int(constraint.coordinate_to_block(p, "self")) for p in recovery_points]
            if not all(1 <= zone <= 24 for zone in landing_zones + recovery_zones):
                raise RuntimeError(
                    "At least one proposed grid centre is outside CoachAI's 24-zone court: "
                    f"landing={landing_zones}, recovery={recovery_zones}"
                )

            report["runtime"] = {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "reset_ms": round(reset_ms, 3),
                "reset_return_length": len(reset_result),
                "launch": bool(launch),
                "done": bool(done),
                "info_keys": sorted(str(k) for k in info),
                "state": helpers.summarize_state(state),
                "contact": {
                    "hitting_position_native": list(hit_xy),
                    "player_position_native": [float(state[1][0]), float(state[1][1])],
                    "normalized_reach_distance": contact_distance,
                    "reachable": contact_ok,
                    "native_violation": contact_violation,
                    "hit_zone": hit_zone,
                    "y_movement_bin": movement_bin,
                    "previous_native_shot_id": int(state[0]),
                },
                "landing_zones": landing_zones,
                "recovery_zones": recovery_zones,
            }

            if not (1 <= hit_zone <= 24):
                raise RuntimeError(f"Current shuttle contact maps outside CoachAI zones: {hit_zone}")
            if not contact_ok:
                report["status"] = "CONTACT_UNREACHABLE_AT_PROBE_STATE"
                report["diagnostic"] = (
                    "This B serve produced an unreturnable state for A. No legal-action mask was "
                    "computed; the production backend must resolve this as a forced native miss "
                    "before exposing a policy decision."
                )
            else:
                mask = np.zeros((N_FAMILIES, N_LANDING, N_RECOVERY), dtype=np.bool_)
                family_stats: dict[str, Any] = {}
                dist1 = constraint.action_distribution_1.get(int(state[0]))
                if dist1 is None:
                    raise RuntimeError(
                        f"No native shot distribution for previous native shot ID {int(state[0])}"
                    )
                valid_type_vector = np.asarray(dist1[hit_zone - 1][movement_bin], dtype=np.int8)
                if valid_type_vector.shape != (10,):
                    raise RuntimeError(f"Unexpected native shot mask shape: {valid_type_vector.shape}")

                first_valid_action: tuple[int, int, int, tuple[Any, ...]] | None = None
                for entry in PROPOSED_FAMILY_MAP:
                    family = str(entry["family"])
                    family_id = int(entry["family_id"])
                    model_id = entry["model_shot_id"]
                    native_id = entry["native_shot_id"]
                    if model_id is None or native_id is None:
                        family_stats[family] = {
                            "status": "unsupported_proposed",
                            "valid_actions": 0,
                            "reason": "No exact native class in the pinned CoachAI+ action taxonomy",
                        }
                        continue

                    native_id = int(native_id)
                    model_id = int(model_id)
                    type_ok = bool(valid_type_vector[native_id - 1])
                    landing_distribution = constraint.action_distribution_2.get(native_id, [1] * 24)
                    family_valid = 0
                    type_valid_by_landing = 0
                    recovery_valid_count = 0

                    for landing_id, landing_xy in enumerate(landing_points):
                        landing_zone = landing_zones[landing_id]
                        landing_ok = bool(landing_distribution[landing_zone - 1])
                        if type_ok and landing_ok:
                            type_valid_by_landing += 1
                        if not (type_ok and landing_ok):
                            continue

                        for recovery_id, recovery_xy in enumerate(recovery_points):
                            corrected = constraint.correct_position(
                                hit_xy[0], hit_xy[1], recovery_xy[0], recovery_xy[1]
                            )
                            recovery_ok = bool(np.allclose(
                                np.asarray(corrected, dtype=np.float64),
                                np.asarray(recovery_xy, dtype=np.float64),
                                rtol=0.0, atol=1e-8,
                            ))
                            if not recovery_ok:
                                continue
                            recovery_valid_count += 1

                            probability = [0.0] * 10
                            probability[model_id - 1] = 1.0
                            candidate = (
                                model_id,
                                hit_xy,
                                landing_xy,
                                recovery_xy,
                                tuple(probability),
                            )
                            violation = env.check_out_violation(candidate, state, False)
                            if violation is not None:
                                continue

                            mask[family_id, landing_id, recovery_id] = True
                            family_valid += 1
                            if first_valid_action is None:
                                first_valid_action = (family_id, landing_id, recovery_id, candidate)

                    family_stats[family] = {
                        "status": "supported_proposed",
                        "model_shot_id": model_id,
                        "native_shot_id": native_id,
                        "native_name": entry["native_name"],
                        "native_shot_constraint_accepts": type_ok,
                        "landing_cells_passing_shot_and_landing_constraints": type_valid_by_landing,
                        "recovery_cells_passing_movement_constraint": recovery_valid_count,
                        "valid_joint_actions": family_valid,
                    }

                flat_mask = mask.reshape(-1)
                valid_ids = np.flatnonzero(flat_mask).astype(int).tolist()
                mask_bytes = flat_mask.astype(np.uint8).tobytes()
                report["mask"] = {
                    "shape": list(mask.shape),
                    "flat_shape": [N_ACTIONS],
                    "dtype": str(flat_mask.dtype),
                    "valid_count": int(flat_mask.sum()),
                    "invalid_count": int(N_ACTIONS - flat_mask.sum()),
                    "valid_action_ids": valid_ids,
                    "sha256": hashlib.sha256(mask_bytes).hexdigest(),
                    "valid_counts_by_family": {
                        name: int(mask[index].sum())
                        for index, name in enumerate((
                            "CLEAR", "DROP", "SMASH", "HALF_SMASH", "DRIVE",
                            "PUSH", "LIFT", "NET_SHOT", "BLOCK", "CUT",
                        ))
                    },
                }
                report["family_constraint_results"] = family_stats

                if not valid_ids or first_valid_action is None:
                    report["status"] = "NO_VALID_JOINT_ACTION"
                    report["diagnostic"] = (
                        "The proposed direct mappings produced an empty joint mask at this state. "
                        "Do not enable a fallback action; inspect the source-state and constraints."
                    )
                else:
                    # Validate every action marked legal. Illegal candidates are never passed to
                    # validate_action(), since that method mutates/corrects them by design.
                    validator_checks = 0
                    validator_failures: list[dict[str, Any]] = []
                    for family_id, landing_id, recovery_id in np.argwhere(mask):
                        entry = PROPOSED_FAMILY_MAP[int(family_id)]
                        model_id = int(entry["model_shot_id"])
                        native_id = int(entry["native_shot_id"])
                        probability = [0.0] * 10
                        probability[model_id - 1] = 1.0
                        candidate = (
                            model_id,
                            hit_xy,
                            landing_points[int(landing_id)],
                            recovery_points[int(recovery_id)],
                            tuple(probability),
                        )
                        checked_action, native_reward, is_constraint = constraint.validate_action(state, candidate)
                        validator_checks += 1
                        valid_unchanged = (
                            int(checked_action[0]) == native_id
                            and tuple(checked_action[1]) == tuple(candidate[1])
                            and tuple(checked_action[2]) == tuple(candidate[2])
                            and tuple(checked_action[3]) == tuple(candidate[3])
                            and float(native_reward) == 0.0
                            and bool(is_constraint) is False
                        )
                        if not valid_unchanged:
                            validator_failures.append({
                                "family_id": int(family_id),
                                "landing_id": int(landing_id),
                                "recovery_id": int(recovery_id),
                                "checked_action": helpers.to_builtin(checked_action),
                                "native_reward": float(native_reward),
                                "is_constraint": bool(is_constraint),
                            })
                    report["validator_cross_check"] = {
                        "valid_candidates_checked": validator_checks,
                        "failures": validator_failures,
                        "all_masked_in_actions_accepted_without_mutation": not validator_failures,
                        "probability_vector_rule": "one-hot at RallyNet model_shot_id - 1; legality mask prevents repair path",
                    }
                    if validator_failures:
                        report["status"] = "MASK_VALIDATOR_MISMATCH"
                    else:
                        family_id, landing_id, recovery_id, candidate = first_valid_action
                        started = time.perf_counter()
                        step_result = env.step(candidate, False)
                        step_ms = (time.perf_counter() - started) * 1000.0
                        if not isinstance(step_result, (tuple, list)) or len(step_result) != 5:
                            raise RuntimeError(
                                f"Expected five-value native step, received: {type(step_result).__name__}"
                            )
                        next_state, native_reward, next_info, next_done, next_launch = step_result
                        report["one_native_step"] = {
                            "action_id": encode_action(int(family_id), int(landing_id), int(recovery_id)),
                            "canonical_action": {
                                "family": PROPOSED_FAMILY_MAP[int(family_id)]["family"],
                                "landing_cell": int(landing_id),
                                "recovery_cell": int(recovery_id),
                            },
                            "native_action_before_step": helpers.summarize_action(candidate),
                            "step_return_length": len(step_result),
                            "native_reward": float(native_reward),
                            "done": bool(next_done),
                            "launch": bool(next_launch),
                            "next_state": helpers.summarize_state(next_state),
                            "next_info_keys": sorted(str(k) for k in next_info),
                            "step_ms": round(step_ms, 3),
                        }
                        report["status"] = "PASS"

            env.close()
            env = None
            report["output_file"] = {
                "exists_after_close": output_path.is_file(),
                "bytes": output_path.stat().st_size if output_path.is_file() else None,
            }

    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        os.chdir(old_cwd)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": report.get("status"),
        "report": str(report_path),
        "contact_reachable": report.get("runtime", {}).get("contact", {}).get("reachable"),
        "valid_action_count": report.get("mask", {}).get("valid_count"),
        "validator_checks": report.get("validator_cross_check", {}).get("valid_candidates_checked"),
        "validator_failures": len(report.get("validator_cross_check", {}).get("failures", [])),
        "error": report.get("error"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
