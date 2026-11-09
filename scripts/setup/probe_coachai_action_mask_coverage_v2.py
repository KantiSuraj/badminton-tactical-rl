#!/usr/bin/env python3
"""Measure multi-state coverage and validate the proposed B2 CoachAI+ action mask.

Run from the badminton-tactical-rl project root with the isolated CoachAI+ worker
interpreter:

    .venv-coachai/bin/python scripts/setup/probe_coachai_action_mask_coverage_v2.py

The probe samples naturally occurring A-decision states from complete B-serving
rallies. Both players use the pinned CoachAI+ RallyNet agents to generate the
trajectory; the proposed action mask is evaluated observationally and does not
control the rollout.

For every sampled state, the probe reconstructs the proposed 10 x 15 x 9 mask
from the pinned native constraint tables and checks every mask-in candidate with
CoachAI+'s mutating validate_action() method. Only mask-in candidates are passed
to validate_action(); mask-out candidates are deliberately not passed because
that method repairs invalid actions and can use random fallbacks.

This v2 diagnostic additionally records, for every state, the native shot IDs
allowed by the shot constraint and each proposed family's shot/landing/recovery/joint
counts. This is intended to diagnose empty masks without changing mask semantics.
It does not freeze the family mapping or modify the CoachAI+ checkout. All native CSV
output is written to a temporary directory.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / "scripts" / "setup" / "probe_coachai_runtime.py"
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_action_mask_coverage_v2.json"
EXPECTED_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"

# Canonical ShuttleSet family ID -> RallyNet/model output ID -> CoachAI native ID.
# This remains a proposal. Do not interpret empirical mask coverage as semantic
# approval of these correspondences.
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

FAMILY_NAMES = tuple(str(item["family"]) for item in PROPOSED_FAMILY_MAP)
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
    # check_out_violation's contact/reach branch is independent of the chosen shot
    # category; keep this diagnostic candidate well inside the opponent bounds.
    probe_action = (
        3, (hit_x, hit_y), (0.0, 100.0), (0.0, -100.0),
        tuple([1.0] + [0.0] * 9),
    )
    violation = env.check_out_violation(probe_action, state, False)
    return violation is None, distance, violation


def encode_action(family_id: int, landing_id: int, recovery_id: int) -> int:
    return (family_id * N_LANDING + landing_id) * N_RECOVERY + recovery_id


def stable_state_signature(state: Any) -> str:
    payload = {
        "previous_native_shot_id": int(state[0]),
        "player_position": [round(float(v), 5) for v in state[1]],
        "opponent_position": [round(float(v), 5) for v in state[2]],
        "shuttle_position": [round(float(v), 5) for v in state[3]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_candidate(model_id: int, hit_xy: tuple[float, float],
                    landing_xy: tuple[float, float],
                    recovery_xy: tuple[float, float]) -> tuple[Any, ...]:
    probability = [0.0] * 10
    probability[model_id - 1] = 1.0
    return (model_id, hit_xy, landing_xy, recovery_xy, tuple(probability))


def evaluate_state_mask(
    *, env: Any, state: Any, constraint: Any,
    landing_points: list[tuple[float, float]],
    recovery_points: list[tuple[float, float]],
    landing_zones: list[int], recovery_zones: list[int],
) -> dict[str, Any]:
    """Reconstruct one state mask using the same direct checks as the v1 probe."""
    if not isinstance(state, (tuple, list)) or len(state) != 4:
        raise RuntimeError(f"Expected CoachAI state tuple of length 4, got {type(state).__name__}")

    hit_xy = (float(state[3][0]), float(state[3][1]))
    hit_zone = int(constraint.coordinate_to_block(hit_xy, "self"))
    movement_bin = y_movement_bin(state)
    contact_ok, contact_distance, contact_violation = native_contact_reachable(env, state)
    mask = np.zeros((N_FAMILIES, N_LANDING, N_RECOVERY), dtype=np.bool_)
    family_stats: dict[str, dict[str, Any]] = {}

    if not (1 <= hit_zone <= 24):
        return {
            "mask": mask,
            "hit_xy": hit_xy,
            "hit_zone": hit_zone,
            "movement_bin": movement_bin,
            "contact_reachable": contact_ok,
            "contact_distance": contact_distance,
            "contact_violation": contact_violation,
            "family_stats": family_stats,
            "empty_reason": "contact_zone_outside_native_24_zone_court",
        }

    if not contact_ok:
        for entry in PROPOSED_FAMILY_MAP:
            family_stats[str(entry["family"])] = {
                "status": str(entry["status"]),
                "model_shot_id": entry["model_shot_id"],
                "native_shot_id": entry["native_shot_id"],
                "valid_joint_actions": 0,
                "reason": "contact_unreachable",
            }
        return {
            "mask": mask,
            "hit_xy": hit_xy,
            "hit_zone": hit_zone,
            "movement_bin": movement_bin,
            "contact_reachable": contact_ok,
            "contact_distance": contact_distance,
            "contact_violation": contact_violation,
            "family_stats": family_stats,
            "empty_reason": "contact_unreachable",
        }

    previous_native_id = int(state[0])
    distribution_1 = constraint.action_distribution_1.get(previous_native_id)
    if distribution_1 is None:
        raise RuntimeError(
            f"No native shot distribution for previous native shot ID {previous_native_id}"
        )
    valid_type_vector = np.asarray(
        distribution_1[hit_zone - 1][movement_bin], dtype=np.int8
    )
    if valid_type_vector.shape != (10,):
        raise RuntimeError(f"Unexpected native shot mask shape: {valid_type_vector.shape}")
    allowed_native_shot_ids = [int(index) + 1 for index in np.flatnonzero(valid_type_vector)]

    for entry in PROPOSED_FAMILY_MAP:
        family = str(entry["family"])
        family_id = int(entry["family_id"])
        model_id = entry["model_shot_id"]
        native_id = entry["native_shot_id"]
        if model_id is None or native_id is None:
            family_stats[family] = {
                "status": "unsupported_proposed",
                "model_shot_id": None,
                "native_shot_id": None,
                "valid_joint_actions": 0,
                "reason": "no_exact_native_class_in_pinned_CoachAI_action_taxonomy",
            }
            continue

        model_id = int(model_id)
        native_id = int(native_id)
        type_ok = bool(valid_type_vector[native_id - 1])
        landing_distribution = constraint.action_distribution_2.get(native_id, [1] * 24)
        if len(landing_distribution) != 24:
            raise RuntimeError(
                f"Native landing distribution for shot {native_id} has length "
                f"{len(landing_distribution)}, expected 24"
            )

        family_valid = 0
        legal_landing_ids: list[int] = []
        legal_recovery_ids: set[int] = set()
        for landing_id, landing_xy in enumerate(landing_points):
            landing_zone = landing_zones[landing_id]
            landing_ok = bool(landing_distribution[landing_zone - 1])
            if not (type_ok and landing_ok):
                continue
            legal_landing_ids.append(landing_id)

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

                candidate = build_candidate(model_id, hit_xy, landing_xy, recovery_xy)
                violation = env.check_out_violation(candidate, state, False)
                if violation is not None:
                    continue

                mask[family_id, landing_id, recovery_id] = True
                family_valid += 1
                legal_recovery_ids.add(recovery_id)

        family_stats[family] = {
            "status": "supported_proposed",
            "model_shot_id": model_id,
            "native_shot_id": native_id,
            "native_name": entry["native_name"],
            "native_shot_constraint_accepts": type_ok,
            "landing_cells_passing_shot_and_landing_constraints": len(legal_landing_ids),
            "recovery_cells_passing_movement_constraint": len(legal_recovery_ids),
            "valid_joint_actions": family_valid,
        }

    empty_reason = None if bool(mask.any()) else "no_joint_action_satisfies_all_native_constraints"
    return {
        "mask": mask,
        "hit_xy": hit_xy,
        "hit_zone": hit_zone,
        "movement_bin": movement_bin,
        "contact_reachable": contact_ok,
        "contact_distance": contact_distance,
        "contact_violation": contact_violation,
        "allowed_native_shot_ids": allowed_native_shot_ids,
        "family_stats": family_stats,
        "empty_reason": empty_reason,
    }


def validate_mask_in_actions(
    *, env: Any, state: Any, evaluated: dict[str, Any],
    landing_points: list[tuple[float, float]],
    recovery_points: list[tuple[float, float]],
    helpers: Any,
) -> dict[str, Any]:
    """Cross-check every mask-in candidate; never invoke the repair path for mask-outs."""
    mask = evaluated["mask"]
    failures: list[dict[str, Any]] = []
    checks_by_family: Counter[str] = Counter()
    failures_by_family: Counter[str] = Counter()
    checked = 0

    for family_id, landing_id, recovery_id in np.argwhere(mask):
        family_id = int(family_id)
        landing_id = int(landing_id)
        recovery_id = int(recovery_id)
        entry = PROPOSED_FAMILY_MAP[family_id]
        model_id = int(entry["model_shot_id"])
        native_id = int(entry["native_shot_id"])
        hit_xy = evaluated["hit_xy"]
        candidate = build_candidate(
            model_id, hit_xy, landing_points[landing_id], recovery_points[recovery_id]
        )
        checked_action, native_reward, is_constraint = env.badminton_constraint.validate_action(
            state, candidate
        )
        checked += 1
        family_name = str(entry["family"])
        checks_by_family[family_name] += 1

        same_probability = False
        try:
            same_probability = np.array_equal(
                np.asarray(checked_action[4], dtype=np.float64),
                np.asarray(candidate[4], dtype=np.float64),
            )
        except (IndexError, TypeError, ValueError):
            same_probability = False

        valid_unchanged = (
            isinstance(checked_action, (tuple, list))
            and len(checked_action) == 5
            and int(checked_action[0]) == native_id
            and tuple(checked_action[1]) == tuple(candidate[1])
            and tuple(checked_action[2]) == tuple(candidate[2])
            and tuple(checked_action[3]) == tuple(candidate[3])
            and same_probability
            and float(native_reward) == 0.0
            and bool(is_constraint) is False
        )
        if not valid_unchanged:
            failures_by_family[family_name] += 1
            if len(failures) < 50:
                failures.append({
                    "family": family_name,
                    "family_id": family_id,
                    "landing_id": landing_id,
                    "recovery_id": recovery_id,
                    "candidate": helpers.to_builtin(candidate),
                    "checked_action": helpers.to_builtin(checked_action),
                    "native_reward": float(native_reward),
                    "is_constraint": bool(is_constraint),
                    "probability_unchanged": same_probability,
                })

    return {
        "valid_candidates_present": int(mask.sum()),
        "valid_candidates_checked": checked,
        "failures_count": int(sum(failures_by_family.values())),
        "failures": failures,
        "checks_by_family": dict(checks_by_family),
        "failures_by_family": dict(failures_by_family),
        "all_masked_in_actions_accepted_without_mutation": checked == int(mask.sum())
        and not failures_by_family,
        "masked_out_actions_tested_with_validator": 0,
        "note": (
            "Masked-out actions are not passed to validate_action because it repairs invalid "
            "actions and may use random fallbacks."
        ),
    }


def reset_episode_context(env: Any, episode_match_id: int) -> dict[str, Any]:
    """Reset the fields proven by the controlled-reset probe; retain run-level resources."""
    required = (
        "score_a", "score_b", "pre_score_a", "pre_score_b", "serving_player",
        "rally", "match", "round", "state", "info_dict", "output",
        "badminton_constraint", "baseServe", "agent_a", "agent_b",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise RuntimeError("Missing pinned CoachAI+ episode fields: " + ", ".join(missing))

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
    env.score_a = 0
    env.score_b = 0
    env.pre_score_a = 0
    env.pre_score_b = 0
    env.serving_player = "B"
    env.rally = 0
    env.match = int(episode_match_id)
    env.round = 1
    env.state = None
    env.info_dict = {}
    return before


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--player", default="CHOU Tien Chen")
    parser.add_argument("--opponent", default="Kento MOMOTA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8,
                        help="Number of independent B-serving rallies to sample (default: 8).")
    parser.add_argument("--max-decisions", type=int, default=64,
                        help="Safety cap on A decisions per rally (default: 64).")
    parser.add_argument("--max-states", type=int, default=200,
                        help="Maximum sampled decision states; 0 means unlimited (default: 200).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_decisions < 1:
        raise ValueError("--max-decisions must be at least 1")
    if args.max_states < 0:
        raise ValueError("--max-states must be >= 0 (0 means unlimited)")

    root = args.root.expanduser().resolve()
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()

    report: dict[str, Any] = {
        "probe_version": "b2_coachai_action_mask_coverage_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "coachai_root": str(root),
        "expected_commit": EXPECTED_COMMIT,
        "seed": args.seed,
        "episode_seeds": [args.seed + i for i in range(args.episodes)],
        "players": {"A": args.player, "B": args.opponent},
        "rollout_policy": "pinned_CoachAI_RallyNet_agents_for_both_sides; mask_is_observational_only",
        "reset_strategy": "persistent_native_env_with_explicit_neutral_score_B_serve_reset",
        "mask": {"shape": [N_FAMILIES, N_LANDING, N_RECOVERY],
                 "flat_shape": [N_ACTIONS], "dtype": "bool"},
        "mapping_status": "PROPOSED_ONLY_NOT_FROZEN",
        "family_mapping": list(PROPOSED_FAMILY_MAP),
        "sampling_limits": {
            "episodes_requested": args.episodes,
            "max_decisions_per_episode": args.max_decisions,
            "max_states": args.max_states,
            "validator_policy": "exhaustive_all_masked_in_actions_per_sampled_state",
        },
        "coordinate_conversion": {
            "source_frame": "centred metres; net y=0; A own half negative y",
            "target_frame": "CoachAI+ A-relative action coordinates",
            "native_x_units_per_metre": NATIVE_HALF_WIDTH / HALF_WIDTH_M,
            "native_y_units_per_metre": NATIVE_HALF_LENGTH / HALF_LENGTH_M,
            "native_singles_bounds": {"x": [-NATIVE_HALF_WIDTH, NATIVE_HALF_WIDTH],
                                      "opponent_landing_y": [0.0, NATIVE_HALF_LENGTH]},
        },
        "states": [],
        "episodes": [],
    }

    env = None
    old_cwd = Path.cwd()
    aggregate_family = {
        str(entry["family"]): {
            "status": str(entry["status"]),
            "model_shot_id": entry["model_shot_id"],
            "native_shot_id": entry["native_shot_id"],
            "states_legal": 0,
            "total_valid_joint_actions": 0,
            "landing_cells_seen_legal": set(),
            "recovery_cells_seen_legal": set(),
            "validator_checks": 0,
            "validator_failures": 0,
        }
        for entry in PROPOSED_FAMILY_MAP
    }
    previous_shot_counts: Counter[str] = Counter()
    hit_zone_counts: Counter[str] = Counter()
    movement_bin_counts: Counter[str] = Counter()
    shot_zone_movement_counts: Counter[str] = Counter()
    mask_sizes: list[int] = []
    state_signatures: set[str] = set()
    total_validator_checks = 0
    total_validator_failures = 0
    validator_failure_examples: list[dict[str, Any]] = []
    empty_mask_states = 0
    reachable_contact_states = 0
    unreachable_states = 0
    out_of_zone_states = 0
    sampled_state_count = 0
    complete_episodes = 0
    episode_incomplete = False
    mask_mismatch = False
    stop_after_episode = False

    try:
        helpers = load_helpers()
        report["coachai_checkout"] = helpers.verify_checkout(root)
        sys.path[:0] = [str(root), str(root / "BadmintonEnv")]
        os.chdir(root)

        import torch
        import pandas as pd
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

        report["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "target_player_count": len(target_players),
            "agent_a_eval_mode": not bool(getattr(agent_a, "training", False)),
            "agent_b_eval_mode": not bool(getattr(agent_b, "training", False)),
        }

        landing_points = [metres_to_native(landing_centre(i)) for i in range(N_LANDING)]
        recovery_points = [metres_to_native(recovery_centre(i)) for i in range(N_RECOVERY)]
        # Zone IDs depend only on the grid and the pinned constraint object's geometry.
        with tempfile.TemporaryDirectory(prefix="coachai_b2_mask_coverage_") as temp_dir:
            output_path = Path(temp_dir) / "native_output.csv"
            constructor_started = time.perf_counter()
            env = BadmintonEnv(
                agent_a, agent_b, args.player, args.opponent, 1,
                is_match=True, is_constraint=True, is_have_serve_state=False,
                filepath=str(output_path),
            )
            constructor_ms = (time.perf_counter() - constructor_started) * 1000.0
            constraint = env.badminton_constraint
            landing_zones = [int(constraint.coordinate_to_block(point, "opp"))
                             for point in landing_points]
            recovery_zones = [int(constraint.coordinate_to_block(point, "self"))
                              for point in recovery_points]
            if not all(1 <= zone <= 24 for zone in landing_zones + recovery_zones):
                raise RuntimeError(
                    "Proposed landing/recovery grid contains a point outside CoachAI's 24 zones: "
                    f"landing={landing_zones}, recovery={recovery_zones}"
                )

            report["runtime"].update({
                "env_constructor_ms": round(constructor_ms, 3),
                "landing_zones": landing_zones,
                "recovery_zones": recovery_zones,
                "constraint_object_reused_across_episodes": True,
            })

            try:
                for episode_index in range(args.episodes):
                    episode_seed = args.seed + episode_index
                    helpers.seed_everything(episode_seed, torch)
                    reset_fields_before = reset_episode_context(env, episode_index + 1)
                    score_before = {"A": int(env.score_a), "B": int(env.score_b)}
                    output_rows_before = int(len(env.output))
                    reset_started = time.perf_counter()
                    reset_result = env.reset()
                    reset_ms = (time.perf_counter() - reset_started) * 1000.0
                    if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
                        raise RuntimeError(
                            "Expected CoachAI+ four-value reset; received "
                            f"{type(reset_result).__name__}"
                        )
                    state, info, done, launch = reset_result
                    reset_info = info
                    b_serve_action = None
                    if isinstance(reset_info, dict) and isinstance(reset_info.get("action"), (tuple, list)) and reset_info["action"]:
                        b_serve_action = reset_info["action"][-1]
                    if bool(done):
                        raise RuntimeError(f"Episode {episode_index + 1}: reset returned done=True")

                    episode_decisions = 0
                    episode_sampled_states = 0
                    episode_empty_masks = 0
                    episode_validator_checks = 0
                    episode_validator_failures = 0
                    while not bool(done) and episode_decisions < args.max_decisions:
                        should_sample = args.max_states == 0 or sampled_state_count < args.max_states
                        if should_sample:
                            evaluated = evaluate_state_mask(
                                env=env,
                                state=state,
                                constraint=constraint,
                                landing_points=landing_points,
                                recovery_points=recovery_points,
                                landing_zones=landing_zones,
                                recovery_zones=recovery_zones,
                            )
                            mask = evaluated["mask"]
                            valid_count = int(mask.sum())
                            sampled_state_count += 1
                            episode_sampled_states += 1
                            mask_sizes.append(valid_count)

                            signature = stable_state_signature(state)
                            state_signatures.add(signature)
                            previous_id = int(state[0])
                            hit_zone = int(evaluated["hit_zone"])
                            if evaluated["contact_reachable"]:
                                reachable_contact_states += 1
                            movement_bin = int(evaluated["movement_bin"])
                            previous_shot_counts[str(previous_id)] += 1
                            hit_zone_counts[str(hit_zone)] += 1
                            movement_bin_counts[str(movement_bin)] += 1
                            shot_zone_movement_counts[f"shot={previous_id}|zone={hit_zone}|move={movement_bin}"] += 1

                            family_counts: dict[str, int] = {}
                            for family_id, entry in enumerate(PROPOSED_FAMILY_MAP):
                                family_name = str(entry["family"])
                                count = int(mask[family_id].sum())
                                family_counts[family_name] = count
                                aggregate = aggregate_family[family_name]
                                aggregate["total_valid_joint_actions"] += count
                                if count > 0:
                                    aggregate["states_legal"] += 1
                                    aggregate["landing_cells_seen_legal"].update(
                                        int(value) for value in np.flatnonzero(mask[family_id].any(axis=1))
                                    )
                                    aggregate["recovery_cells_seen_legal"].update(
                                        int(value) for value in np.flatnonzero(mask[family_id].any(axis=0))
                                    )

                            if valid_count == 0:
                                empty_mask_states += 1
                                episode_empty_masks += 1
                                if evaluated["empty_reason"] == "contact_unreachable":
                                    unreachable_states += 1
                                elif evaluated["empty_reason"] == "contact_zone_outside_native_24_zone_court":
                                    out_of_zone_states += 1

                            validation = validate_mask_in_actions(
                                env=env,
                                state=state,
                                evaluated=evaluated,
                                landing_points=landing_points,
                                recovery_points=recovery_points,
                                helpers=helpers,
                            )
                            total_validator_checks += int(validation["valid_candidates_checked"])
                            total_validator_failures += int(validation["failures_count"])
                            episode_validator_checks += int(validation["valid_candidates_checked"])
                            episode_validator_failures += int(validation["failures_count"])
                            if validation["failures_count"]:
                                mask_mismatch = True
                                for failure in validation["failures"]:
                                    if len(validator_failure_examples) >= 100:
                                        break
                                    validator_failure_examples.append({
                                        "sample_index": sampled_state_count,
                                        "episode": episode_index + 1,
                                        "decision_index": episode_decisions + 1,
                                        "state_signature_sha256": signature,
                                        **failure,
                                    })

                            state_record = {
                                "sample_index": sampled_state_count,
                                "episode": episode_index + 1,
                                "episode_seed": episode_seed,
                                "decision_index": episode_decisions + 1,
                                "state_signature_sha256": signature,
                                "previous_native_shot_id": previous_id,
                                "player_position_native": [float(v) for v in state[1]],
                                "opponent_position_native": [float(v) for v in state[2]],
                                "shuttle_position_native": [float(v) for v in state[3]],
                                "hit_zone": hit_zone,
                                "y_movement_bin": movement_bin,
                                "contact_reachable": bool(evaluated["contact_reachable"]),
                                "normalized_reach_distance": float(evaluated["contact_distance"]),
                                "contact_violation": evaluated["contact_violation"],
                                "native_shot_constraint_allowed_ids": evaluated.get("allowed_native_shot_ids"),
                                "family_constraint_breakdown": evaluated["family_stats"],
                                "valid_joint_actions": valid_count,
                                "valid_action_fraction": valid_count / N_ACTIONS,
                                "valid_counts_by_family": family_counts,
                                "empty_mask_reason": evaluated["empty_reason"],
                                "validator_candidates_checked": int(validation["valid_candidates_checked"]),
                                "validator_failures": int(validation["failures_count"]),
                                "mask_sha256": hashlib.sha256(
                                    mask.reshape(-1).astype(np.uint8).tobytes()
                                ).hexdigest(),
                            }
                            report["states"].append(state_record)

                            for family_name, checks in validation["checks_by_family"].items():
                                aggregate_family[family_name]["validator_checks"] += int(checks)
                            for family_name, failures in validation["failures_by_family"].items():
                                aggregate_family[family_name]["validator_failures"] += int(failures)

                        # Keep the state distribution independent of the proposed mask:
                        # the baseline A RallyNet action, not a mask-selected action,
                        # drives the next transition.
                        action_a = agent_a.action(state, info, launch)
                        step_result = env.step(action_a, launch)
                        if not isinstance(step_result, (tuple, list)) or len(step_result) != 5:
                            raise RuntimeError(
                                "Expected CoachAI+ five-value step; received "
                                f"{type(step_result).__name__}"
                            )
                        state, native_reward, info, done, launch = step_result
                        episode_decisions += 1

                    episode_completed = bool(done)
                    if not episode_completed:
                        episode_incomplete = True

                    score_after = {"A": int(env.score_a), "B": int(env.score_b)}
                    score_delta = {
                        side: score_after[side] - score_before[side] for side in ("A", "B")
                    }
                    if score_delta == {"A": 1, "B": 0}:
                        winner_side = "A"
                    elif score_delta == {"A": 0, "B": 1}:
                        winner_side = "B"
                    else:
                        winner_side = None

                    new_rows = env.output.iloc[output_rows_before:]
                    final_record = new_rows.iloc[-1].to_dict() if len(new_rows) else None
                    if final_record is not None:
                        final_record = {
                            key: (value.item() if isinstance(value, np.generic) else value)
                            for key, value in final_record.items()
                        }
                    record_winner = final_record.get("getpoint_player") if final_record else None
                    expected_name = (
                        env.agent_a_name if winner_side == "A"
                        else env.agent_b_name if winner_side == "B" else None
                    )
                    winner_matches = (
                        winner_side in {"A", "B"} and record_winner == expected_name
                    )
                    if episode_completed:
                        complete_episodes += 1

                    report["episodes"].append({
                        "episode": episode_index + 1,
                        "seed": episode_seed,
                        "reset_fields_before": reset_fields_before,
                        "reset_ms": round(reset_ms, 3),
                        "reset_return_length": len(reset_result),
                        "reset_state_signature_sha256": stable_state_signature(reset_result[0]),
                        "b_serve_history_length": (
                            len(reset_info.get("action", [])) if isinstance(reset_info, dict) else None
                        ),
                        "b_serve_action": helpers.summarize_action(b_serve_action),
                        "decisions": episode_decisions,
                        "completed": episode_completed,
                        "sampled_states": episode_sampled_states,
                        "empty_mask_states": episode_empty_masks,
                        "validator_candidates_checked": episode_validator_checks,
                        "validator_failures": episode_validator_failures,
                        "score_before": score_before,
                        "score_after": score_after,
                        "score_delta": score_delta,
                        "winner_side_from_score_delta": winner_side,
                        "winner_name_from_final_record": record_winner,
                        "winner_record_matches_score_delta": winner_matches,
                        "output_rows_added": int(len(env.output)) - output_rows_before,
                    })

                    if not episode_completed:
                        break
                    if args.max_states and sampled_state_count >= args.max_states:
                        stop_after_episode = True
                        break

            finally:
                env.close()
                env = None

            report["output_file"] = {
                "exists_after_close": output_path.is_file(),
                "bytes": output_path.stat().st_size if output_path.is_file() else None,
                "rows_read_back": (
                    int(len(pd.read_csv(output_path))) if output_path.is_file() else None
                ),
            }

        report["sampling"] = {
            "episodes_requested": args.episodes,
            "episodes_completed": complete_episodes,
            "states_sampled": sampled_state_count,
            "unique_state_signatures": len(state_signatures),
            "state_signature_uniqueness_fraction": (
                len(state_signatures) / sampled_state_count if sampled_state_count else 0.0
            ),
            "episode_limit_reached": stop_after_episode,
            "incomplete_episode": episode_incomplete,
        }
        report["coverage"] = {
            "states_with_nonempty_mask": sampled_state_count - empty_mask_states,
            "empty_mask_states": empty_mask_states,
            "reachable_contact_states": reachable_contact_states,
            "unreachable_contact_states": unreachable_states,
            "out_of_zone_states": out_of_zone_states,
            "previous_native_shot_id_counts": dict(previous_shot_counts),
            "hitting_zone_counts": dict(hit_zone_counts),
            "y_movement_bin_counts": dict(movement_bin_counts),
            "previous_shot_zone_movement_counts": dict(shot_zone_movement_counts),
            "mask_size": {
                "min": min(mask_sizes) if mask_sizes else None,
                "max": max(mask_sizes) if mask_sizes else None,
                "mean": float(np.mean(mask_sizes)) if mask_sizes else None,
                "median": float(np.median(mask_sizes)) if mask_sizes else None,
                "p95": float(np.percentile(mask_sizes, 95)) if mask_sizes else None,
                "total_possible_actions_per_state": N_ACTIONS,
            },
            "families": {
                name: {
                    "status": values["status"],
                    "model_shot_id": values["model_shot_id"],
                    "native_shot_id": values["native_shot_id"],
                    "states_legal": values["states_legal"],
                    "legal_state_fraction": (
                        values["states_legal"] / sampled_state_count if sampled_state_count else 0.0
                    ),
                    "total_valid_joint_actions": values["total_valid_joint_actions"],
                    "landing_cells_seen_legal": sorted(values["landing_cells_seen_legal"]),
                    "recovery_cells_seen_legal": sorted(values["recovery_cells_seen_legal"]),
                    "validator_checks": values["validator_checks"],
                    "validator_failures": values["validator_failures"],
                }
                for name, values in aggregate_family.items()
            },
        }
        report["validator_cross_check"] = {
            "valid_candidates_checked": total_validator_checks,
            "failures_count": total_validator_failures,
            "all_masked_in_actions_accepted_without_mutation": (
                total_validator_checks > 0 and total_validator_failures == 0
            ),
            "scope": "all mask-in candidates in every sampled state",
            "masked_out_actions_tested": False,
            "mismatch_examples": validator_failure_examples,
            "interpretation": (
                "This tests false-positive legality in the proposed mask. It does not prove "
                "that every mask-out action is truly illegal; validate_action is a repair path."
            ),
        }

        if sampled_state_count == 0:
            report["status"] = "NO_STATES_SAMPLED"
        elif total_validator_failures:
            report["status"] = "MASK_VALIDATOR_MISMATCH"
        elif total_validator_checks == 0:
            report["status"] = "NO_VALIDATOR_CANDIDATES"
        elif episode_incomplete:
            report["status"] = "INCOMPLETE_EPISODE"
        elif complete_episodes == 0:
            report["status"] = "NO_COMPLETE_EPISODES"
        else:
            report["status"] = "PASS"

        warnings = []
        if empty_mask_states:
            warnings.append(
                f"{empty_mask_states} sampled states had empty masks; inspect whether they were "
                "unreachable contacts or genuine no-joint-action states."
            )
        if not total_validator_checks:
            warnings.append("No mask-in candidates were available for native validator cross-checking.")
        for name, values in report["coverage"]["families"].items():
            if values["status"] == "supported_proposed" and values["states_legal"] == 0:
                warnings.append(
                    f"Proposed supported family {name} was never legal in the sampled states; "
                    "this is not proof that it is globally illegal."
                )
        if stop_after_episode:
            warnings.append("The --max-states limit was reached; the final sampled rally was completed.")
        report["warnings"] = warnings

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
        "episodes_completed": report.get("sampling", {}).get("episodes_completed"),
        "states_sampled": report.get("sampling", {}).get("states_sampled"),
        "unique_states": report.get("sampling", {}).get("unique_state_signatures"),
        "empty_mask_states": report.get("coverage", {}).get("empty_mask_states"),
        "valid_candidates_checked": report.get("validator_cross_check", {}).get("valid_candidates_checked"),
        "validator_failures": report.get("validator_cross_check", {}).get("failures_count"),
        "valid_states_by_family": {
            name: values.get("states_legal")
            for name, values in report.get("coverage", {}).get("families", {}).items()
        },
        "error": report.get("error"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
