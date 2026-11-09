#!/usr/bin/env python3
"""Runtime-test B2 empty-mask fallback construction against pinned CoachAI+.

Run with the isolated CoachAI+ Python environment from the project root:

    .venv-coachai/bin/python scripts/setup/probe_b2_fallback_runtime.py

This is a diagnostic harness, NOT the production worker or Gymnasium wrapper. It
uses three controlled state fixtures and the real pinned CoachAI+ validator and
Environment.step() implementation. A stub B agent returns None after A's
fallback transition so the test can isolate the A-side native call without
loading RallyNet or consuming any RallyNet RNG stream.

The NO_JOINT_ACTION_AFTER_GEOMETRY case is explicitly fault-injected if the
canonical grid itself has a feasible action. The report says whether a natural
empty geometry case was found by the exhaustive grid-geometry scan.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_fallback_runtime_probe.json"
DEFAULT_EVENTS = PROJECT_ROOT / "reports" / "b2_fallback_events_probe.jsonl"
EXPECTED_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"
EXPECTED_CONSTRAINT_SHA256 = "779a9756690a1c1b4efcda57cf0b0ab0089a17bf716a9554f76b1e40957ee3c3"

# Canonical family -> RallyNet/model shot ID -> CoachAI native shot ID.
# This mapping remains proposed; the harness tests the fallback boundary only.
PROPOSED_FAMILY_MAP = {
    "CLEAR": {"model_id": 3, "native_id": 2},
    "DROP": {"model_id": 5, "native_id": 9},
    "SMASH": {"model_id": 4, "native_id": 4},
    "DRIVE": {"model_id": 7, "native_id": 6},
    "PUSH": {"model_id": 9, "native_id": 3},
    "LIFT": {"model_id": 6, "native_id": 8},
    "NET_SHOT": {"model_id": 8, "native_id": 7},
}
MODEL_ID_BY_NATIVE_ID = {
    1: 1, 10: 2, 2: 3, 4: 4, 9: 5, 8: 6,
    6: 7, 7: 8, 3: 9, 5: 10, 11: 11,
}
NATIVE_FALLBACK_IDS = tuple(range(2, 10))
EXPECTED_COMMON_FALLBACK_ZONES = [9, 10, 11, 16]

N_FAMILIES, N_LANDING, N_RECOVERY = 10, 15, 9
HALF_WIDTH_M, HALF_LENGTH_M = 2.59, 6.70
NATIVE_HALF_WIDTH, NATIVE_HALF_LENGTH = 127.5, 330.0
EXPECTED_POLICY_LANDING_ZONES = [24, 23, 21, 20, 19, 17, 16, 15, 13, 8, 7, 5, 4, 3, 1]

REASON_NATIVE_EMPTY = "NATIVE_CONSTRAINT_EMPTY"
REASON_TAXONOMY_GAP = "B2_TAXONOMY_GAP"
REASON_GEOMETRY_EMPTY = "NO_JOINT_ACTION_AFTER_GEOMETRY"


class ProbeError(RuntimeError):
    """A required fallback probe invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkout(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        raise ProbeError(f"CoachAI+ checkout has no .git directory: {root}")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != EXPECTED_COMMIT:
        raise ProbeError(f"Expected CoachAI+ {EXPECTED_COMMIT}, found {head}")
    constraint_path = root / "BadmintonEnv" / "Utils" / "Constraint.py"
    dataset_path = root / "input_data" / "All_dataset.csv"
    if not constraint_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("Pinned CoachAI+ Constraint.py or All_dataset.csv is missing")
    source_hash = sha256_file(constraint_path)
    if source_hash != EXPECTED_CONSTRAINT_SHA256:
        raise ProbeError(
            "Pinned Constraint.py hash mismatch: "
            f"expected {EXPECTED_CONSTRAINT_SHA256}, got {source_hash}"
        )
    return {
        "expected_commit": EXPECTED_COMMIT,
        "head": head,
        "constraint_sha256": source_hash,
        "dataset_sha256": sha256_file(dataset_path),
        "constraint_source_path": str(constraint_path.resolve()),
    }


def derive_fallback_seed(
    worker_seed: int, episode_index: int, decision_index: int, fallback_counter: int
) -> int:
    """Stable seed independent of Python's process-randomized hash()."""
    payload = (
        f"b2-fallback-v1|{int(worker_seed)}|{int(episode_index)}|"
        f"{int(decision_index)}|{int(fallback_counter)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"B2-fallback-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def json_safe(value: Any) -> Any:
    """Convert native tuples / NumPy scalar values to JSON-safe primitives."""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return repr(value)


def state_signature(state: Any) -> str:
    data = {
        "previous_native_shot_id": int(state[0]),
        "player_position": [round(float(v), 7) for v in state[1]],
        "opponent_position": [round(float(v), 7) for v in state[2]],
        "shuttle_position": [round(float(v), 7) for v in state[3]],
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def movement_bin(state: Any) -> int:
    movement = (float(state[3][1]) - float(state[1][1])) / 240.0
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


def native_context(constraint: Any, state: Any) -> dict[str, Any]:
    hit_zone = int(constraint.coordinate_to_block(state[3], "self"))
    move_bin = movement_bin(state)
    previous_id = int(state[0])
    if not 1 <= hit_zone <= 24:
        raise ProbeError(f"Fixture contact maps outside the 24 native hit zones: {hit_zone}")
    vector = constraint.action_distribution_1[previous_id][hit_zone - 1][move_bin]
    allowed = [int(i) + 1 for i, value in enumerate(vector) if int(value) == 1]
    return {
        "previous_native_shot_id": previous_id,
        "hit_zone": hit_zone,
        "y_movement_bin": move_bin,
        "native_allowed_shot_ids": allowed,
    }


def classify_empty_reason(context: dict[str, Any]) -> str:
    allowed = set(context["native_allowed_shot_ids"])
    if not allowed:
        return REASON_NATIVE_EMPTY
    mapped = {entry["native_id"] for entry in PROPOSED_FAMILY_MAP.values()}
    if not (allowed & mapped):
        non_service_unmapped = allowed & set(NATIVE_FALLBACK_IDS) - mapped
        if non_service_unmapped:
            return REASON_TAXONOMY_GAP
        raise ProbeError(
            "Native allowed categories do not intersect B2 families, but none are recognized "
            "non-service fallback IDs 2..9; inspect the taxonomy contract."
        )
    # Native table coverage alone cannot classify a geometry-empty mask.
    # A separate full joint-grid evaluation must establish that condition.
    return "NATIVE_AND_B2_SHOT_SET_NONEMPTY"


def b2_reward_from_score_delta(delta_a: int, delta_b: int) -> float:
    """B2 rally outcome reward; independent of CoachAI+'s internal validator reward."""
    if delta_a == 1 and delta_b == 0:
        return 1.0
    if delta_b == 1 and delta_a == 0:
        return -1.0
    if delta_a == 0 and delta_b == 0:
        return 0.0
    raise ProbeError(f"Unexpected one-rally score delta: A={delta_a}, B={delta_b}")


def native_zone_centre(constraint: Any, zone_id: int, *, opponent_frame: bool) -> tuple[float, float]:
    """Return an interior point for a native block zone without using CoachAI RNG."""
    area = next((entry for entry in constraint.block_area if int(entry[2]) == int(zone_id)), None)
    if area is None:
        raise ProbeError(f"Native block zone {zone_id} is absent from block_area")
    centre_x, centre_y = float(area[3][0]), float(area[3][1])
    if opponent_frame:
        point = (177.5 - centre_x, 480.0 - centre_y)
        actual_zone = int(constraint.coordinate_to_block(point, "opp"))
    else:
        point = (centre_x - 177.5, centre_y - 480.0)
        actual_zone = int(constraint.coordinate_to_block(point, "self"))
    if actual_zone != int(zone_id):
        raise ProbeError(
            f"Zone-centre transform failed: requested {zone_id}, mapped to {actual_zone}"
        )
    return float(point[0]), float(point[1])


def canonical_grid_points() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    landing: list[tuple[float, float]] = []
    for cell_id in range(N_LANDING):
        row, col = divmod(cell_id, 3)
        x_m = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
        y_m = (row + 0.5) * (HALF_LENGTH_M / 5.0)
        landing.append((
            x_m * NATIVE_HALF_WIDTH / HALF_WIDTH_M,
            y_m * NATIVE_HALF_LENGTH / HALF_LENGTH_M,
        ))
    recovery: list[tuple[float, float]] = []
    for cell_id in range(N_RECOVERY):
        row, col = divmod(cell_id, 3)
        x_m = -HALF_WIDTH_M + (col + 0.5) * (2.0 * HALF_WIDTH_M / 3.0)
        y_m = -(row + 0.5) * (HALF_LENGTH_M / 3.0)
        recovery.append((
            x_m * NATIVE_HALF_WIDTH / HALF_WIDTH_M,
            y_m * NATIVE_HALF_LENGTH / HALF_LENGTH_M,
        ))
    return landing, recovery


def common_native_fallback_landing_zones(constraint: Any) -> list[int]:
    common = set(range(1, 25))
    for native_id in NATIVE_FALLBACK_IDS:
        vector = constraint.action_distribution_2[native_id]
        valid = {i + 1 for i, value in enumerate(vector) if int(value) == 1}
        common &= valid
    return sorted(common)


def evaluate_policy_grid_count(
    env: Any, constraint: Any, state: Any,
    landing_points: list[tuple[float, float]],
    recovery_points: list[tuple[float, float]],
    landing_zones: list[int],
) -> dict[str, Any]:
    context = native_context(constraint, state)
    allowed = set(context["native_allowed_shot_ids"])
    native_to_family = {v["native_id"]: (k, v["model_id"]) for k, v in PROPOSED_FAMILY_MAP.items()}
    hit = (float(state[3][0]), float(state[3][1]))
    contact_dx = (hit[0] - float(state[1][0])) / 177.5
    contact_dy = (hit[1] - float(state[1][1])) / 240.0
    contact_reachable = math.hypot(contact_dx, contact_dy) < 1.6
    family_counts: dict[str, int] = {}
    total = 0
    for native_id in sorted(allowed & set(native_to_family)):
        family, model_id = native_to_family[native_id]
        valid_landing = [
            i for i, zone in enumerate(landing_zones)
            if int(constraint.action_distribution_2[native_id][zone - 1]) == 1
        ]
        valid_recovery = []
        for i, recovery in enumerate(recovery_points):
            projected = constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
            if math.isclose(float(projected[0]), recovery[0], abs_tol=1e-9) and math.isclose(
                float(projected[1]), recovery[1], abs_tol=1e-9
            ):
                valid_recovery.append(i)
        valid_joint = 0
        if contact_reachable:
            for landing_i in valid_landing:
                landing = landing_points[landing_i]
                for recovery_i in valid_recovery:
                    probability = [0.0] * 10
                    probability[model_id - 1] = 1.0
                    candidate = (model_id, hit, landing, recovery_points[recovery_i], tuple(probability))
                    if env.check_out_violation(candidate, state, False) is None:
                        valid_joint += 1
        family_counts[family] = valid_joint
        total += valid_joint
    return {
        "native_allowed_shot_ids": sorted(allowed),
        "mapped_allowed_native_shot_ids": sorted(allowed & set(native_to_family)),
        "valid_joint_actions": total,
        "valid_joint_actions_by_family": family_counts,
        "contact_reachable": contact_reachable,
    }


def scan_canonical_grid_geometry(env: Any, constraint: Any,
                                 landing_points: list[tuple[float, float]],
                                 recovery_points: list[tuple[float, float]],
                                 landing_zones: list[int]) -> dict[str, Any]:
    """Check if the fixed landing/recovery grids can ever empty a mapped shot set.

    Uses a representative, reachable player position for every previous-shot / hit-zone /
    movement-bin table context. This is a deterministic geometry coverage test, not a
    trajectory-frequency estimate.
    """
    movement_representatives = (-0.60, -0.375, -0.125, 0.125, 0.375, 0.60)
    recovery_count_by_hit_zone: dict[str, int] = {}
    for zone in range(1, 25):
        hit = native_zone_centre(constraint, zone, opponent_frame=False)
        count = 0
        for recovery in recovery_points:
            projected = constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
            if math.isclose(float(projected[0]), recovery[0], abs_tol=1e-9) and math.isclose(
                float(projected[1]), recovery[1], abs_tol=1e-9
            ):
                count += 1
        recovery_count_by_hit_zone[str(zone)] = count

    empty_contexts: list[dict[str, Any]] = []
    reachable_contexts = 0
    mapped_contexts = 0
    for previous_id in range(1, 11):
        for zone in range(1, 25):
            hit = native_zone_centre(constraint, zone, opponent_frame=False)
            for move_bin, movement in enumerate(movement_representatives):
                player_y = hit[1] - movement * 240.0
                # Keep the synthetic receiver inside A's own half for plausibility.
                if not -330.0 <= player_y <= 0.0:
                    continue
                state = (previous_id, (hit[0], player_y), (0.0, 200.0), hit)
                context = native_context(constraint, state)
                result = evaluate_policy_grid_count(
                    env, constraint, state, landing_points, recovery_points, landing_zones
                )
                if not result["contact_reachable"]:
                    continue
                reachable_contexts += 1
                if result["mapped_allowed_native_shot_ids"]:
                    mapped_contexts += 1
                    if result["valid_joint_actions"] == 0:
                        empty_contexts.append({
                            **context,
                            "synthetic_state": json_safe(state),
                            "valid_joint_actions_by_family": result["valid_joint_actions_by_family"],
                        })
    return {
        "method": "1440 table contexts with representative reachable contact states; fixed 3x5 landing and 3x3 recovery grid",
        "reachable_synthetic_contexts_examined": reachable_contexts,
        "contexts_with_mapped_native_shot": mapped_contexts,
        "contexts_with_mapped_shot_but_zero_joint_grid_actions": len(empty_contexts),
        "empty_context_examples": empty_contexts[:20],
        "recovery_grid_legal_cells_by_hit_zone": recovery_count_by_hit_zone,
        "minimum_recovery_cells_for_any_hit_zone": min(recovery_count_by_hit_zone.values()),
        "maximum_recovery_cells_for_any_hit_zone": max(recovery_count_by_hit_zone.values()),
        "interpretation": (
            "NO_JOINT_ACTION_AFTER_GEOMETRY was observed in this deterministic grid scan."
            if empty_contexts else
            "No natural geometry-empty context was found in this deterministic scan; keep the reason code as a defensive invariant for changed grids, coordinate contracts, or future feasibility filters."
        ),
    }


def build_fixture_states(constraint: Any) -> dict[str, tuple[Any, ...]]:
    # Actual previously observed reachable CoachAI+ state: (previous=1, zone=11, bin=1).
    native_empty = (
        1,
        (6.01813166428073, -149.70839170242823),
        (-8.972772918641567, 263.10115456581116),
        (43.79595033824444, -214.00402128696442),
    )
    # Audit-confirmed static context: previous=4, zone=5, bin=1 permits native ID 5 only.
    gap_hit = native_zone_centre(constraint, 5, opponent_frame=False)
    taxonomy_gap = (4, (gap_hit[0], gap_hit[1] + 84.0), (0.0, 200.0), gap_hit)
    # Audit-confirmed mapped context: previous=1, zone=11, bin=2 permits IDs 3 and 7.
    geometry_hit = native_zone_centre(constraint, 11, opponent_frame=False)
    geometry_empty = (1, (geometry_hit[0], geometry_hit[1] + 30.0), (0.0, 200.0), geometry_hit)
    return {
        REASON_NATIVE_EMPTY: native_empty,
        REASON_TAXONOMY_GAP: taxonomy_gap,
        REASON_GEOMETRY_EMPTY: geometry_empty,
    }


class OneShotValidationCapture:
    """Capture exactly one A-state validator result, with a scoped worker RNG.

    Non-matching states (including B-side calls) are delegated immediately with the
    module's original random binding. The module binding is changed only during the
    matching A validation call and restored in a finally block.
    """
    def __init__(self, constraint: Any, constraint_module: Any, target_state: Any,
                 event_rng: random.Random):
        self.constraint = constraint
        self.constraint_module = constraint_module
        self.target_signature = state_signature(target_state)
        self.event_rng = event_rng
        self.original = constraint.validate_action
        self.captured_result: Any = None
        self.matching_calls = 0
        self.nonmatching_calls = 0
        self.installed = False

    def __enter__(self) -> "OneShotValidationCapture":
        def wrapper(state_arg: Any, action_arg: Any) -> Any:
            matches = state_signature(state_arg) == self.target_signature
            if matches and self.matching_calls == 0:
                self.matching_calls += 1
                previous_random = self.constraint_module.random
                self.constraint_module.random = self.event_rng
                try:
                    result = self.original(state_arg, action_arg)
                finally:
                    self.constraint_module.random = previous_random
                self.captured_result = result
                return result
            self.nonmatching_calls += 1
            return self.original(state_arg, action_arg)

        self.constraint.validate_action = wrapper
        self.installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.installed:
            self.constraint.validate_action = self.original
            self.installed = False


def build_fallback_carrier(
    *, env: Any, constraint: Any, state: Any, reason: str, rng: random.Random,
    policy_landing_zones: set[int],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    context = native_context(constraint, state)
    allowed = set(context["native_allowed_shot_ids"])
    mapped_native = {entry["native_id"] for entry in PROPOSED_FAMILY_MAP.values()}
    if reason == REASON_NATIVE_EMPTY:
        if allowed:
            raise ProbeError("NATIVE_CONSTRAINT_EMPTY fixture unexpectedly has native categories")
        common_zones = common_native_fallback_landing_zones(constraint)
        if common_zones != EXPECTED_COMMON_FALLBACK_ZONES:
            raise ProbeError(
                f"Common landing-zone assertion failed: {common_zones}; "
                f"expected {EXPECTED_COMMON_FALLBACK_ZONES}"
            )
        # Carrier ID 3 maps to native Clear (2), deliberately disallowed in this state.
        # The pinned validator's empty-vector branch then selects native ID 2..9 itself.
        model_id = 3
        landing_zone = int(rng.choice(common_zones))
        selection_method = "pinned_native_empty_vector_random_choice_2_through_9"
        allowed_ids_for_selection = list(NATIVE_FALLBACK_IDS)
    elif reason == REASON_TAXONOMY_GAP:
        if not allowed:
            raise ProbeError("B2_TAXONOMY_GAP fixture has an empty native vector")
        b2_allowed = allowed & mapped_native
        if b2_allowed:
            raise ProbeError(f"B2_TAXONOMY_GAP fixture has mapped IDs too: {sorted(b2_allowed)}")
        allowed_ids_for_selection = sorted((allowed - mapped_native) & set(NATIVE_FALLBACK_IDS))
        if not allowed_ids_for_selection:
            raise ProbeError("B2_TAXONOMY_GAP has no native category to select")
        native_id = int(rng.choice(allowed_ids_for_selection))
        model_id = MODEL_ID_BY_NATIVE_ID[native_id]
        valid_landing = [
            i + 1 for i, value in enumerate(constraint.action_distribution_2[native_id])
            if int(value) == 1
        ]
        if not valid_landing:
            raise ProbeError(f"Native taxonomy-gap shot {native_id} has no valid landing zones")
        landing_zone = int(rng.choice(valid_landing))
        selection_method = "worker_rng_choice_from_native_allowed_unmapped_ids"
    elif reason == REASON_GEOMETRY_EMPTY:
        if not (allowed & mapped_native):
            raise ProbeError("NO_JOINT_ACTION_AFTER_GEOMETRY requires a native-allowed mapped shot")
        allowed_ids_for_selection = sorted(allowed & mapped_native)
        native_id = int(rng.choice(allowed_ids_for_selection))
        model_id = MODEL_ID_BY_NATIVE_ID[native_id]
        valid_landing = [
            i + 1 for i, value in enumerate(constraint.action_distribution_2[native_id])
            if int(value) == 1 and (i + 1) not in policy_landing_zones
        ]
        if not valid_landing:
            # In a genuine grid-empty case, a valid native zone may still coincide with a
            # policy-grid landing zone; allow any native-valid zone, while keeping the
            # fallback outside the policy action record.
            valid_landing = [
                i + 1 for i, value in enumerate(constraint.action_distribution_2[native_id])
                if int(value) == 1
            ]
        if not valid_landing:
            raise ProbeError(f"Native shot {native_id} has no valid landing zone")
        landing_zone = int(rng.choice(valid_landing))
        selection_method = "worker_rng_choice_from_native_allowed_mapped_ids"
    else:
        raise ProbeError(f"Unknown fallback reason code: {reason}")

    landing = native_zone_centre(constraint, landing_zone, opponent_frame=True)
    hit = (float(state[3][0]), float(state[3][1]))
    player = (float(state[1][0]), float(state[1][1]))
    recovery = tuple(float(v) for v in constraint.correct_position(hit[0], hit[1], player[0], player[1]))
    projected_again = constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
    if not (math.isclose(projected_again[0], recovery[0], abs_tol=1e-9)
            and math.isclose(projected_again[1], recovery[1], abs_tol=1e-9)):
        raise ProbeError("Fallback recovery position does not satisfy correct_position()")
    if int(constraint.coordinate_to_block(landing, "opp")) != landing_zone:
        raise ProbeError("Fallback landing coordinate does not round-trip to its native zone")
    if int(constraint.action_distribution_2[landing_zone_for_model(constraint, model_id)][landing_zone - 1]) != 1:
        # In native-empty case, actual shot is selected inside validate_action and may vary.
        # It is therefore validated against every possible native fallback shot below.
        if reason != REASON_NATIVE_EMPTY:
            raise ProbeError("Fallback carrier landing zone is invalid for selected native shot")
    if reason == REASON_NATIVE_EMPTY:
        for native_id in NATIVE_FALLBACK_IDS:
            if int(constraint.action_distribution_2[native_id][landing_zone - 1]) != 1:
                raise ProbeError("Common fallback landing zone is not legal for every fallback shot")

    probability = [0.0] * 10
    probability[model_id - 1] = 1.0
    carrier = (int(model_id), hit, landing, recovery, tuple(probability))
    violation = env.check_out_violation(carrier, state, False)
    if violation is not None:
        raise ProbeError(f"Fallback carrier fails Environment.check_out_violation: {violation}")
    diagnostic = {
        "reason": reason,
        "native_context": context,
        "allowed_ids_for_selection": allowed_ids_for_selection,
        "selection_method": selection_method,
        "fallback_input_model_shot_id": model_id,
        "carrier_native_landing_zone_id": landing_zone,
        "carrier_landing_position_native": list(landing),
        "carrier_recovery_position_native": list(recovery),
        "recovery_is_fixed_point_of_correct_position": True,
        "carrier_passes_environment_precheck": True,
        "carrier_landing_zone_in_policy_grid": landing_zone in policy_landing_zones,
    }
    return carrier, diagnostic


def landing_zone_for_model(constraint: Any, model_id: int) -> int:
    """Return the native ID corresponding to a model ID under pinned t2c."""
    inverse = {model: native for native, model in MODEL_ID_BY_NATIVE_ID.items()}
    if model_id not in inverse:
        raise ProbeError(f"Unknown model shot ID {model_id}")
    return int(inverse[model_id])


class NoReturnAgent:
    """Test double: make B miss after the A-side native transition is recorded."""
    def serve_state(self) -> Any:
        raise RuntimeError("serve_state should not be called by this controlled-state probe")

    def action(self, state: Any, info: Any, launch: bool = False) -> None:
        return None


def reset_env_to_fixture(env: Any, state: Any, pandas_module: Any) -> None:
    env.score_a = 0
    env.score_b = 0
    env.pre_score_a = 0
    env.pre_score_b = 0
    env.rally = 1
    env.match = 1
    env.round = 1
    env.serving_player = "B"
    env.state = state
    env.output = pandas_module.DataFrame()
    env.info_dict = env.info("B", None, state, violation=False, constraint=False)


def native_action_record_matches(row: Any, native_action: Any) -> bool:
    pairs = (
        ("type", int(native_action[0])),
        ("player_hit_x", float(native_action[1][0])),
        ("player_hit_y", float(native_action[1][1])),
        ("landing_x", float(native_action[2][0])),
        ("landing_y", float(native_action[2][1])),
        ("player_movement_x", float(native_action[3][0])),
        ("player_movement_y", float(native_action[3][1])),
    )
    for key, expected in pairs:
        actual = row[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-8):
                return False
        elif int(actual) != expected:
            return False
    return True


def snapshot_rng_states(torch_module: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch_cpu": torch_module.random.get_rng_state().clone() if torch_module else None,
    }
    if torch_module is not None and torch_module.cuda.is_available():
        result["torch_cuda"] = [state.clone() for state in torch_module.cuda.get_rng_state_all()]
    else:
        result["torch_cuda"] = None
    return result


def compare_rng_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    np = __import__("numpy")
    py_same = before["python"] == after["python"]
    np_before, np_after = before["numpy"], after["numpy"]
    numpy_same = (
        np_before[0] == np_after[0]
        and np.array_equal(np_before[1], np_after[1])
        and np_before[2:] == np_after[2:]
    )
    torch_same = True
    if before["torch_cpu"] is not None:
        torch_same = bool(__import__("torch").equal(before["torch_cpu"], after["torch_cpu"]))
    cuda_same = True
    if before["torch_cuda"] is not None:
        torch = __import__("torch")
        cuda_same = len(before["torch_cuda"]) == len(after["torch_cuda"]) and all(
            torch.equal(a, b) for a, b in zip(before["torch_cuda"], after["torch_cuda"])
        )
    return {
        "python_global_rng_unchanged": py_same,
        "numpy_global_rng_unchanged": numpy_same,
        "torch_cpu_rng_unchanged": torch_same,
        "torch_cuda_rng_unchanged": cuda_same,
        "all_observed_global_rngs_unchanged": py_same and numpy_same and torch_same and cuda_same,
    }


def execute_fallback_case(
    *, env: Any, constraint_module: Any, state: Any, reason: str,
    worker_seed: int, episode_index: int, decision_index: int, fallback_counter: int,
    landing_points: list[tuple[float, float]], recovery_points: list[tuple[float, float]],
    landing_zones: list[int], torch_module: Any | None,
    synthetic_fault_injection: bool = False,
    precomputed_grid_count: int | None = None,
) -> dict[str, Any]:
    constraint = env.badminton_constraint
    context = native_context(constraint, state)
    classified = classify_empty_reason(context)
    if reason in (REASON_NATIVE_EMPTY, REASON_TAXONOMY_GAP) and reason != classified:
        raise ProbeError(f"Requested reason {reason}, but fixture classifies as {classified}")
    if reason == REASON_GEOMETRY_EMPTY:
        if classified != "NATIVE_AND_B2_SHOT_SET_NONEMPTY":
            raise ProbeError(f"Geometry-empty case requires native + mapped shot availability; got {classified}")
        if synthetic_fault_injection and precomputed_grid_count == 0:
            raise ProbeError("Fault injection requested even though the canonical joint grid is already empty")
        if not synthetic_fault_injection and precomputed_grid_count != 0:
            raise ProbeError("Natural geometry-empty case requires an evaluated grid count of zero")

    fallback_seed = derive_fallback_seed(worker_seed, episode_index, decision_index, fallback_counter)
    event_rng = random.Random(fallback_seed)
    policy_landing_zones = {int(v) for v in landing_zones}
    carrier, build_info = build_fallback_carrier(
        env=env, constraint=constraint, state=state, reason=reason,
        rng=event_rng, policy_landing_zones=policy_landing_zones,
    )
    reset_env_to_fixture(env, state, __import__("pandas"))
    if env.check_out_violation(carrier, state, False) is not None:
        raise ProbeError("Carrier action became invalid immediately before Environment.step")

    native_action_buffer: list[Any] = []
    native_logprob_buffer: list[Any] = []
    policy_action_buffer_before = len(native_action_buffer)
    policy_logprob_buffer_before = len(native_logprob_buffer)
    rng_before = snapshot_rng_states(torch_module)
    capture = OneShotValidationCapture(
        constraint, constraint_module, state, event_rng,
    )
    output_rows_before = int(len(env.output))
    score_before = {"A": int(env.score_a), "B": int(env.score_b)}
    try:
        with capture:
            # Not a launch: this deliberately enters Environment.step's controlled-A branch.
            next_state, native_step_reward, info, done, launch = env.step(carrier, is_launch=False)
    finally:
        # The context manager restores both the instance method and module random binding.
        pass
    rng_after = snapshot_rng_states(torch_module)
    rng_isolation = compare_rng_states(rng_before, rng_after)

    if capture.matching_calls != 1 or capture.captured_result is None:
        raise ProbeError(
            "Expected exactly one captured A-side validate_action call; "
            f"matching_calls={capture.matching_calls}"
        )
    checked_action, validator_reward, validator_corrected = capture.captured_result
    if len(env.output) < output_rows_before + 2:
        raise ProbeError("Expected A and B output rows after controlled native step")
    a_row = env.output.iloc[output_rows_before]
    # CoachAI+ Record() stores the side marker ("A"/"B") in output["player"].
    # agent_a_name is the participant/display name and is not the side marker.
    observed_side = str(a_row["player"])
    if observed_side != "A":
        new_rows = env.output.iloc[output_rows_before:output_rows_before + 2]
        observed = [
            {
                "player": str(row.get("player")),
                "type": int(row.get("type")),
                "serve": bool(row.get("serve")),
            }
            for _, row in new_rows.iterrows()
        ]
        raise ProbeError(
            "First newly written output row is not side A: "
            f"observed_side={observed_side!r}, expected_side='A', "
            f"agent_a_name={getattr(env, 'agent_a_name', None)!r}, "
            f"new_rows={observed}"
        )
    output_match = native_action_record_matches(a_row, checked_action)
    if not output_match:
        raise ProbeError("Captured corrected native action disagrees with env.output A row")

    score_after = {"A": int(env.score_a), "B": int(env.score_b)}
    delta_a = score_after["A"] - score_before["A"]
    delta_b = score_after["B"] - score_before["B"]
    b2_reward = b2_reward_from_score_delta(delta_a, delta_b)
    winner = "A" if b2_reward == 1.0 else "B" if b2_reward == -1.0 else None

    if reason == REASON_NATIVE_EMPTY:
        if int(checked_action[0]) not in NATIVE_FALLBACK_IDS:
            raise ProbeError(f"Native empty-vector fallback selected unexpected shot {checked_action[0]}")
        if any(float(v) != 0.0 for v in checked_action[4]):
            raise ProbeError("Pinned native empty-vector fallback did not zero the shot probability vector")
        if not bool(validator_corrected):
            raise ProbeError("Expected pinned native empty-vector fallback to report correction")
    else:
        selected_native = landing_zone_for_model(constraint, int(carrier[0]))
        if int(checked_action[0]) != selected_native:
            raise ProbeError(
                f"Native-allowed fallback shot changed unexpectedly: input mapped to {selected_native}, "
                f"returned {checked_action[0]}"
            )
        if bool(validator_corrected):
            raise ProbeError("Native-allowed fallback carrier was unexpectedly corrected")

    event = {
        "event_type": "b2_empty_mask_fallback",
        "fallback_event_id": f"episode-{episode_index}-decision-{decision_index}-fallback-{fallback_counter}",
        "episode_seed": int(worker_seed + episode_index - 1),
        "worker_seed": int(worker_seed),
        "fallback_seed": int(fallback_seed),
        "episode_index": int(episode_index),
        "decision_index": int(decision_index),
        "fallback_counter": int(fallback_counter),
        "state_signature_sha256": state_signature(state),
        "empty_mask_reason": reason,
        "reason_source": "synthetic_fault_injection" if synthetic_fault_injection else "fixture_native_context",
        "native_context": context,
        "policy_action_id": None,
        "policy_action_log_probability": None,
        "policy_selected": False,
        "fallback_input_model_shot_id": int(carrier[0]),
        "fallback_selection": build_info,
        "corrected_native_action": {
            "native_shot_id": int(checked_action[0]),
            "hitting_position": json_safe(checked_action[1]),
            "landing_position": json_safe(checked_action[2]),
            "recovery_position": json_safe(checked_action[3]),
            "shot_probability": json_safe(checked_action[4]),
        },
        "native_validator_reward": float(validator_reward),
        "native_validator_corrected": bool(validator_corrected),
        "native_validator_matching_call_count": int(capture.matching_calls),
        "native_validator_nonmatching_call_count": int(capture.nonmatching_calls),
        "captured_from_actual_env_step_call": True,
        "executed_by_native_environment": True,
        "native_step_reward": float(native_step_reward),
        "score_before": score_before,
        "score_after": score_after,
        "rally_winner_from_score_delta": winner,
        "b2_rally_reward_from_score_delta": float(b2_reward),
        "native_output_record_matches_captured_action": bool(output_match),
        "global_rng_isolation": rng_isolation,
        "policy_action_buffer_size_before": policy_action_buffer_before,
        "policy_action_buffer_size_after": len(native_action_buffer),
        "policy_logprob_buffer_size_before": policy_logprob_buffer_before,
        "policy_logprob_buffer_size_after": len(native_logprob_buffer),
        "policy_action_buffer_unchanged": len(native_action_buffer) == policy_action_buffer_before,
        "policy_logprob_buffer_unchanged": len(native_logprob_buffer) == policy_logprob_buffer_before,
        "policy_invoked": False,
        "all_false_mask_passed_to_policy": False,
        "native_done": bool(done),
        "native_launch_flag": bool(launch),
        "native_environment_transitions_executed": 1,
    }
    if not rng_isolation["all_observed_global_rngs_unchanged"]:
        raise ProbeError("Fallback consumed a process-global Python/NumPy/Torch RNG state")
    if not event["policy_action_buffer_unchanged"] or not event["policy_logprob_buffer_unchanged"]:
        raise ProbeError("Fallback contaminated a policy action/log-probability buffer")
    return event


def run_capture_unit_test() -> dict[str, Any]:
    """Prove matching A capture is scoped; non-matching B calls keep normal RNG binding."""
    normal_random = object()
    fake_module = SimpleNamespace(random=normal_random)
    target = (1, (0.0, -100.0), (0.0, 100.0), (0.0, -120.0))
    other = (2, (10.0, -100.0), (0.0, 100.0), (10.0, -120.0))
    event_rng = random.Random(1234)
    seen: list[tuple[str, Any]] = []

    class FakeConstraint:
        def validate_action(self, state_arg: Any, action_arg: Any) -> Any:
            seen.append((state_signature(state_arg), fake_module.random))
            return (action_arg, 0.0, False)

    fake = FakeConstraint()
    capture = OneShotValidationCapture(fake, fake_module, target, event_rng)
    with capture:
        fake.validate_action(other, (1,))   # B/non-target call: no RNG scope.
        fake.validate_action(target, (1,))  # A call: scoped event RNG.
        fake.validate_action(other, (1,))   # Later B call: normal RNG restored.
    assert capture.matching_calls == 1
    assert capture.nonmatching_calls == 2
    assert seen[0][1] is normal_random
    assert seen[1][1] is event_rng
    assert seen[2][1] is normal_random
    assert fake_module.random is normal_random
    return {
        "status": "PASS",
        "matching_A_calls_captured": capture.matching_calls,
        "nonmatching_calls_delegated_without_rng_scope": capture.nonmatching_calls,
        "module_random_restored": fake_module.random is normal_random,
    }


def run_unit_tests() -> dict[str, Any]:
    seed_a = derive_fallback_seed(42, 4, 1, 0)
    seed_b = derive_fallback_seed(42, 4, 1, 0)
    seed_c = derive_fallback_seed(42, 4, 1, 1)
    assert seed_a == seed_b and seed_a != seed_c
    capture_test = run_capture_unit_test()
    assert b2_reward_from_score_delta(1, 0) == 1.0
    assert b2_reward_from_score_delta(0, 1) == -1.0
    assert b2_reward_from_score_delta(0, 0) == 0.0
    # Separate event collections by construction; no shared policy-action IDs are made.
    policy_actions: list[Any] = []
    policy_logprobs: list[Any] = []
    fallback_events: list[Any] = [{"policy_action_id": None, "policy_action_log_probability": None, "policy_selected": False}]
    assert len(policy_actions) == 0 and len(policy_logprobs) == 0 and len(fallback_events) == 1
    return {
        "status": "PASS",
        "seed_derivation_repeatable": True,
        "different_fallback_counter_changes_seed": True,
        "one_shot_capture_scope": capture_test,
        "fallback_event_separate_from_policy_action_buffer": True,
        "b2_reward_mapping_plus_one_minus_one_zero": True,
    }


def create_native_environment(root: Path, output_path: Path) -> tuple[Any, Any, Any, Any]:
    root = root.expanduser().resolve()
    sys.path[:0] = [str(root), str(root / "BadmintonEnv")]
    os.chdir(root)
    env_module = importlib.import_module("BadmintonEnv.Environment")
    constraint_module = importlib.import_module("Utils.Constraint")
    expected_constraint_path = (root / "BadmintonEnv" / "Utils" / "Constraint.py").resolve()
    actual_constraint_path = Path(constraint_module.__file__).resolve()
    if actual_constraint_path != expected_constraint_path:
        raise ProbeError(
            f"Imported unexpected Constraint.py: {actual_constraint_path}; expected {expected_constraint_path}"
        )
    no_return_a, no_return_b = NoReturnAgent(), NoReturnAgent()
    env = env_module.BadmintonEnv(
        no_return_a, no_return_b, "B2_CONTROLLED_A", "B2_STUB_B", 1,
        is_match=False, is_constraint=True, is_have_serve_state=True,
        filepath=str(output_path),
    )
    return env, constraint_module, __import__("pandas"), importlib.import_module("torch")


def run_real_probe(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    checkout = verify_checkout(root)
    original_cwd = Path.cwd()
    report: dict[str, Any] = {
        "probe_version": "b2_fallback_runtime_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "coachai_checkout": checkout,
        "worker_seed": int(args.seed),
        "fallback_reason_codes": [REASON_NATIVE_EMPTY, REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY],
        "scope": {
            "actual_pinned_coachai_validator": True,
            "actual_pinned_environment_step": True,
            "rallynet_loaded": False,
            "gymnasium_wrapper_tested": False,
            "maskableppo_collector_tested": False,
            "b_agent": "stub_returns_none_after_A_transition_to_isolate_A_native_fallback",
            "production_worker_modified": False,
        },
        "unit_tests": run_unit_tests(),
        "cases": [],
    }
    env = None
    try:
        with tempfile.TemporaryDirectory(prefix="b2_fallback_runtime_") as tmp:
            temp_root = Path(tmp)
            env, constraint_module, pandas_module, torch_module = create_native_environment(
                root, temp_root / "native_output.csv"
            )
            constraint = env.badminton_constraint
            landing_points, recovery_points = canonical_grid_points()
            landing_zones = [int(constraint.coordinate_to_block(point, "opp")) for point in landing_points]
            recovery_zones = [int(constraint.coordinate_to_block(point, "self")) for point in recovery_points]
            if landing_zones != EXPECTED_POLICY_LANDING_ZONES:
                raise ProbeError(
                    f"Canonical landing-zone mapping changed: {landing_zones}; "
                    f"expected {EXPECTED_POLICY_LANDING_ZONES}"
                )
            if not all(1 <= z <= 24 for z in landing_zones + recovery_zones):
                raise ProbeError("A canonical grid point mapped outside native zones 1..24")
            report["canonical_grid"] = {
                "landing_zone_ids": landing_zones,
                "recovery_zone_ids": recovery_zones,
                "common_native_fallback_landing_zones_2_to_9": common_native_fallback_landing_zones(constraint),
            }
            if report["canonical_grid"]["common_native_fallback_landing_zones_2_to_9"] != EXPECTED_COMMON_FALLBACK_ZONES:
                raise ProbeError("Fallback common landing-zone assertion failed")
            report["canonical_grid_geometry_scan"] = scan_canonical_grid_geometry(
                env, constraint, landing_points, recovery_points, landing_zones
            )

            fixtures = build_fixture_states(constraint)
            natural_geometry_examples = report["canonical_grid_geometry_scan"]["empty_context_examples"]
            if natural_geometry_examples:
                raw_state = natural_geometry_examples[0]["synthetic_state"]
                fixtures[REASON_GEOMETRY_EMPTY] = (
                    int(raw_state[0]), tuple(raw_state[1]), tuple(raw_state[2]), tuple(raw_state[3])
                )
            cases_to_run: list[tuple[str, Any, bool]]
            if args.internal_case:
                if args.internal_case not in fixtures:
                    raise ProbeError(f"Unknown internal case {args.internal_case}")
                cases_to_run = [(args.internal_case, fixtures[args.internal_case], args.internal_case == REASON_GEOMETRY_EMPTY)]
            else:
                cases_to_run = [
                    (REASON_NATIVE_EMPTY, fixtures[REASON_NATIVE_EMPTY], False),
                    (REASON_TAXONOMY_GAP, fixtures[REASON_TAXONOMY_GAP], False),
                    (REASON_GEOMETRY_EMPTY, fixtures[REASON_GEOMETRY_EMPTY], False),
                ]

            for index, (reason, state, injected) in enumerate(cases_to_run):
                if reason == REASON_GEOMETRY_EMPTY:
                    actual_geometry = evaluate_policy_grid_count(
                        env, constraint, state, landing_points, recovery_points, landing_zones
                    )
                    injected = actual_geometry["valid_joint_actions"] > 0
                else:
                    actual_geometry = None
                    injected = False
                if args.internal_case:
                    episode_index = int(args.episode_index)
                    decision_index = int(args.decision_index)
                    fallback_counter = int(args.fallback_counter)
                else:
                    episode_index = index + 4
                    decision_index = 1
                    fallback_counter = index
                event = execute_fallback_case(
                    env=env, constraint_module=constraint_module, state=state,
                    reason=reason, worker_seed=args.seed,
                    episode_index=episode_index, decision_index=decision_index,
                    fallback_counter=fallback_counter,
                    landing_points=landing_points, recovery_points=recovery_points,
                    landing_zones=landing_zones, torch_module=torch_module,
                    synthetic_fault_injection=injected,
                    precomputed_grid_count=(actual_geometry["valid_joint_actions"] if actual_geometry is not None else None),
                )
                if actual_geometry is not None:
                    event["pre_fallback_grid_evaluation"] = actual_geometry
                    event["geometry_empty_was_natural"] = actual_geometry["valid_joint_actions"] == 0
                    event["geometry_case_uses_fault_injection"] = injected
                report["cases"].append(event)

            # Same worker + same fixture/event identity must replay identically.
            first_empty = next((e for e in report["cases"] if e["empty_mask_reason"] == REASON_NATIVE_EMPTY), None)
            if first_empty is not None:
                replay_state = fixtures[REASON_NATIVE_EMPTY]
                replay_event = execute_fallback_case(
                    env=env, constraint_module=constraint_module, state=replay_state,
                    reason=REASON_NATIVE_EMPTY, worker_seed=args.seed,
                    episode_index=int(first_empty["episode_index"]),
                    decision_index=int(first_empty["decision_index"]),
                    fallback_counter=int(first_empty["fallback_counter"]),
                    landing_points=landing_points, recovery_points=recovery_points,
                    landing_zones=landing_zones, torch_module=torch_module,
                )
                report["same_worker_replay"] = {
                    "status": "PASS" if replay_event["corrected_native_action"] == first_empty["corrected_native_action"] else "FAIL",
                    "same_event_seed": replay_event["fallback_seed"] == first_empty["fallback_seed"],
                    "same_corrected_native_action": replay_event["corrected_native_action"] == first_empty["corrected_native_action"],
                }
                if report["same_worker_replay"]["status"] != "PASS":
                    raise ProbeError("Same-worker same-event-seed fallback replay did not match")
            else:
                report["same_worker_replay"] = {"status": "NOT_APPLICABLE", "reason": "No native-empty fixture in this invocation"}

            report["runtime"] = {
                "python": sys.version.split()[0],
                "numpy": __import__("numpy").__version__,
                "torch": torch_module.__version__,
                "constraint_module": str(Path(constraint_module.__file__).resolve()),
                "native_environment_class": f"{type(env).__module__}.{type(env).__name__}",
                "validator_capture_method": "one-shot wrapper around env.badminton_constraint.validate_action; only matching A-state uses scoped worker RNG",
            }
            env.close()
            report["native_output_file_written"] = (temp_root / "native_output.csv").is_file()
            if not report["native_output_file_written"]:
                raise ProbeError("CoachAI+ close() did not write its temporary native output")
            env = None  # Already closed while the temporary output directory still exists.

        if not args.skip_fresh_process_check and not args.internal_case:
            report["fresh_process_replay"] = run_fresh_process_replay(root, args.seed)
            if report["fresh_process_replay"]["status"] != "PASS":
                raise ProbeError("Fresh-process same-seed replay did not match")
        report["status"] = "PASS"
        return report
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        os.chdir(original_cwd)


def run_fresh_process_replay(root: Path, worker_seed: int) -> dict[str, Any]:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="b2_fallback_fresh_process_") as tmp:
        directory = Path(tmp)
        outputs: list[dict[str, Any]] = []
        for run_index in range(2):
            result_path = directory / f"fresh_worker_{run_index}.json"
            command = [
                sys.executable, str(script),
                "--root", str(root),
                "--seed", str(worker_seed),
                "--internal-case", REASON_NATIVE_EMPTY,
                "--episode-index", "4",
                "--decision-index", "1",
                "--fallback-counter", "0",
                "--result-file", str(result_path),
                "--skip-fresh-process-check",
            ]
            completed = subprocess.run(
                command, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if completed.returncode != 0 or not result_path.is_file():
                raise ProbeError(
                    "Fresh worker subprocess failed: "
                    f"returncode={completed.returncode}\nstdout={completed.stdout[-2000:]}\n"
                    f"stderr={completed.stderr[-4000:]}"
                )
            child = json.loads(result_path.read_text(encoding="utf-8"))
            outputs.append(child["cases"][0]["corrected_native_action"])
        same = outputs[0] == outputs[1]
        return {
            "status": "PASS" if same else "FAIL",
            "fresh_processes": 2,
            "same_worker_seed_and_event_coordinates": True,
            "same_corrected_native_action": same,
            "corrected_native_actions": outputs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT)
    parser.add_argument("--seed", type=int, default=42, help="Worker/run seed (default: 42)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--self-test", action="store_true", help="Run pure unit tests without CoachAI+")
    parser.add_argument("--internal-case", choices=(REASON_NATIVE_EMPTY, REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY), help=argparse.SUPPRESS)
    parser.add_argument("--episode-index", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--decision-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--fallback-counter", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--skip-fresh-process-check", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_unit_tests(), indent=2))
        return 0
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if min(args.episode_index, args.decision_index, args.fallback_counter) < 0:
        raise ValueError("Episode, decision and fallback indices must be non-negative")

    report_path = args.result_file or args.report
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()
    events_path = args.events.expanduser()
    if not events_path.is_absolute():
        events_path = (PROJECT_ROOT / events_path).resolve()

    try:
        report = run_real_probe(args)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        events = report.get("cases", [])
        if not args.internal_case:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.write_text("".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n" for event in events), encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report": str(report_path),
            "events": str(events_path) if not args.internal_case else None,
            "cases": [
                {
                    "reason": event["empty_mask_reason"],
                    "native_context": event["native_context"],
                    "corrected_native_shot_id": event["corrected_native_action"]["native_shot_id"],
                    "validator_corrected": event["native_validator_corrected"],
                    "output_matches": event["native_output_record_matches_captured_action"],
                    "b2_reward": event["b2_rally_reward_from_score_delta"],
                    "global_rng_isolated": event["global_rng_isolation"]["all_observed_global_rngs_unchanged"],
                }
                for event in events
            ],
            "same_worker_replay": report.get("same_worker_replay"),
            "fresh_process_replay": report.get("fresh_process_replay"),
            "natural_geometry_empty_contexts": report.get("canonical_grid_geometry_scan", {}).get(
                "contexts_with_mapped_shot_but_zero_joint_grid_actions"
            ),
        }, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "probe_version": "b2_fallback_runtime_probe_v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
