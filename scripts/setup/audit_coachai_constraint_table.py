#!/usr/bin/env python3
"""Exhaustively audit CoachAI+'s 10 x 24 x 6 native shot-constraint table.

Run from the badminton-tactical-rl project root with the isolated worker env:

    .venv-coachai/bin/python scripts/setup/audit_coachai_constraint_table.py

This is a read-only audit. It constructs BadmintonConstraint from the pinned
CoachAI+ checkout and ShuttleSet-derived All_dataset.csv, enumerates every
(previous native shot ID, hitting zone, y-movement bin) context, and writes a
JSON report. It does not instantiate RallyNet, create a Gymnasium environment,
run a rally, mutate the CoachAI+ checkout, or call validate_action().

Important scope:
- Exhaustive: action_distribution_1 (10 previous shot IDs x 24 zones x 6 bins).
- Static table inspection: action_distribution_2 (native shot -> legal landing zones).
- Not covered: continuous contact reachability, recovery projection, court-grid
  discretization, and the final joint mask at arbitrary observed coordinates.

The canonical-to-native family table below is still PROPOSED, not frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "4fd05934f172eec16c74d05c2902983b64c8521e"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COACHAI_ROOT = (PROJECT_ROOT.parent / "external" / "CoachAI-Plus").resolve()
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "b2_coachai_constraint_table_audit.json"

N_PREVIOUS_SHOTS = 10
N_HIT_ZONES = 24
N_MOVEMENT_BINS = 6
N_NATIVE_OUTPUT_TYPES = 10
EXPECTED_CONTEXTS = N_PREVIOUS_SHOTS * N_HIT_ZONES * N_MOVEMENT_BINS

# Canonical family -> RallyNet/model ID -> CoachAI+ native ID.
# This mirrors the current proposal; it is not an approval of the semantic mapping.
PROPOSED_FAMILY_MAP = (
    {"family": "CLEAR", "family_id": 0, "model_shot_id": 3, "native_shot_id": 2},
    {"family": "DROP", "family_id": 1, "model_shot_id": 5, "native_shot_id": 9},
    {"family": "SMASH", "family_id": 2, "model_shot_id": 4, "native_shot_id": 4},
    {"family": "HALF_SMASH", "family_id": 3, "model_shot_id": None, "native_shot_id": None},
    {"family": "DRIVE", "family_id": 4, "model_shot_id": 7, "native_shot_id": 6},
    {"family": "PUSH", "family_id": 5, "model_shot_id": 9, "native_shot_id": 3},
    {"family": "LIFT", "family_id": 6, "model_shot_id": 6, "native_shot_id": 8},
    {"family": "NET_SHOT", "family_id": 7, "model_shot_id": 8, "native_shot_id": 7},
    {"family": "BLOCK", "family_id": 8, "model_shot_id": None, "native_shot_id": None},
    {"family": "CUT", "family_id": 9, "model_shot_id": None, "native_shot_id": None},
)

# Inverse of the pinned Constraint.py t2c mapping. This is useful for the
# separate fallback contract, not used to change the proposed policy action space.
MODEL_ID_BY_NATIVE_ID = {
    1: 1, 10: 2, 2: 3, 4: 4, 9: 5, 8: 6,
    6: 7, 7: 8, 3: 9, 5: 10, 11: 11,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_checkout(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        raise RuntimeError(f"CoachAI+ checkout does not contain .git: {root}")
    head = git_output(root, "rev-parse", "HEAD")
    if head != EXPECTED_COMMIT:
        raise RuntimeError(
            "CoachAI+ commit mismatch: "
            f"expected {EXPECTED_COMMIT}, found {head}"
        )

    constraint_path = root / "BadmintonEnv" / "Utils" / "Constraint.py"
    dataset_path = root / "input_data" / "All_dataset.csv"
    if not constraint_path.is_file():
        raise FileNotFoundError(constraint_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    # A locally edited Constraint.py would invalidate the pinned-source audit.
    changed_constraint = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--",
         "BadmintonEnv/Utils/Constraint.py"],
        check=False,
    ).returncode != 0
    staged_constraint = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet", "HEAD", "--",
         "BadmintonEnv/Utils/Constraint.py"],
        check=False,
    ).returncode != 0
    if changed_constraint or staged_constraint:
        raise RuntimeError(
            "BadmintonEnv/Utils/Constraint.py has local changes; restore the pinned file "
            "before running this source-table audit."
        )

    dirty = git_output(root, "status", "--porcelain")
    return {
        "expected_commit": EXPECTED_COMMIT,
        "head": head,
        "working_tree_clean": not bool(dirty),
        "working_tree_status_porcelain": dirty,
        "constraint_file": {
            "path": str(constraint_path),
            "sha256": sha256_file(constraint_path),
            "bytes": constraint_path.stat().st_size,
        },
        "dataset_file": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
        },
    }


def _normalise_native_names(type_mapping: dict[Any, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, native_id in type_mapping.items():
        result[int(native_id)] = str(name)
    return result


def audit_distribution_tables(
    action_distribution_1: dict[Any, Any],
    action_distribution_2: dict[Any, Any],
    native_names: dict[int, str],
) -> dict[str, Any]:
    """Pure audit core, also exercised with synthetic tables in --self-test."""
    distribution_keys = {int(key) for key in action_distribution_1.keys()}
    expected_keys = set(range(1, N_PREVIOUS_SHOTS + 1))
    if distribution_keys != expected_keys:
        raise ValueError(
            "Unexpected action_distribution_1 previous-shot keys: "
            f"expected {sorted(expected_keys)}, got {sorted(distribution_keys)}"
        )

    family_by_native = {
        int(entry["native_shot_id"]): str(entry["family"])
        for entry in PROPOSED_FAMILY_MAP
        if entry["native_shot_id"] is not None
    }
    mapped_native_ids = set(family_by_native)
    native_id_presence: Counter[str] = Counter()
    family_context_presence: Counter[str] = Counter()
    previous_summary: dict[str, dict[str, Any]] = {}
    zone_summary: dict[str, dict[str, Any]] = {}
    movement_summary: dict[str, dict[str, Any]] = {}
    empty_native_by_previous: Counter[str] = Counter()
    empty_native_by_zone: Counter[str] = Counter()
    empty_native_by_movement: Counter[str] = Counter()
    empty_b2_by_previous: Counter[str] = Counter()
    empty_b2_by_zone: Counter[str] = Counter()
    empty_b2_by_movement: Counter[str] = Counter()
    native_only_unmapped_by_id: Counter[str] = Counter()
    contexts: list[dict[str, Any]] = []

    for previous_id in range(1, N_PREVIOUS_SHOTS + 1):
        previous_table = action_distribution_1[previous_id]
        if len(previous_table) != N_HIT_ZONES:
            raise ValueError(
                f"Previous shot {previous_id}: expected {N_HIT_ZONES} hit zones, "
                f"got {len(previous_table)}"
            )
        previous_summary[str(previous_id)] = {
            "native_name": native_names.get(previous_id, f"native_{previous_id}"),
            "contexts": 0,
            "native_empty_contexts": 0,
            "b2_empty_contexts": 0,
            "families_legal_contexts": {name: 0 for name in family_by_native.values()},
        }

        for zone_index, zone_table in enumerate(previous_table):
            hit_zone = zone_index + 1
            if len(zone_table) != N_MOVEMENT_BINS:
                raise ValueError(
                    f"Previous shot {previous_id}, zone {hit_zone}: expected "
                    f"{N_MOVEMENT_BINS} movement bins, got {len(zone_table)}"
                )
            zone_key = str(hit_zone)
            zone_summary.setdefault(zone_key, {
                "contexts": 0,
                "native_empty_contexts": 0,
                "b2_empty_contexts": 0,
                "families_legal_contexts": {name: 0 for name in family_by_native.values()},
            })

            for movement_bin, raw_vector in enumerate(zone_table):
                vector = list(raw_vector)
                if len(vector) != N_NATIVE_OUTPUT_TYPES:
                    raise ValueError(
                        f"Context ({previous_id}, {hit_zone}, {movement_bin}): "
                        f"expected output vector length {N_NATIVE_OUTPUT_TYPES}, got {len(vector)}"
                    )
                values = [int(value) for value in vector]
                if any(value not in (0, 1) for value in values):
                    raise ValueError(
                        f"Context ({previous_id}, {hit_zone}, {movement_bin}) contains non-binary values"
                    )

                native_allowed = [index + 1 for index, value in enumerate(values) if value]
                b2_allowed_families = [
                    family_by_native[native_id]
                    for native_id in native_allowed
                    if native_id in family_by_native
                ]
                unmapped_native_allowed = [
                    native_id for native_id in native_allowed
                    if native_id not in mapped_native_ids
                ]
                native_empty = len(native_allowed) == 0
                b2_empty = len(b2_allowed_families) == 0

                previous_key = str(previous_id)
                movement_key = str(movement_bin)
                movement_summary.setdefault(movement_key, {
                    "contexts": 0,
                    "native_empty_contexts": 0,
                    "b2_empty_contexts": 0,
                    "families_legal_contexts": {name: 0 for name in family_by_native.values()},
                })

                previous_summary[previous_key]["contexts"] += 1
                zone_summary[zone_key]["contexts"] += 1
                movement_summary[movement_key]["contexts"] += 1
                if native_empty:
                    empty_native_by_previous[previous_key] += 1
                    empty_native_by_zone[zone_key] += 1
                    empty_native_by_movement[movement_key] += 1
                    previous_summary[previous_key]["native_empty_contexts"] += 1
                    zone_summary[zone_key]["native_empty_contexts"] += 1
                    movement_summary[movement_key]["native_empty_contexts"] += 1
                if b2_empty:
                    empty_b2_by_previous[previous_key] += 1
                    empty_b2_by_zone[zone_key] += 1
                    empty_b2_by_movement[movement_key] += 1
                    previous_summary[previous_key]["b2_empty_contexts"] += 1
                    zone_summary[zone_key]["b2_empty_contexts"] += 1
                    movement_summary[movement_key]["b2_empty_contexts"] += 1
                for native_id in native_allowed:
                    native_id_presence[str(native_id)] += 1
                for family_name in b2_allowed_families:
                    family_context_presence[family_name] += 1
                    previous_summary[previous_key]["families_legal_contexts"][family_name] += 1
                    zone_summary[zone_key]["families_legal_contexts"][family_name] += 1
                    movement_summary[movement_key]["families_legal_contexts"][family_name] += 1
                for native_id in unmapped_native_allowed:
                    native_only_unmapped_by_id[str(native_id)] += 1

                contexts.append({
                    "previous_native_shot_id": previous_id,
                    "previous_native_shot_name": native_names.get(previous_id),
                    "hit_zone": hit_zone,
                    "y_movement_bin": movement_bin,
                    "allowed_native_shot_ids": native_allowed,
                    "allowed_native_shot_names": [
                        native_names.get(native_id, f"native_{native_id}")
                        for native_id in native_allowed
                    ],
                    "allowed_proposed_b2_families": b2_allowed_families,
                    "allowed_native_ids_without_proposed_b2_family": unmapped_native_allowed,
                    "native_shot_constraint_empty": native_empty,
                    "proposed_b2_shot_set_empty": b2_empty,
                })

    family_summary = {}
    for family in PROPOSED_FAMILY_MAP:
        name = str(family["family"])
        native_id = family["native_shot_id"]
        if native_id is None:
            family_summary[name] = {
                "status": "unsupported_proposed",
                "native_shot_id": None,
                "contexts_legal": 0,
                "context_fraction": 0.0,
            }
        else:
            count = int(family_context_presence[name])
            family_summary[name] = {
                "status": "supported_proposed",
                "native_shot_id": int(native_id),
                "contexts_legal": count,
                "context_fraction": count / EXPECTED_CONTEXTS,
            }

    landing_summary: dict[str, Any] = {}
    landing_valid_zones_by_native: dict[int, set[int]] = {}
    for native_id in range(1, N_NATIVE_OUTPUT_TYPES + 1):
        raw = action_distribution_2.get(native_id)
        if raw is None:
            raise ValueError(f"action_distribution_2 is missing native shot ID {native_id}")
        if len(raw) != N_HIT_ZONES:
            raise ValueError(
                f"action_distribution_2[{native_id}] expected {N_HIT_ZONES} zones, got {len(raw)}"
            )
        valid_zones = {index + 1 for index, is_valid in enumerate(raw) if bool(is_valid)}
        landing_valid_zones_by_native[native_id] = valid_zones
        landing_summary[str(native_id)] = {
            "native_name": native_names.get(native_id, f"native_{native_id}"),
            "valid_landing_zone_count": len(valid_zones),
            "valid_landing_zone_ids": sorted(valid_zones),
        }

    native_fallback_ids = set(range(2, 10))
    common_fallback_landing_zones = set.intersection(
        *(landing_valid_zones_by_native[native_id] for native_id in native_fallback_ids)
    )
    fallback_safe_zones = sorted(common_fallback_landing_zones)

    expected_context_key = "previous=1|zone=11|movement=1"
    known_context = next(
        (
            row for row in contexts
            if row["previous_native_shot_id"] == 1
            and row["hit_zone"] == 11
            and row["y_movement_bin"] == 1
        ),
        None,
    )

    return {
        "dimensions": {
            "previous_native_shot_ids": N_PREVIOUS_SHOTS,
            "hitting_zones": N_HIT_ZONES,
            "y_movement_bins": N_MOVEMENT_BINS,
            "native_output_categories": N_NATIVE_OUTPUT_TYPES,
            "expected_context_count": EXPECTED_CONTEXTS,
            "audited_context_count": len(contexts),
        },
        "summary": {
            "native_empty_contexts": sum(empty_native_by_previous.values()),
            "native_empty_context_fraction": sum(empty_native_by_previous.values()) / EXPECTED_CONTEXTS,
            "proposed_b2_shot_set_empty_contexts": sum(empty_b2_by_previous.values()),
            "proposed_b2_shot_set_empty_context_fraction": sum(empty_b2_by_previous.values()) / EXPECTED_CONTEXTS,
            "native_nonempty_but_b2_empty_contexts": sum(
                1 for row in contexts
                if not row["native_shot_constraint_empty"] and row["proposed_b2_shot_set_empty"]
            ),
            "contexts_with_at_least_one_proposed_b2_family": sum(
                1 for row in contexts if row["allowed_proposed_b2_families"]
            ),
        },
        "native_allowed_context_counts_by_shot_id": dict(sorted(native_id_presence.items(), key=lambda item: int(item[0]))),
        "proposed_b2_family_legal_context_counts": family_summary,
        "native_categories_without_proposed_b2_family_counts": dict(
            sorted(native_only_unmapped_by_id.items(), key=lambda item: int(item[0]))
        ),
        "empty_contexts_by_previous_native_shot": {
            "native_empty": dict(sorted(empty_native_by_previous.items(), key=lambda item: int(item[0]))),
            "proposed_b2_empty": dict(sorted(empty_b2_by_previous.items(), key=lambda item: int(item[0]))),
        },
        "empty_contexts_by_hitting_zone": {
            "native_empty": dict(sorted(empty_native_by_zone.items(), key=lambda item: int(item[0]))),
            "proposed_b2_empty": dict(sorted(empty_b2_by_zone.items(), key=lambda item: int(item[0]))),
        },
        "empty_contexts_by_movement_bin": {
            "native_empty": dict(sorted(empty_native_by_movement.items(), key=lambda item: int(item[0]))),
            "proposed_b2_empty": dict(sorted(empty_b2_by_movement.items(), key=lambda item: int(item[0]))),
        },
        "previous_shot_breakdown": previous_summary,
        "hitting_zone_breakdown": zone_summary,
        "movement_bin_breakdown": movement_summary,
        "landing_constraint_table": landing_summary,
        "fallback_design_inputs": {
            "native_fallback_candidate_ids_from_pinned_validator_range_2_to_9": sorted(native_fallback_ids),
            "common_landing_zones_valid_for_every_native_id_2_to_9": fallback_safe_zones,
            "model_id_by_native_id": {str(key): value for key, value in sorted(MODEL_ID_BY_NATIVE_ID.items())},
            "known_empty_mask_context": {
                "lookup_key": expected_context_key,
                "context": known_context,
            },
        },
        "contexts": contexts,
    }


def run_self_test() -> dict[str, Any]:
    fake_1 = {
        previous: [
            [[0] * N_NATIVE_OUTPUT_TYPES for _ in range(N_MOVEMENT_BINS)]
            for _ in range(N_HIT_ZONES)
        ]
        for previous in range(1, N_PREVIOUS_SHOTS + 1)
    }
    # Make exactly one family-supported native shot legal in every context.
    for previous in fake_1:
        for zone in range(N_HIT_ZONES):
            for movement in range(N_MOVEMENT_BINS):
                fake_1[previous][zone][movement][1] = 1  # native CLEAR ID 2
    fake_2 = {native_id: [1] * N_HIT_ZONES for native_id in range(1, N_NATIVE_OUTPUT_TYPES + 1)}
    names = {1: "Serve short", 2: "Clear", 3: "Push Shot", 4: "Smash", 5: "Smash Defence",
             6: "Drive", 7: "Net Shot", 8: "Lob", 9: "Drop", 10: "Serve long"}
    result = audit_distribution_tables(fake_1, fake_2, names)
    assert result["dimensions"]["audited_context_count"] == EXPECTED_CONTEXTS
    assert result["summary"]["native_empty_contexts"] == 0
    assert result["summary"]["proposed_b2_shot_set_empty_contexts"] == 0
    assert result["proposed_b2_family_legal_context_counts"]["CLEAR"]["contexts_legal"] == EXPECTED_CONTEXTS
    assert result["fallback_design_inputs"]["common_landing_zones_valid_for_every_native_id_2_to_9"] == list(range(1, 25))
    return {"status": "PASS", "synthetic_contexts_checked": EXPECTED_CONTEXTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_COACHAI_ROOT,
                        help="Pinned CoachAI+ checkout (default: ../external/CoachAI-Plus).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="JSON report path (default: reports/b2_coachai_constraint_table_audit.json).")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a synthetic audit-core test without importing CoachAI+ or reading data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return 0

    root = args.root.expanduser().resolve()
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()

    report: dict[str, Any] = {
        "probe_version": "b2_coachai_constraint_table_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "coachai_root": str(root),
        "expected_commit": EXPECTED_COMMIT,
        "mapping_status": "PROPOSED_ONLY_NOT_FROZEN",
        "family_mapping": list(PROPOSED_FAMILY_MAP),
        "audit_scope": {
            "action_distribution_1": "exhaustive 10 previous IDs x 24 hitting zones x 6 movement bins",
            "action_distribution_2": "all native shot IDs x 24 landing zones, static table inspection",
            "continuous_recovery_constraint": "not enumerated; correct_position is continuous/geometric",
            "contact_reachability_and_joint_grid_mask": "not part of this table-only audit",
            "validate_action_called": False,
            "rallies_simulated": 0,
            "rallynet_loaded": False,
        },
    }
    old_cwd = Path.cwd()

    try:
        checkout = verify_checkout(root)
        report["coachai_checkout"] = checkout
        os.chdir(root)
        sys.path.insert(0, str(root / "BadmintonEnv"))

        # Match the import path used by CoachAI+'s Environment.py.
        from Utils.Constraint import BadmintonConstraint  # type: ignore[import-not-found]

        constraint_path = Path(sys.modules["Utils.Constraint"].__file__).resolve()
        expected_constraint_path = (root / "BadmintonEnv" / "Utils" / "Constraint.py").resolve()
        if constraint_path != expected_constraint_path:
            raise RuntimeError(
                f"Imported unexpected Constraint.py: {constraint_path}; expected {expected_constraint_path}"
            )

        constructor_started = datetime.now(timezone.utc)
        constraint = BadmintonConstraint(file_path=str(root / "input_data" / "All_dataset.csv"))
        constructor_finished = datetime.now(timezone.utc)
        native_names = _normalise_native_names(constraint.type_mapping)
        audit = audit_distribution_tables(
            constraint.action_distribution_1,
            constraint.action_distribution_2,
            native_names,
        )
        report["constraint_runtime"] = {
            "python": sys.version.split()[0],
            "constraint_class": f"{BadmintonConstraint.__module__}.{BadmintonConstraint.__name__}",
            "constraint_source_path": str(constraint_path),
            "constructor_started_utc": constructor_started.isoformat(),
            "constructor_finished_utc": constructor_finished.isoformat(),
            "action_distribution_1_previous_shot_keys": sorted(
                int(value) for value in constraint.action_distribution_1.keys()
            ),
            "action_distribution_2_native_shot_keys": sorted(
                int(value) for value in constraint.action_distribution_2.keys()
            ),
            "action_distribution_3_entry_count": len(constraint.action_distribution_3),
            "native_type_mapping": {str(key): value for key, value in sorted(native_names.items())},
        }
        report["audit"] = audit
        report["status"] = (
            "PASS"
            if audit["dimensions"]["audited_context_count"] == EXPECTED_CONTEXTS
            else "INCOMPLETE_AUDIT"
        )
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        os.chdir(old_cwd)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    audit = report.get("audit", {})
    summary = {
        "status": report.get("status"),
        "report": str(report_path),
        "contexts_checked": audit.get("dimensions", {}).get("audited_context_count"),
        "native_empty_contexts": audit.get("summary", {}).get("native_empty_contexts"),
        "b2_proposed_shot_set_empty_contexts": audit.get("summary", {}).get("proposed_b2_shot_set_empty_contexts"),
        "native_nonempty_but_b2_empty_contexts": audit.get("summary", {}).get("native_nonempty_but_b2_empty_contexts"),
        "b2_family_context_coverage": {
            name: values.get("contexts_legal")
            for name, values in audit.get("proposed_b2_family_legal_context_counts", {}).items()
        },
        "common_fallback_landing_zones_2_to_9": audit.get("fallback_design_inputs", {}).get(
            "common_landing_zones_valid_for_every_native_id_2_to_9"
        ),
        "error": report.get("error"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
