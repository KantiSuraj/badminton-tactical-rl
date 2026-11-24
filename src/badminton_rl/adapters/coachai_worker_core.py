"""Persistent native B2 runtime and JSON-serializable state machine.

`CoachAIWorker` owns all mutable rally state (native state, mask, score
baseline, fallback counter, ...) and therefore keeps `reset`, `step` and
`_advance_to_policy_state` together.  NumPy, PyTorch and CoachAI+ are imported
lazily inside `initialize()`; nothing heavyweight is imported at module level.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import numbers
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from badminton_rl.adapters.coachai_history_serializer import (
    REQUIRED_OUTPUT_COLUMNS,
    SOURCE_FALLBACK,
    SOURCE_POLICY,
    SOURCE_SERVE,
    SOURCE_UNPLAYABLE,
    NativeTransitionCapture,
    serialize_rally_history,
)
from badminton_rl.adapters.coachai_worker_capture import OneShotValidationCapture
from badminton_rl.adapters.coachai_worker_utils import (
    ACTION_MAP_VERSION,
    EXPECTED_COACHAI_COMMIT,
    EXPECTED_COMMON_FALLBACK_ZONES,
    EXPECTED_CONSTRAINT_SHA256,
    EXPECTED_FILE_SHA256,
    EXPECTED_POLICY_LANDING_ZONES,
    FAMILY_MAP,
    MODEL_TO_NATIVE,
    NATIVE_FALLBACK_IDS,
    NATIVE_HALF_LENGTH,
    NATIVE_HALF_WIDTH,
    NATIVE_TO_MODEL,
    N_ACTIONS,
    N_FAMILIES,
    N_LANDING,
    N_RECOVERY,
    OBSERVATION_SCHEMA_VERSION,
    REASON_GEOMETRY_EMPTY,
    REASON_NATIVE_EMPTY,
    REASON_TAXONOMY_GAP,
    UNSUPPORTED_FAMILIES,
    UnexpectedNativeAgentCall,
    WorkerFault,
    _build_candidate,
    _describe_history_structure,
    _json_safe,
    _landing_centre_m,
    _metres_to_native,
    _recovery_centre_m,
    _sha256_file,
    _state_signature,
    actions_equal,
    classify_empty_mask_reason,
    decode_action,
    derive_action_jitter_seed,
    derive_fallback_seed,
    derive_reset_retry_seed,
)


class ExternalControlledAgent:
    """A-side sentinel: only the parent is allowed to choose A's action."""

    def serve_state(self) -> Any:
        raise UnexpectedNativeAgentCall("CoachAI requested A's serve state; B2 v1 requires B to serve")

    def action(self, *_: Any, **__: Any) -> Any:
        raise UnexpectedNativeAgentCall("CoachAI attempted to generate controlled side A's action")


@dataclass
class NativeStepResult:
    next_state: Any
    native_reward: float
    info: dict[str, Any]
    done: bool
    launch: bool
    checked_action: Any = None
    validator_reward: float | None = None
    validator_corrected: bool | None = None
    validation_capture: OneShotValidationCapture | None = None
    output_record_matches: bool | None = None
    score_before: dict[str, int] | None = None
    score_after: dict[str, int] | None = None
    # Absolute positions of the output rows this native transition appended (half-open).
    output_start: int | None = None
    output_end: int | None = None


class CoachAIWorker:
    """Persistent native B2 runtime and JSON-serializable state machine."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.coachai_root = Path(args.coachai_root).expanduser().resolve()
        self.expected_commit = str(args.expected_coachai_commit)
        self.player_name = str(args.player_name)
        self.opponent_name = str(args.opponent_name)
        self.device_request = str(args.device)
        self.worker_seed = int(args.worker_seed)
        self.max_decisions = int(args.max_decisions)
        self.max_internal_transitions = int(args.max_internal_transitions)
        self.max_reset_retries = int(getattr(args, "max_reset_retries", 0))
        self.cell_jitter_fraction = float(getattr(args, "cell_jitter_fraction", 0.0))
        self.fallback_win_reward = float(getattr(args, "fallback_win_reward", 1.0))
        self.fallback_events_path = Path(args.fallback_events).expanduser().resolve()
        self.np: Any = None
        self.torch: Any = None
        self.env: Any = None
        self.constraint_module: Any = None
        self.rallynet: Any = None
        self.constraint: Any = None
        self.device: Any = None
        self.checkout_fingerprint: dict[str, Any] = {}
        self._native_temp: tempfile.TemporaryDirectory[str] | None = None
        self._native_output_path: Path | None = None
        self._state: Any = None
        self._info: dict[str, Any] = {}
        self._launch = False
        self._episode_index = -1
        self._episode_seed: int | None = None
        self._episode_id: str | None = None
        self._policy_decisions = 0
        self._fallback_counter = 0
        self._score_baseline = {"A": 0, "B": 0}
        self._mask: Any = None
        self._mask_evaluation: dict[str, Any] | None = None
        self._ready = False
        self._terminated = False
        self._truncated = False
        self._closed = False
        self._constraint_hash = ""
        self._dataset_hash = ""
        self._rallynet_hash = ""
        self._landing_points: list[tuple[float, float]] = []
        self._recovery_points: list[tuple[float, float]] = []
        self._landing_zones: list[int] = []
        self._recovery_zones: list[int] = []
        self._policy_landing_zones: set[int] = set()
        self._common_fallback_zones: list[int] = []
        self._action_mapping_hash = ""
        # Attempt-scoped rally-event cursors (re-initialized on every reset attempt).
        self._output_cursor = 0
        self._output_cursor_attempt = -1
        self._history_cursor = 0
        self._event_index = 0
        self._last_ball_round = 0
        self._transition_log: list[NativeTransitionCapture] = []

    # ---------- Initialization / source verification ----------
    def initialize(self) -> None:
        if not self.coachai_root.is_dir():
            raise FileNotFoundError(f"CoachAI+ checkout not found: {self.coachai_root}")
        self._verify_action_mapping_file()
        self.checkout_fingerprint = self._verify_checkout()
        self._add_coachai_import_paths()
        self.np = importlib.import_module("numpy")
        self.torch = importlib.import_module("torch")
        self._select_device()
        self._seed_everything(self.worker_seed)
        self._install_cpu_torch_load_default()
        self._load_rallynet_opponent()
        self._construct_native_environment()
        self.constraint = self.env.badminton_constraint
        self._verify_import_origins()
        self._initialize_action_geometry()
        self._verify_native_action_tables()

    def _verify_action_mapping_file(self) -> None:
        """Fail closed if the checked-in mapping contract drifts from this worker.

        The executable mapping remains the explicit allow-list above.  The JSON
        file is its reviewable, versioned companion; verifying exact parity here
        prevents a changed config from silently describing a different policy
        action space than the running adapter.
        """
        path = self.project_root / "configs" / "action_mappings" / "coachai_v1.json"
        if not path.is_file():
            raise WorkerFault(
                "unsupported_condition", f"Required B2 action mapping file missing: {path}", fatal=True
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerFault(
                "unsupported_condition", f"Could not read B2 action mapping file: {exc}", fatal=True
            ) from exc
        expected_supported = [
            {
                "family_id": family_id,
                "family": item["family"],
                "model_shot_id": int(item["model_id"]),
                "coachai_native_shot_id": int(item["native_id"]),
                "coachai_native_name": item["native_name"],
            }
            for family_id, item in sorted(FAMILY_MAP.items())
        ]
        expected_unsupported = [
            {"family_id": family_id, "family": name}
            for family_id, name in sorted(UNSUPPORTED_FAMILIES.items())
        ]
        expected_bridge = {str(key): value for key, value in MODEL_TO_NATIVE.items()}
        expected = {
            "schema_version": ACTION_MAP_VERSION,
            "canonical_action": ["shot_family", "landing_cell", "recovery_cell"],
            "action_count": N_ACTIONS,
            "dimensions": {
                "shot_families": N_FAMILIES,
                "landing_cells": N_LANDING,
                "recovery_cells": N_RECOVERY,
            },
            "encoding": "(family_id * 15 + landing_cell_id) * 9 + recovery_cell_id",
            "supported_families": expected_supported,
            "unsupported_families": expected_unsupported,
            "model_shot_id_to_coachai_native_shot_id": expected_bridge,
            "service_policy": "B_SERVES_A_RECEIVES",
        }
        if not isinstance(document, dict):
            raise WorkerFault("unsupported_condition", "B2 action mapping root must be an object", fatal=True)
        mismatches = [key for key, expected_value in expected.items() if document.get(key) != expected_value]
        if mismatches:
            raise WorkerFault(
                "unsupported_condition",
                "B2 action mapping file does not match the worker's frozen contract",
                fatal=True,
                details={"mismatched_fields": mismatches},
            )
        self._action_mapping_hash = _sha256_file(path)

    def _verify_checkout(self) -> dict[str, Any]:
        def git(*command: str) -> str:
            completed = subprocess.run(
                ["git", "-C", str(self.coachai_root), *command],
                check=True, capture_output=True, text=True,
            )
            return completed.stdout.strip()

        if not (self.coachai_root / ".git").exists():
            if not (self.coachai_root / ".git").is_file():
                raise FileNotFoundError(f"CoachAI+ Git metadata missing: {self.coachai_root}")
        head = git("rev-parse", "HEAD")
        if head != self.expected_commit or head != EXPECTED_COACHAI_COMMIT:
            raise WorkerFault(
                "unsupported_condition",
                f"CoachAI+ commit mismatch: expected {self.expected_commit}, got {head}",
                fatal=True,
            )
        dirty = git("status", "--porcelain")
        if dirty:
            raise WorkerFault(
                "unsupported_condition",
                "CoachAI+ working tree must be clean; preserve/move generated artifacts outside the checkout",
                fatal=True,
                details={"git_status_porcelain": dirty[:4000]},
            )
        files: dict[str, dict[str, Any]] = {}
        for relative, expected_hash in EXPECTED_FILE_SHA256.items():
            path = self.coachai_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required CoachAI+ runtime file missing: {path}")
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise WorkerFault(
                    "unsupported_condition",
                    f"CoachAI+ runtime file hash mismatch: {relative}",
                    fatal=True,
                    details={"expected_sha256": expected_hash, "actual_sha256": actual_hash},
                )
            files[relative] = {"sha256": actual_hash, "bytes": path.stat().st_size}
        constraint_path = self.coachai_root / "BadmintonEnv" / "Utils" / "Constraint.py"
        constraint_hash = _sha256_file(constraint_path)
        if constraint_hash != EXPECTED_CONSTRAINT_SHA256:
            raise WorkerFault(
                "unsupported_condition", "Pinned Constraint.py fingerprint mismatch", fatal=True,
                details={"expected_sha256": EXPECTED_CONSTRAINT_SHA256, "actual_sha256": constraint_hash},
            )
        files["BadmintonEnv/Utils/Constraint.py"] = {
            "sha256": constraint_hash, "bytes": constraint_path.stat().st_size,
        }
        return {
            "root": str(self.coachai_root),
            "head": head,
            "expected_commit": self.expected_commit,
            "working_tree_clean": True,
            "files": files,
        }

    def _add_coachai_import_paths(self) -> None:
        root = str(self.coachai_root)
        env_root = str(self.coachai_root / "BadmintonEnv")
        # Keep these paths worker-local; the parent never imports CoachAI+.
        for item in (env_root, root):
            if item in sys.path:
                sys.path.remove(item)
            sys.path.insert(0, item)
        self._original_cwd = Path.cwd()
        os.chdir(self.coachai_root)

    def _verify_import_origins(self) -> None:
        # Module file paths are checked directly from the imported module objects.
        env_module = sys.modules.get("BadmintonEnv.Environment")
        constraint_module = sys.modules.get("Utils.Constraint")
        if env_module is None or constraint_module is None:
            raise WorkerFault("native_state_error", "Could not resolve CoachAI+ imported modules", fatal=True)
        expected_env = (self.coachai_root / "BadmintonEnv" / "Environment.py").resolve()
        expected_constraint = (self.coachai_root / "BadmintonEnv" / "Utils" / "Constraint.py").resolve()
        if Path(env_module.__file__).resolve() != expected_env:
            raise WorkerFault("native_state_error", "Imported Environment.py does not belong to pinned checkout", fatal=True)
        if Path(constraint_module.__file__).resolve() != expected_constraint:
            raise WorkerFault("native_state_error", "Imported Constraint.py does not belong to pinned checkout", fatal=True)
        self.constraint_module = constraint_module

    def _select_device(self) -> None:
        if self.device_request == "auto":
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        elif self.device_request == "cpu":
            self.device = self.torch.device("cpu")
        elif self.device_request == "cuda":
            if not self.torch.cuda.is_available():
                raise WorkerFault("unsupported_condition", "CUDA requested but unavailable in worker", fatal=True)
            self.device = self.torch.device("cuda")
        else:
            raise WorkerFault("unsupported_condition", f"Unsupported device {self.device_request!r}", fatal=True)

    def _seed_everything(self, seed: int) -> None:
        random.seed(int(seed))
        self.np.random.seed(int(seed) % (2**32))
        self.torch.manual_seed(int(seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seed))
        if hasattr(self.torch.backends, "cudnn"):
            self.torch.backends.cudnn.deterministic = True
            self.torch.backends.cudnn.benchmark = False

    def _install_cpu_torch_load_default(self) -> None:
        if str(self.device).startswith("cpu"):
            original_load = self.torch.load

            def cpu_load(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("map_location", "cpu")
                return original_load(*args, **kwargs)

            self.torch.load = cpu_load

    def _load_rallynet_opponent(self) -> None:
        import pickle

        with (self.coachai_root / "input_data" / "target_players_ids.pkl").open("rb") as stream:
            target_players = pickle.load(stream)
        with (self.coachai_root / "BadmintonEnv" / "Agent" / "hyperparameters.pkl").open("rb") as stream:
            hyperparameters = pickle.load(stream)
        if self.opponent_name not in target_players:
            raise WorkerFault(
                "unsupported_condition",
                f"Opponent {self.opponent_name!r} is not in CoachAI target_players_ids.pkl",
                fatal=True,
            )
        rallynet_module = importlib.import_module("BadmintonEnv.Agent.RallyNet")
        RallyNet = rallynet_module.RallyNet
        player_id = target_players.index(self.opponent_name)
        self.rallynet = RallyNet(
            data_size=hyperparameters["data_size"],
            latent_size=hyperparameters["latent_size"],
            context_size=hyperparameters["context_size"],
            hidden_size=hyperparameters["hidden_size"],
            player_ids_len=hyperparameters["player_ids_len"],
            target_players=hyperparameters["target_players"],
            shot_type_len=hyperparameters["shot_type_len"],
            ts=hyperparameters["ts"],
            id=player_id,
            device=self.device,
        ).to(self.device)
        checkpoint = self.coachai_root / "BadmintonEnv" / "Agent" / "RallyNet_weight.trc"
        weights = self.torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.rallynet.load_state_dict(weights)
        self.rallynet.eval()
        self._rallynet_hash = _sha256_file(checkpoint)
        self._constraint_hash = _sha256_file(self.coachai_root / "BadmintonEnv" / "Utils" / "Constraint.py")
        self._dataset_hash = _sha256_file(self.coachai_root / "input_data" / "All_dataset.csv")

    def _construct_native_environment(self) -> None:
        self._native_temp = tempfile.TemporaryDirectory(prefix="b2_coachai_native_")
        self._native_output_path = Path(self._native_temp.name) / "native_output.csv"
        env_module = importlib.import_module("BadmintonEnv.Environment")
        self.env = env_module.BadmintonEnv(
            ExternalControlledAgent(),
            self.rallynet,
            self.player_name,
            self.opponent_name,
            1,
            is_match=True,
            is_constraint=True,
            is_have_serve_state=False,
            filepath=str(self._native_output_path),
        )

    def _initialize_action_geometry(self) -> None:
        self._landing_points = [_metres_to_native(_landing_centre_m(i)) for i in range(N_LANDING)]
        self._recovery_points = [_metres_to_native(_recovery_centre_m(i)) for i in range(N_RECOVERY)]
        self._landing_zones = [
            int(self.constraint.coordinate_to_block(point, "opp")) for point in self._landing_points
        ]
        self._recovery_zones = [
            int(self.constraint.coordinate_to_block(point, "self")) for point in self._recovery_points
        ]
        if self._landing_zones != EXPECTED_POLICY_LANDING_ZONES:
            raise WorkerFault(
                "unsupported_condition", "Canonical landing grid no longer maps to audited native zones",
                fatal=True,
                details={"actual": self._landing_zones, "expected": EXPECTED_POLICY_LANDING_ZONES},
            )
        if not all(1 <= zone <= 24 for zone in self._landing_zones + self._recovery_zones):
            raise WorkerFault("native_state_error", "Canonical grid maps outside native 24-zone court", fatal=True)
        self._policy_landing_zones = set(self._landing_zones)
        common = set(range(1, 25))
        for native_id in NATIVE_FALLBACK_IDS:
            distribution = self.constraint.action_distribution_2.get(native_id)
            if distribution is None or len(distribution) != 24:
                raise WorkerFault(
                    "native_state_error", f"Missing/invalid native landing distribution for shot {native_id}", fatal=True
                )
            common &= {index + 1 for index, valid in enumerate(distribution) if int(valid) == 1}
        self._common_fallback_zones = sorted(common)
        if self._common_fallback_zones != EXPECTED_COMMON_FALLBACK_ZONES:
            raise WorkerFault(
                "unsupported_condition", "Native empty-vector fallback landing-zone intersection changed",
                fatal=True,
                details={"actual": self._common_fallback_zones, "expected": EXPECTED_COMMON_FALLBACK_ZONES},
            )

    def _verify_native_action_tables(self) -> None:
        for family_id, item in FAMILY_MAP.items():
            model_id = int(item["model_id"])
            native_id = int(item["native_id"])
            if MODEL_TO_NATIVE.get(model_id) != native_id:
                raise WorkerFault(
                    "unsupported_condition", f"Action-map bridge mismatch for family {item['family']}", fatal=True
                )
            if self.env.type_mapping.get(native_id) != item["native_name"]:
                raise WorkerFault(
                    "unsupported_condition", f"Native shot name mismatch for {item['family']}", fatal=True
                )
        if self.env.is_constraint is not True:
            raise WorkerFault("native_state_error", "CoachAI native constraints must be enabled", fatal=True)

    # ---------- Protocol operations ----------
    def handle(self, op: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed and op != "health":
            raise WorkerFault("worker_not_ready", "Worker has been closed", fatal=True)
        if op in {"health", "close"} and payload:
            raise WorkerFault("protocol_violation", f"{op} payload must be empty")
        if op == "health":
            return self.health_payload()
        if op == "reset":
            allowed = {"seed", "episode_index", "episode_id", "options"}
            unexpected = set(payload) - allowed
            if unexpected:
                raise WorkerFault("protocol_violation", f"Unexpected reset payload fields: {sorted(unexpected)}")
            episode_id = payload.get("episode_id")
            if episode_id is not None and (not isinstance(episode_id, str) or not episode_id.strip()):
                raise WorkerFault("protocol_violation", "episode_id must be a non-empty string when supplied")
            return self.reset(payload)
        if op == "step":
            if set(payload) != {"action_id"}:
                raise WorkerFault("protocol_violation", "step payload must contain exactly action_id")
            return self.step(payload)
        if op == "close":
            self.close()
            return {"closed": True}
        raise WorkerFault("protocol_violation", f"Unknown operation {op!r}")

    def health_payload(self) -> dict[str, Any]:
        if self.env is None or self.rallynet is None or self.constraint is None:
            raise WorkerFault("worker_not_ready", "Native runtime has not initialized", fatal=True)
        return {
            "ready": True,
            "mode": "B2",
            "python_version": sys.version.split()[0],
            "coachai_commit": self.checkout_fingerprint["head"],
            "coachai_root": str(self.coachai_root),
            "rallynet_checkpoint_sha256": self._rallynet_hash,
            "constraint_source_sha256": self._constraint_hash,
            "constraint_data_sha256": self._dataset_hash,
            "action_map_version": ACTION_MAP_VERSION,
            "action_mapping_config_sha256": self._action_mapping_hash,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "device": str(self.device),
            "opponent": self.opponent_name,
            "controlled_side": "A",
            "opponent_side": "B",
            "service_policy": "B_SERVES_A_RECEIVES",
            "action_count": N_ACTIONS,
            "supported_families": [item["family"] for _, item in sorted(FAMILY_MAP.items())],
            "unsupported_families": [UNSUPPORTED_FAMILIES[key] for key in sorted(UNSUPPORTED_FAMILIES)],
            "native_output_path_is_temporary": True,
        }

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise WorkerFault("protocol_violation", "reset payload requires integer seed")
        seed = int(seed)
        options = payload.get("options", {})
        if not isinstance(options, dict):
            raise WorkerFault("protocol_violation", "reset options must be an object")
        allowed_options = {
            "server",
            "score_a",
            "score_b",
            "round",
            "sample_score",
            "max_reset_retries",
        }
        unexpected = set(options) - allowed_options
        if unexpected:
            raise WorkerFault("unsupported_condition", f"Unsupported reset options: {sorted(unexpected)}")
        server = options.get("server", "B")
        if server != "B":
            raise WorkerFault("unsupported_condition", "B2 v1 requires B to serve and A to receive")
        episode_index = payload.get("episode_index")
        if episode_index is None:
            episode_index = self._episode_index + 1
        if isinstance(episode_index, bool) or not isinstance(episode_index, numbers.Integral) or episode_index < 0:
            raise WorkerFault("protocol_violation", "episode_index must be a non-negative integer")
        self._episode_index = int(episode_index)
        self._episode_seed = int(seed)
        self._episode_id = str(payload.get("episode_id") or f"b2-episode-{self._episode_index:06d}")
        max_retries = options.get("max_reset_retries", self.max_reset_retries)
        if isinstance(max_retries, bool) or not isinstance(max_retries, numbers.Integral) or int(max_retries) < 0:
            raise WorkerFault("protocol_violation", "max_reset_retries must be a non-negative integer")
        max_retries = int(max_retries)

        score_rng = random.Random(self._episode_seed)
        if bool(options.get("sample_score", False)):
            init_score_a = score_rng.randint(0, 20)
            init_score_b = score_rng.randint(0, 20)
        else:
            init_score_a = options.get("score_a", 0)
            init_score_b = options.get("score_b", 0)
        for name, val in (("score_a", init_score_a), ("score_b", init_score_b)):
            if isinstance(val, bool) or not isinstance(val, numbers.Integral) or not 0 <= int(val) <= 20:
                raise WorkerFault("protocol_violation", f"{name} must be an integer in [0, 20]")
        init_score_a, init_score_b = int(init_score_a), int(init_score_b)
        init_round = options.get("round", init_score_a + init_score_b + 1)
        if isinstance(init_round, bool) or not isinstance(init_round, numbers.Integral) or int(init_round) < 1:
            raise WorkerFault("protocol_violation", "round must be a positive integer")
        init_round = int(init_round)

        last_payload: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            attempt_seed = derive_reset_retry_seed(self._episode_seed, attempt)
            self._policy_decisions = 0
            self._fallback_counter = 0
            self._terminated = False
            self._truncated = False
            self._ready = False
            self._mask = None
            self._mask_evaluation = None
            self._seed_everything(attempt_seed)

            # Truncate previous episode rows from env.output so memory and step
            # slice latency stay O(rally_length) rather than growing O(N^2).
            self._clear_native_output()
            self._begin_event_attempt(attempt)

            # CoachAI+ reset increments rally and initializes RallyNet's new history.
            # Resource objects (RallyNet, constraint tables, BaseServe, output frame)
            # remain persistent across episodes; only rally-scoped state is cleared.
            self.env.score_a = init_score_a
            self.env.score_b = init_score_b
            self.env.pre_score_a = init_score_a
            self.env.pre_score_b = init_score_b
            self.env.round = init_round
            self.env.match = 1
            self.env.state = None
            self.env.info_dict = {}
            self.env.serving_player = "B"
            self._score_baseline = {"A": init_score_a, "B": init_score_b}
            output_rows_before = int(len(self.env.output))
            reset_result = self.env.reset()
            if not isinstance(reset_result, (tuple, list)) or len(reset_result) != 4:
                raise WorkerFault(
                    "native_state_error", "CoachAI reset must return (state, info, done, launch)", fatal=True,
                    details={"return_type": type(reset_result).__name__, "return_length": len(reset_result) if isinstance(reset_result, (tuple, list)) else None},
                )
            state, info, done, launch = reset_result
            self._state, self._info, self._launch = state, info, bool(launch)
            if self._launch:
                raise WorkerFault("unsupported_condition", "B-serve reset unexpectedly left A in launch/service state", fatal=True)
            self._record_transition(
                source=SOURCE_SERVE, decision_index=0,
                output_start=output_rows_before, output_end=int(len(self.env.output)),
            )
            if done:
                self._terminated = True
                terminal = self._resolve_terminal(fallback_occurred=False)
                self._ready = False
                self._mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
                # Only the attempt that is actually returned contributes policy-visible events.
                events = self._collect_rally_events(terminal=True) if attempt >= max_retries else []
                last_payload = self._state_payload(
                    status="terminal_without_policy", reset_return_length=4,
                    reset_output_rows_added=int(len(self.env.output)) - output_rows_before,
                    internal_native_transitions=0, native_reward_trace=[], fallback_events=[],
                    terminal=terminal, no_policy_rally=True,
                    rally_events_delta=events,
                )
                if attempt < max_retries:
                    continue
                return last_payload

            # Empty masks are resolved internally before any state is exposed to PPO.
            fallback_events: list[dict[str, Any]] = []
            native_rewards: list[float] = []
            transitions = self._advance_to_policy_state(
                fallback_events=fallback_events,
                native_rewards=native_rewards,
                decision_index=1,
            )
            terminal = self._resolve_terminal(fallback_occurred=bool(fallback_events)) if self._terminated else None
            status = "ready_for_policy" if self._ready else "terminal_without_policy"
            # Only the attempt that is actually returned contributes policy-visible events.
            events = (
                self._collect_rally_events(terminal=self._terminated)
                if (self._ready or attempt >= max_retries) else []
            )
            last_payload = self._state_payload(
                status=status,
                reset_return_length=4,
                reset_output_rows_added=int(len(self.env.output)) - output_rows_before,
                internal_native_transitions=transitions,
                native_reward_trace=native_rewards,
                fallback_events=fallback_events,
                terminal=terminal,
                no_policy_rally=bool(self._terminated and self._policy_decisions == 0),
                rally_events_delta=events,
            )
            if self._ready or attempt >= max_retries:
                return last_payload
        assert last_payload is not None
        return last_payload

    def step(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._episode_seed is None or self._state is None:
            raise WorkerFault("worker_not_ready", "step called before reset")
        if self._terminated or self._truncated:
            raise WorkerFault("worker_not_ready", "step called after terminal/truncated state")
        if not self._ready or self._mask is None or not bool(self._mask.any()):
            raise WorkerFault("worker_not_ready", "worker is not waiting at a policy decision state")
        action_id = payload.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, numbers.Integral) or not 0 <= int(action_id) < N_ACTIONS:
            raise WorkerFault("invalid_action_id", f"action_id must be an integer in [0,{N_ACTIONS})")
        action_id = int(action_id)
        if not bool(self._mask[action_id]):
            raise WorkerFault("masked_action_selected", f"action_id {action_id} is false in the current mask")

        decision_index = self._policy_decisions + 1
        native_action = self._build_policy_action(action_id, self._state, decision_index=decision_index)
        output_rows_before = int(len(self.env.output))
        try:
            transition = self._execute_a_transition(native_action, self._launch, fallback_reason=None, event_rng=None)
        except WorkerFault:
            raise
        except Exception as exc:
            raise WorkerFault("native_state_error", f"CoachAI native step failed: {exc}", fatal=True) from exc
        self._policy_decisions += 1
        self._record_transition(
            source=SOURCE_POLICY, decision_index=decision_index,
            output_start=output_rows_before, output_end=int(len(self.env.output)),
            policy_action_id=action_id,
        )
        native_rewards = [transition.native_reward]
        fallback_events: list[dict[str, Any]] = []
        internal_transitions = 1
        self._state, self._info, self._launch = transition.next_state, transition.info, transition.launch
        if transition.done:
            self._terminated = True
            terminal = self._resolve_terminal(fallback_occurred=False)
            self._ready = False
            self._mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
            events = self._collect_rally_events(terminal=True)
            return self._transition_payload(
                action_id=action_id,
                observation=self._observation(),
                reward=terminal["b2_rally_reward"],
                terminated=True,
                truncated=False,
                action_mask=self._mask,
                native_rewards=native_rewards,
                fallback_events=fallback_events,
                internal_native_transitions=internal_transitions,
                output_rows_added=int(len(self.env.output)) - output_rows_before,
                terminal=terminal,
                rally_events_delta=events,
                normal_action={
                    "policy_action_id": action_id,
                    "canonical_action": list(decode_action(action_id)),
                    "native_validator_corrected": bool(transition.validator_corrected),
                    "native_output_record_matches": bool(transition.output_record_matches),
                },
            )

        extra_transitions = self._advance_to_policy_state(
            fallback_events=fallback_events,
            native_rewards=native_rewards,
            decision_index=self._policy_decisions + 1,
        )
        internal_transitions += extra_transitions
        terminal = self._resolve_terminal(fallback_occurred=bool(fallback_events)) if self._terminated else None

        # A policy-decision limit is a safety watchdog, not a badminton rule.
        # Resolve the complete decision epoch (including automatic fallbacks)
        # first. A natural rally terminal on the boundary is valid; otherwise
        # fail closed instead of returning a neutral time-limit truncation that
        # a policy could exploit by prolonging the rally.
        if terminal is None and self._policy_decisions >= self.max_decisions:
            watchdog_event = {
                "event_type": "b2_policy_decision_watchdog_exceeded",
                "episode_id": self._episode_id,
                "episode_index": self._episode_index,
                "episode_seed": self._episode_seed,
                "decision_index": decision_index,
                "policy_action_id": int(action_id),
                "policy_selected": True,
                "policy_decisions": int(self._policy_decisions),
                "max_decisions": int(self.max_decisions),
                "internal_native_transitions": int(internal_transitions),
                "fallback_event_ids": [
                    item.get("fallback_event_id")
                    for item in fallback_events
                    if isinstance(item, Mapping)
                ],
                "score": {"A": int(self.env.score_a), "B": int(self.env.score_b)},
                "terminal": False,
            }
            self._append_event(watchdog_event)
            raise WorkerFault(
                "policy_decision_limit_exceeded",
                "B2 policy-decision watchdog reached before the rally was naturally resolved; "
                "the episode and training run must be aborted, not converted to truncation.",
                fatal=True,
                details=watchdog_event,
            )

        reward = float(terminal["b2_rally_reward"]) if terminal else 0.0
        events = self._collect_rally_events(terminal=self._terminated)
        return self._transition_payload(
            action_id=action_id,
            observation=self._observation(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            action_mask=self._mask,
            native_rewards=native_rewards,
            fallback_events=fallback_events,
            internal_native_transitions=internal_transitions,
            output_rows_added=int(len(self.env.output)) - output_rows_before,
            terminal=terminal,
            rally_events_delta=events,
            normal_action={
                "policy_action_id": action_id,
                "canonical_action": list(decode_action(action_id)),
                "native_validator_corrected": bool(transition.validator_corrected),
                "native_output_record_matches": bool(transition.output_record_matches),
            },
        )

    def _advance_to_policy_state(
        self, *, fallback_events: list[dict[str, Any]], native_rewards: list[float], decision_index: int,
    ) -> int:
        """Auto-execute empty-mask fallbacks until policy-ready or rally terminal."""
        transitions = 0
        while True:
            evaluation = self._evaluate_mask(self._state)
            self._mask_evaluation = evaluation
            mask = evaluation["mask"]
            if bool(mask.any()):
                self._mask = mask
                self._ready = True
                return transitions

            reason = evaluation.get("empty_reason")
            if reason == "CONTACT_OUT_OF_ZONE":
                raise WorkerFault(
                    "native_state_error",
                    "Reachable contact point does not map to CoachAI's 24 native zones",
                    fatal=True,
                    details={key: value for key, value in evaluation.items() if key != "mask"},
                )
            if reason == "CONTACT_UNREACHABLE":
                # Native A miss/fault path resolves the rally. It is not an invented
                # legal macro-action and is logged independently from policy actions.
                transition = self._execute_a_transition(
                    None, self._launch, fallback_reason=None, event_rng=None
                )
                native_rewards.append(transition.native_reward)
                transitions += 1
                self._state, self._info, self._launch = transition.next_state, transition.info, transition.launch
                self._record_transition(
                    source=SOURCE_UNPLAYABLE, decision_index=decision_index - 1,
                    output_start=transition.output_start, output_end=transition.output_end,
                )
                event = {
                    "event_type": "b2_unplayable_state",
                    "episode_id": self._episode_id,
                    "episode_index": self._episode_index,
                    "episode_seed": self._episode_seed,
                    "decision_index": decision_index,
                    "reason": "CONTACT_UNREACHABLE",
                    "policy_action_id": None,
                    "policy_selected": False,
                    "native_done": bool(transition.done),
                    "native_step_reward": transition.native_reward,
                }
                self._append_event(event)
                if transition.done:
                    self._terminated = True
                    self._ready = False
                    self._mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
                    return transitions
                if transitions >= self.max_internal_transitions:
                    raise WorkerFault("native_state_error", "Too many internally advanced unplayable states", fatal=True)
                continue

            if reason not in {REASON_NATIVE_EMPTY, REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY}:
                raise WorkerFault(
                    "native_state_error", f"Unexpected empty-mask classification: {reason!r}", fatal=True,
                    details={key: value for key, value in evaluation.items() if key != "mask"},
                )
            if transitions >= self.max_internal_transitions:
                raise WorkerFault(
                    "native_state_error", "Exceeded max_internal_transitions while resolving empty masks", fatal=True
                )
            event, transition = self._execute_fallback(reason, evaluation, decision_index)
            self._record_transition(
                source=SOURCE_FALLBACK, decision_index=decision_index - 1,
                output_start=transition.output_start, output_end=transition.output_end,
                fallback_event_id=event["fallback_event_id"],
            )
            fallback_events.append(event)
            self._append_event(event)
            native_rewards.append(transition.native_reward)
            transitions += 1
            self._state, self._info, self._launch = transition.next_state, transition.info, transition.launch
            if transition.done:
                self._terminated = True
                self._ready = False
                self._mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
                return transitions

    # ---------- Action mask / construction ----------
    def _evaluate_mask(self, state: Any) -> dict[str, Any]:
        if not isinstance(state, (tuple, list)) or len(state) != 4:
            raise WorkerFault("native_state_error", "CoachAI state must be a four-field tuple", fatal=True)
        try:
            hit = (float(state[3][0]), float(state[3][1]))
            player = (float(state[1][0]), float(state[1][1]))
            hit_zone = int(self.constraint.coordinate_to_block(hit, "self"))
        except Exception as exc:
            raise WorkerFault("native_state_error", f"Could not parse CoachAI state coordinates: {exc}", fatal=True) from exc
        if not all(math.isfinite(value) for value in (*hit, *player)):
            raise WorkerFault("native_state_error", "CoachAI state coordinates contain non-finite values", fatal=True)
        movement_bin = self._movement_bin(state)
        mask = self.np.zeros((N_FAMILIES, N_LANDING, N_RECOVERY), dtype=self.np.bool_)
        contact_candidate = _build_candidate(3, hit, (0.0, 100.0), (0.0, -100.0))
        contact_violation = self.env.check_out_violation(contact_candidate, state, False)
        contact_reachable = contact_violation is None
        if not 1 <= hit_zone <= 24:
            reason = "CONTACT_OUT_OF_ZONE"
            allowed_ids: list[int] = []
        elif not contact_reachable:
            reason = "CONTACT_UNREACHABLE"
            allowed_ids = []
        else:
            previous_native_id = int(state[0])
            distribution = self.constraint.action_distribution_1.get(previous_native_id)
            if distribution is None:
                raise WorkerFault(
                    "native_state_error", f"Missing shot constraint distribution for previous native ID {previous_native_id}", fatal=True
                )
            shot_vector = self.np.asarray(distribution[hit_zone - 1][movement_bin], dtype=self.np.int8)
            if shot_vector.shape != (10,):
                raise WorkerFault("native_state_error", f"Unexpected native shot-vector shape {shot_vector.shape}", fatal=True)
            if self.np.any((shot_vector != 0) & (shot_vector != 1)):
                raise WorkerFault("native_state_error", "Native shot constraint vector is not Boolean", fatal=True)
            allowed_ids = [int(index) + 1 for index in self.np.flatnonzero(shot_vector)]
            legal_recoveries: list[int] = []
            for recovery_id, recovery in enumerate(self._recovery_points):
                corrected = self.constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
                fixed = self.np.allclose(
                    self.np.asarray(corrected, dtype=self.np.float64),
                    self.np.asarray(recovery, dtype=self.np.float64),
                    rtol=0.0, atol=1e-8,
                )
                if fixed:
                    legal_recoveries.append(recovery_id)
            for family_id, item in FAMILY_MAP.items():
                native_id = int(item["native_id"])
                if not bool(shot_vector[native_id - 1]):
                    continue
                landing_distribution = self.constraint.action_distribution_2.get(native_id)
                if landing_distribution is None or len(landing_distribution) != 24:
                    raise WorkerFault("native_state_error", f"Invalid landing distribution for native shot {native_id}", fatal=True)
                legal_landings = [
                    landing_id for landing_id, zone in enumerate(self._landing_zones)
                    if int(landing_distribution[zone - 1]) == 1
                ]
                for landing_id in legal_landings:
                    landing = self._landing_points[landing_id]
                    for recovery_id in legal_recoveries:
                        recovery = self._recovery_points[recovery_id]
                        candidate = _build_candidate(int(item["model_id"]), hit, landing, recovery)
                        violation = self.env.check_out_violation(candidate, state, False)
                        if violation is None:
                            mask[family_id, landing_id, recovery_id] = True
            reason = None if bool(mask.any()) else classify_empty_mask_reason(
                contact_reachable=contact_reachable,
                contact_zone_valid=True,
                native_allowed_ids=allowed_ids,
                mapped_native_ids={int(item["native_id"]) for item in FAMILY_MAP.values()},
            )

        flat_mask = mask.reshape(N_ACTIONS)
        return {
            "mask": flat_mask,
            "mask_shape": [N_ACTIONS],
            "valid_action_count": int(flat_mask.sum()),
            "mask_sha256": hashlib.sha256(flat_mask.astype(self.np.uint8).tobytes()).hexdigest(),
            "hit_xy_native": list(hit),
            "hit_zone": hit_zone,
            "movement_bin": movement_bin,
            "contact_reachable": bool(contact_reachable),
            "contact_violation": contact_violation,
            "previous_native_shot_id": int(state[0]),
            "native_allowed_shot_ids": allowed_ids,
            "empty_reason": reason,
        }

    @staticmethod
    def _movement_bin(state: Any) -> int:
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

    def _build_policy_action(self, action_id: int, state: Any, *, decision_index: int = 1) -> tuple[Any, ...]:
        family_id, landing_id, recovery_id = decode_action(action_id)
        item = FAMILY_MAP.get(family_id)
        if item is None:
            name = UNSUPPORTED_FAMILIES.get(family_id, f"family_{family_id}")
            raise WorkerFault("invalid_action_id", f"Canonical family {name} is unsupported in B2 v1")
        hit = (float(state[3][0]), float(state[3][1]))
        landing = self._landing_points[landing_id]
        recovery = self._recovery_points[recovery_id]
        model_id = int(item["model_id"])

        if self.cell_jitter_fraction > 0.0 and self._episode_seed is not None:
            jitter_seed = derive_action_jitter_seed(
                self.worker_seed, self._episode_seed, self._episode_index, decision_index, action_id
            )
            jrng = random.Random(jitter_seed)
            scale = min(max(self.cell_jitter_fraction, 0.0), 0.45)
            dx_l = (2.0 * NATIVE_HALF_WIDTH / 3.0) * scale
            dy_l = (NATIVE_HALF_LENGTH / 5.0) * scale
            dx_r = (2.0 * NATIVE_HALF_WIDTH / 3.0) * scale
            dy_r = (NATIVE_HALF_LENGTH / 3.0) * scale
            j_landing = (
                float(landing[0] + jrng.uniform(-dx_l, dx_l)),
                float(landing[1] + jrng.uniform(-dy_l, dy_l)),
            )
            j_recovery = (
                float(recovery[0] + jrng.uniform(-dx_r, dx_r)),
                float(recovery[1] + jrng.uniform(-dy_r, dy_r)),
            )
            corrected = self.constraint.correct_position(hit[0], hit[1], j_recovery[0], j_recovery[1])
            if (
                int(self.constraint.coordinate_to_block(j_landing, "opp")) == self._landing_zones[landing_id]
                and int(self.constraint.coordinate_to_block(j_recovery, "self")) == self._recovery_zones[recovery_id]
                and self.np.allclose(corrected, j_recovery, rtol=0.0, atol=1e-8)
            ):
                candidate = _build_candidate(model_id, hit, j_landing, j_recovery)
                if self.env.check_out_violation(candidate, state, False) is None:
                    return candidate

        return _build_candidate(model_id, hit, landing, recovery)

    def _native_context(self, state: Any, evaluation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "previous_native_shot_id": int(state[0]),
            "hit_zone": int(evaluation["hit_zone"]),
            "y_movement_bin": int(evaluation["movement_bin"]),
            "native_allowed_shot_ids": list(evaluation["native_allowed_shot_ids"]),
        }

    def _native_zone_centre(self, zone_id: int, *, opponent_frame: bool) -> tuple[float, float]:
        area = next((entry for entry in self.constraint.block_area if int(entry[2]) == int(zone_id)), None)
        if area is None:
            raise WorkerFault("native_state_error", f"Native block zone {zone_id} is missing", fatal=True)
        centre_x, centre_y = float(area[3][0]), float(area[3][1])
        if opponent_frame:
            point = (177.5 - centre_x, 480.0 - centre_y)
            actual = int(self.constraint.coordinate_to_block(point, "opp"))
        else:
            point = (centre_x - 177.5, centre_y - 480.0)
            actual = int(self.constraint.coordinate_to_block(point, "self"))
        if actual != int(zone_id):
            raise WorkerFault("native_state_error", "Native block-centre transform failed round-trip", fatal=True)
        return float(point[0]), float(point[1])

    def _build_fallback_carrier(
        self, reason: str, evaluation: Mapping[str, Any], rng: random.Random,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        state = self._state
        context = self._native_context(state, evaluation)
        allowed = set(context["native_allowed_shot_ids"])
        mapped_native = {int(item["native_id"]) for item in FAMILY_MAP.values()}
        if reason == REASON_NATIVE_EMPTY:
            if allowed:
                raise WorkerFault("native_state_error", "Native-empty fallback selected for a non-empty native set", fatal=True)
            if self._common_fallback_zones != EXPECTED_COMMON_FALLBACK_ZONES:
                raise WorkerFault("native_state_error", "Native empty-vector carrier zones failed startup audit", fatal=True)
            model_id = 3  # Model CLEAR -> native Clear; pinned validator will sample native ID 2..9.
            native_landing_zone = int(rng.choice(self._common_fallback_zones))
            allowed_for_selection = list(NATIVE_FALLBACK_IDS)
            selection_method = "pinned_native_empty_vector_random_choice_2_through_9"
        elif reason == REASON_TAXONOMY_GAP:
            if not allowed or (allowed & mapped_native):
                raise WorkerFault("native_state_error", "Taxonomy-gap fallback context is inconsistent", fatal=True)
            allowed_for_selection = sorted((allowed - mapped_native) & set(NATIVE_FALLBACK_IDS))
            if not allowed_for_selection:
                raise WorkerFault("native_state_error", "No supported native ID for B2 taxonomy-gap fallback", fatal=True)
            native_id = int(rng.choice(allowed_for_selection))
            model_id = NATIVE_TO_MODEL.get(native_id)
            if model_id is None:
                raise WorkerFault("native_state_error", f"No inverse native/model mapping for native ID {native_id}", fatal=True)
            valid_landing = [
                index + 1 for index, valid in enumerate(self.constraint.action_distribution_2[native_id])
                if int(valid) == 1
            ]
            if not valid_landing:
                raise WorkerFault("native_state_error", f"Native shot {native_id} has no valid landing zone", fatal=True)
            native_landing_zone = int(rng.choice(valid_landing))
            selection_method = "worker_rng_choice_from_native_allowed_unmapped_ids"
        elif reason == REASON_GEOMETRY_EMPTY:
            allowed_for_selection = sorted(allowed & mapped_native)
            if not allowed_for_selection:
                raise WorkerFault("native_state_error", "Geometry fallback has no allowed mapped native shot", fatal=True)
            native_id = int(rng.choice(allowed_for_selection))
            model_id = NATIVE_TO_MODEL[native_id]
            # Prefer landing zones within the canonical policy grid first so the
            # fallback does not unnecessarily introduce out-of-vocabulary court zones.
            valid_landing = [
                index + 1 for index, valid in enumerate(self.constraint.action_distribution_2[native_id])
                if int(valid) == 1 and (index + 1) in self._policy_landing_zones
            ]
            if not valid_landing:
                valid_landing = [
                    index + 1 for index, valid in enumerate(self.constraint.action_distribution_2[native_id])
                    if int(valid) == 1
                ]
            if not valid_landing:
                raise WorkerFault("native_state_error", f"Native shot {native_id} has no valid fallback landing zone", fatal=True)
            native_landing_zone = int(rng.choice(valid_landing))
            selection_method = "worker_rng_choice_from_native_allowed_mapped_ids"
        else:
            raise WorkerFault("native_state_error", f"Unknown fallback reason {reason!r}", fatal=True)

        landing = self._native_zone_centre(native_landing_zone, opponent_frame=True)
        hit = (float(state[3][0]), float(state[3][1]))
        player = (float(state[1][0]), float(state[1][1]))
        recovery = tuple(float(value) for value in self.constraint.correct_position(hit[0], hit[1], player[0], player[1]))
        projected = self.constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
        if not self.np.allclose(projected, recovery, rtol=0.0, atol=1e-8):
            recovery = tuple(float(value) for value in projected)
            projected = self.constraint.correct_position(hit[0], hit[1], recovery[0], recovery[1])
        if not self.np.allclose(projected, recovery, rtol=0.0, atol=1e-8):
            raise WorkerFault("native_state_error", "Fallback recovery is not a fixed point of correct_position", fatal=True)
        if int(self.constraint.coordinate_to_block(landing, "opp")) != native_landing_zone:
            raise WorkerFault("native_state_error", "Fallback landing does not round-trip to native zone", fatal=True)
        if reason == REASON_NATIVE_EMPTY:
            for native_id in NATIVE_FALLBACK_IDS:
                if int(self.constraint.action_distribution_2[native_id][native_landing_zone - 1]) != 1:
                    raise WorkerFault("native_state_error", "Fallback carrier landing is not common to native IDs 2..9", fatal=True)
        else:
            carrier_native_id = MODEL_TO_NATIVE[model_id]
            if int(self.constraint.action_distribution_2[carrier_native_id][native_landing_zone - 1]) != 1:
                raise WorkerFault("native_state_error", "Fallback carrier landing is invalid for its selected native shot", fatal=True)
        carrier = _build_candidate(model_id, hit, landing, recovery)
        violation = self.env.check_out_violation(carrier, state, False)
        if violation is not None:
            raise WorkerFault("native_state_error", f"Fallback carrier failed native precheck: {violation}", fatal=True)
        return carrier, {
            "reason": reason,
            "native_context": context,
            "allowed_ids_for_selection": allowed_for_selection,
            "selection_method": selection_method,
            "fallback_input_model_shot_id": int(model_id),
            "carrier_native_landing_zone_id": int(native_landing_zone),
            "carrier_landing_position_native": list(landing),
            "carrier_recovery_position_native": list(recovery),
            "recovery_is_fixed_point_of_correct_position": True,
            "carrier_passes_environment_precheck": True,
            "carrier_landing_zone_in_policy_grid": native_landing_zone in self._policy_landing_zones,
        }

    # ---------- Native transition and fallback event recording ----------
    def _execute_a_transition(
        self, action: Any, launch: bool, *, fallback_reason: str | None, event_rng: random.Random | None,
    ) -> NativeStepResult:
        state_before = self._state
        score_before = {"A": int(self.env.score_a), "B": int(self.env.score_b)}
        output_rows_before = int(len(self.env.output))
        capture: OneShotValidationCapture | None = None
        try:
            if action is not None:
                capture = OneShotValidationCapture(
                    constraint=self.constraint,
                    constraint_module=self.constraint_module,
                    target_state=state_before,
                    torch_module=self.torch,
                    np_module=self.np,
                    event_rng=event_rng,
                    target_action=action,
                )
                with capture:
                    step_result = self.env.step(action, is_launch=bool(launch))
            else:
                step_result = self.env.step(None, is_launch=bool(launch))
        except Exception:
            # Native code may already have mutated score/history before raising.
            raise
        if not isinstance(step_result, (tuple, list)) or len(step_result) != 5:
            raise WorkerFault(
                "native_state_error", "Normal A-side CoachAI step must return five values", fatal=True,
                details={"return_length": len(step_result) if isinstance(step_result, (tuple, list)) else None},
            )
        next_state, native_reward, next_info, done, next_launch = step_result
        if not math.isfinite(float(native_reward)):
            raise WorkerFault("native_state_error", "CoachAI returned a non-finite native reward", fatal=True)
        checked_action = validator_reward = validator_corrected = output_match = None
        if action is not None:
            if capture is None or capture.matching_calls != 1 or capture.captured_result is None:
                raise WorkerFault(
                    "native_state_error", "Expected exactly one captured A-side validator call", fatal=True,
                    details={"matching_calls": capture.matching_calls if capture else 0},
                )
            checked_action, validator_reward, validator_corrected = capture.captured_result
            self._validate_captured_a_action(
                fallback_reason=fallback_reason,
                proposed_action=action,
                checked_action=checked_action,
                validator_reward=validator_reward,
                validator_corrected=validator_corrected,
            )
            a_row = self._first_new_a_output_row(output_rows_before)
            output_match = self._native_output_matches(a_row, checked_action)
            if not output_match:
                raise WorkerFault("native_state_error", "CoachAI A-side output record disagrees with captured validator action", fatal=True)
        else:
            # Unplayable contact is sent through CoachAI's native miss/fault path;
            # no validator action and no fabricated macro-action exist for this state.
            self._first_new_a_output_row(output_rows_before)
        if bool(next_launch):
            raise WorkerFault("unsupported_condition", "CoachAI returned A to a launch/service decision mid-rally", fatal=True)
        try:
            _json_safe(next_state)
        except Exception as exc:
            raise WorkerFault("native_state_error", f"CoachAI produced unserializable state: {exc}", fatal=True) from exc
        score_after = {"A": int(self.env.score_a), "B": int(self.env.score_b)}
        return NativeStepResult(
            next_state=next_state,
            native_reward=float(native_reward),
            info=next_info,
            done=bool(done),
            launch=bool(next_launch),
            checked_action=checked_action,
            validator_reward=float(validator_reward) if validator_reward is not None else None,
            validator_corrected=bool(validator_corrected) if validator_corrected is not None else None,
            validation_capture=capture,
            output_record_matches=bool(output_match) if output_match is not None else None,
            score_before=score_before,
            score_after=score_after,
            output_start=output_rows_before,
            output_end=int(len(self.env.output)),
        )

    @classmethod
    def _validate_captured_a_action(
        cls, *, fallback_reason: str | None, proposed_action: Any,
        checked_action: Any, validator_reward: Any, validator_corrected: Any,
    ) -> None:
        """Enforce the expected native validator behavior for each action class.

        The empty-native-constraint branch is special: the pinned validator is
        expected to replace the carrier shot with a random native category 2..9.
        The other fallback classes and policy actions must pass through unchanged.
        """
        if fallback_reason == REASON_NATIVE_EMPTY:
            if not bool(validator_corrected):
                raise WorkerFault(
                    "native_state_error",
                    "Native-empty fallback did not enter the pinned empty-vector correction path",
                    fatal=True,
                    details={"validator_reward": float(validator_reward)},
                )
            if not isinstance(checked_action, (tuple, list)) or len(checked_action) != 5:
                raise WorkerFault("native_state_error", "Native-empty validator returned malformed action", fatal=True)
            if int(checked_action[0]) not in NATIVE_FALLBACK_IDS:
                raise WorkerFault(
                    "native_state_error", "Native-empty fallback selected an ID outside native categories 2..9", fatal=True
                )
            if any(float(value) != 0.0 for value in checked_action[4]):
                raise WorkerFault(
                    "native_state_error", "Native-empty fallback did not zero its shot-probability vector", fatal=True
                )
            return
        if not isinstance(proposed_action, (tuple, list)) or len(proposed_action) != 5:
            raise WorkerFault("native_state_error", "Proposed A-side action is malformed", fatal=True)
        model_id = int(proposed_action[0])
        expected_native_id = MODEL_TO_NATIVE.get(model_id)
        if expected_native_id is None:
            raise WorkerFault("native_state_error", f"No native shot mapping for model ID {model_id}", fatal=True)
        # validate_action() always translates the first field from RallyNet's
        # model namespace to CoachAI's native namespace, even when it makes no
        # corrective change. Compare against that translated carrier, not the
        # original model ID.
        expected_native_action = (
            expected_native_id,
            proposed_action[1],
            proposed_action[2],
            proposed_action[3],
            proposed_action[4],
        )
        if bool(validator_corrected) or not cls._actions_equal(checked_action, expected_native_action):
            raise WorkerFault(
                "native_state_error",
                "A-side native validator modified a mask-approved policy/fallback carrier action",
                fatal=True,
                details={"validator_reward": float(validator_reward), "validator_corrected": bool(validator_corrected)},
            )

    @staticmethod
    def _actions_equal(left: Any, right: Any) -> bool:
        # Kept as a staticmethod for API compatibility; the implementation now
        # lives in coachai_worker_utils.actions_equal (shared with the capture
        # module, which may not import this module).
        return actions_equal(left, right)

    def _first_new_a_output_row(self, output_rows_before: int) -> Any:
        new_rows = self.env.output.iloc[output_rows_before:]
        if len(new_rows) == 0:
            raise WorkerFault("native_state_error", "CoachAI did not append an A-side output record", fatal=True)
        a_rows = new_rows[new_rows["player"].astype(str) == "A"]
        if len(a_rows) == 0:
            raise WorkerFault(
                "native_state_error", "No A-side output record found after controlled native transition", fatal=True,
                details={"new_output_sides": [str(value) for value in new_rows["player"].tolist()]},
            )
        return a_rows.iloc[0]

    @staticmethod
    def _native_output_matches(row: Any, native_action: Any) -> bool:
        comparisons = (
            ("type", int(native_action[0])),
            ("player_hit_x", float(native_action[1][0])),
            ("player_hit_y", float(native_action[1][1])),
            ("landing_x", float(native_action[2][0])),
            ("landing_y", float(native_action[2][1])),
            ("player_movement_x", float(native_action[3][0])),
            ("player_movement_y", float(native_action[3][1])),
        )
        for key, expected in comparisons:
            if key not in row.index:
                return False
            actual = row[key]
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-8):
                    return False
            elif int(actual) != expected:
                return False
        return True

    def _execute_fallback(
        self, reason: str, evaluation: Mapping[str, Any], decision_index: int,
    ) -> tuple[dict[str, Any], NativeStepResult]:
        fallback_counter = self._fallback_counter
        self._fallback_counter += 1
        fallback_seed = derive_fallback_seed(
            self.worker_seed, self._episode_index, decision_index, fallback_counter
        )
        event_rng = random.Random(fallback_seed)
        event_id = f"episode-{self._episode_index}-decision-{decision_index}-fallback-{fallback_counter}"
        state_signature = _state_signature(self._state)
        score_before = {"A": int(self.env.score_a), "B": int(self.env.score_b)}
        transition_executed = False
        selection: dict[str, Any] | None = None
        carrier: Any = None
        try:
            carrier, selection = self._build_fallback_carrier(reason, evaluation, event_rng)
            transition = self._execute_a_transition(
                carrier, self._launch, fallback_reason=reason, event_rng=event_rng
            )
            transition_executed = True
            capture = transition.validation_capture
            if capture is None or capture.captured_result is None:
                raise WorkerFault(
                    "native_state_error", "Fallback did not capture the actual native validator result", fatal=True
                )
            checked = capture.captured_result[0]
            after_delta = {
                "A": int(self.env.score_a) - score_before["A"],
                "B": int(self.env.score_b) - score_before["B"],
            }
            transition_reward = self._reward_from_delta(after_delta, terminal=transition.done)
            event = {
                "event_type": "b2_empty_mask_fallback",
                "fallback_event_id": event_id,
                "episode_id": self._episode_id,
                "episode_seed": self._episode_seed,
                "worker_seed": self.worker_seed,
                "fallback_seed": fallback_seed,
                "episode_index": self._episode_index,
                "decision_index": decision_index,
                "fallback_counter": fallback_counter,
                "state_signature_sha256": state_signature,
                "empty_mask_reason": reason,
                # Preserve test-only fault-injection provenance. Production mask
                # evaluations default to runtime_mask_evaluation; a controlled
                # geometry-empty regression must never be mistaken for a
                # naturally observed empty geometry mask.
                "reason_source": str(evaluation.get("reason_source", "runtime_mask_evaluation")),
                "fault_injection": _json_safe(evaluation["fault_injection"]) if "fault_injection" in evaluation else None,
                "native_context": selection["native_context"],
                "policy_action_id": None,
                "policy_action_log_probability": None,
                "policy_selected": False,
                "policy_invoked": False,
                "all_false_mask_passed_to_policy": False,
                "fallback_input_model_shot_id": int(carrier[0]),
                "fallback_selection": selection,
                "corrected_native_action": {
                    "native_shot_id": int(checked[0]),
                    "hitting_position": _json_safe(checked[1]),
                    "landing_position": _json_safe(checked[2]),
                    "recovery_position": _json_safe(checked[3]),
                    "shot_probability": _json_safe(checked[4]),
                },
                "native_validator_reward": float(capture.captured_result[1]),
                "native_validator_corrected": bool(capture.captured_result[2]),
                "native_validator_matching_call_count": int(capture.matching_calls),
                "native_validator_nonmatching_call_count": int(capture.nonmatching_calls),
                "captured_from_actual_env_step_call": True,
                "executed_by_native_environment": True,
                "native_environment_transitions_executed": 1,
                "native_step_reward": transition.native_reward,
                "score_before": score_before,
                "score_after": {"A": int(self.env.score_a), "B": int(self.env.score_b)},
                "score_delta": after_delta,
                "rally_winner_from_score_delta": self._winner_from_delta(after_delta) if transition.done else None,
                "b2_rally_reward_from_score_delta": transition_reward,
                "native_output_record_matches_captured_action": bool(transition.output_record_matches),
                "global_rng_isolation": capture.rng_isolation,
                "native_done": bool(transition.done),
                "native_launch_flag": bool(transition.launch),
            }
            if capture.matching_calls != 1 or not transition.output_record_matches:
                raise WorkerFault(
                    "native_state_error", "Fallback capture/output-record acceptance gate failed", fatal=True
                )
            if capture.rng_isolation is None or not capture.rng_isolation["all_observed_global_rngs_unchanged"]:
                raise WorkerFault(
                    "native_state_error", "Fallback validation consumed a process-global RNG", fatal=True
                )
            if reason == REASON_NATIVE_EMPTY:
                if int(checked[0]) not in NATIVE_FALLBACK_IDS:
                    raise WorkerFault(
                        "native_state_error", "Native empty-vector fallback selected ID outside 2..9", fatal=True
                    )
                if any(float(value) != 0.0 for value in checked[4]):
                    raise WorkerFault(
                        "native_state_error", "Native empty-vector fallback did not zero shot probability", fatal=True
                    )
                if not bool(capture.captured_result[2]):
                    raise WorkerFault(
                        "native_state_error", "Native empty-vector branch did not report correction", fatal=True
                    )
            else:
                expected_native = MODEL_TO_NATIVE[int(carrier[0])]
                if int(checked[0]) != expected_native or bool(capture.captured_result[2]):
                    raise WorkerFault(
                        "native_state_error", "Native-allowed fallback carrier was unexpectedly changed", fatal=True
                    )
            return event, transition
        except Exception as exc:
            # This record is intentionally separate from successful fallback
            # events.  It preserves evidence even if carrier construction,
            # native execution, or post-transition acceptance fails.
            failed = {
                "event_type": "b2_empty_mask_fallback_failed",
                "fallback_event_id": event_id,
                "episode_id": self._episode_id,
                "episode_seed": self._episode_seed,
                "worker_seed": self.worker_seed,
                "fallback_seed": fallback_seed,
                "episode_index": self._episode_index,
                "decision_index": decision_index,
                "fallback_counter": fallback_counter,
                "state_signature_sha256": state_signature,
                "empty_mask_reason": reason,
                "reason_source": str(evaluation.get("reason_source", "runtime_mask_evaluation")),
                "fault_injection": _json_safe(evaluation["fault_injection"]) if "fault_injection" in evaluation else None,
                "failure": f"{type(exc).__name__}: {exc}"[:1200],
                "native_transition_executed": transition_executed,
                "policy_action_id": None,
                "policy_action_log_probability": None,
                "policy_selected": False,
                "fallback_input_model_shot_id": int(carrier[0]) if carrier is not None else None,
                "fallback_selection": selection,
            }
            self._append_event(failed)
            raise

    # ---------- Score, observations and response payloads ----------
    @staticmethod
    def _winner_from_delta(delta: Mapping[str, int]) -> str | None:
        if delta == {"A": 1, "B": 0}:
            return "A"
        if delta == {"A": 0, "B": 1}:
            return "B"
        return None

    def _reward_from_delta(self, delta: Mapping[str, int], *, terminal: bool) -> float:
        if not terminal:
            if delta != {"A": 0, "B": 0}:
                raise WorkerFault("native_state_error", f"Non-terminal native step changed score: {dict(delta)}", fatal=True)
            return 0.0
        winner = self._winner_from_delta(delta)
        if winner is None:
            raise WorkerFault("native_state_error", f"Terminal rally has ambiguous score delta: {dict(delta)}", fatal=True)
        return 1.0 if winner == "A" else -1.0

    def _resolve_terminal(self, *, fallback_occurred: bool = False) -> dict[str, Any]:
        score = {"A": int(self.env.score_a), "B": int(self.env.score_b)}
        delta = {side: score[side] - self._score_baseline[side] for side in ("A", "B")}
        winner = self._winner_from_delta(delta)
        if winner is None:
            raise WorkerFault(
                "native_state_error", f"Native terminal state does not resolve to exactly one rally point: {delta}", fatal=True
            )
        recorded_winner = None
        if len(self.env.output):
            value = self.env.output.iloc[-1].get("getpoint_player")
            if value is not None and str(value) not in {"", "nan", "None"}:
                recorded_winner = str(value)
        expected_name = self.player_name if winner == "A" else self.opponent_name
        if recorded_winner is not None and recorded_winner != expected_name:
            raise WorkerFault(
                "native_state_error",
                "Score-delta winner disagrees with CoachAI final rally record",
                fatal=True,
                details={"winner_side": winner, "score_delta": delta, "recorded_winner": recorded_winner, "expected_name": expected_name},
            )
        raw_reward = 1.0 if winner == "A" else -1.0
        effective_reward = (
            float(self.fallback_win_reward)
            if (winner == "A" and fallback_occurred)
            else raw_reward
        )
        return {
            "winner_side": winner,
            "winner_name": expected_name,
            "recorded_winner_name": recorded_winner,
            "score_before": dict(self._score_baseline),
            "score_after": score,
            "score_delta": delta,
            "raw_rally_reward": raw_reward,
            "terminated_after_fallback": bool(fallback_occurred),
            "b2_rally_reward": effective_reward,
        }

    def _observation(self) -> dict[str, Any]:
        if self._state is None:
            raise WorkerFault("native_state_error", "Cannot create observation without native state", fatal=True)
        try:
            previous = int(self._state[0])
            players = [
                [float(self._state[1][0]), float(self._state[1][1])],
                [float(self._state[2][0]), float(self._state[2][1])],
            ]
            shuttle = [float(self._state[3][0]), float(self._state[3][1])]
        except Exception as exc:
            raise WorkerFault("native_state_error", f"Could not encode native state: {exc}", fatal=True) from exc
        numbers = [value for row in players for value in row] + shuttle
        if not all(math.isfinite(value) for value in numbers):
            raise WorkerFault("native_state_error", "Observation contains non-finite native coordinates", fatal=True)

        if os.environ.get("B2_DEBUG_HISTORY") == "1":
            summary = _describe_history_structure(self._info)
            print(
                "[B2_HISTORY_DEBUG] " + json.dumps(summary, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "coordinate_frame": "A_side_local_CoachAI_native_units",
            "coordinate_units": "CoachAI_native_units",
            "players_xy": players,
            "player_order": ["A", "B"],
            "shuttle_xy": shuttle,
            "previous_native_shot_id": previous,
            "score_ab": [int(self.env.score_a), int(self.env.score_b)],
            "round": int(self.env.round),
            "launch": bool(self._launch),
            "rally_history_length": len(self._info.get("state", [])) if isinstance(self._info, dict) else 0,
        }

    def _state_payload(
        self, *, status: str, reset_return_length: int, reset_output_rows_added: int,
        internal_native_transitions: int, native_reward_trace: list[float],
        fallback_events: list[dict[str, Any]], terminal: dict[str, Any] | None,
        no_policy_rally: bool, rally_events_delta: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._terminated or self._truncated:
            mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
        elif self._mask is not None:
            mask = self._mask
        else:
            mask = self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
        return {
            "status": status,
            "episode_id": self._episode_id,
            "episode_index": self._episode_index,
            "episode_seed": self._episode_seed,
            "observation": self._observation(),
            "action_mask": mask.astype(self.np.bool_).tolist(),
            "action_mask_sha256": hashlib.sha256(mask.astype(self.np.uint8).tobytes()).hexdigest(),
            "ready_for_policy": bool(self._ready),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "reset_return_length": reset_return_length,
            "reset_output_rows_added": reset_output_rows_added,
            "internal_native_transitions": internal_native_transitions,
            "native_reward_trace": native_reward_trace,
            "fallback_events": fallback_events,
            "terminal": terminal,
            "no_policy_rally": no_policy_rally,
            "score": {"A": int(self.env.score_a), "B": int(self.env.score_b)},
            "rally_events_delta": list(rally_events_delta) if rally_events_delta else [],
        }

    def _transition_payload(
        self, *, action_id: int, observation: dict[str, Any], reward: float,
        terminated: bool, truncated: bool, action_mask: Any, native_rewards: list[float],
        fallback_events: list[dict[str, Any]], internal_native_transitions: int,
        output_rows_added: int, terminal: dict[str, Any] | None, normal_action: dict[str, Any],
        rally_events_delta: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        mask = action_mask if action_mask is not None else self.np.zeros((N_ACTIONS,), dtype=self.np.bool_)
        self._ready = bool(not terminated and not truncated and bool(mask.any()))
        return {
            "episode_id": self._episode_id,
            "episode_index": self._episode_index,
            "episode_seed": self._episode_seed,
            "policy_action_id": action_id,
            "observation": observation,
            "action_mask": mask.astype(self.np.bool_).tolist(),
            "action_mask_sha256": hashlib.sha256(mask.astype(self.np.uint8).tobytes()).hexdigest(),
            "ready_for_policy": self._ready,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reward": float(reward),
            "native_reward_trace": native_rewards,
            "internal_native_transitions": internal_native_transitions,
            "fallback_events": fallback_events,
            "normal_action": normal_action,
            "terminal": terminal,
            "output_rows_added": output_rows_added,
            "score": {"A": int(self.env.score_a), "B": int(self.env.score_b)},
            "rally_events_delta": list(rally_events_delta) if rally_events_delta else [],
        }

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.fallback_events_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_safe(dict(event)), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.fallback_events_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    # ---------- Rally-event cursors and collection ----------
    def _clear_native_output(self) -> None:
            # Verbatim former inline block of reset().
            if hasattr(self.env, "output") and hasattr(self.env.output, "iloc"):
                self.env.output = self.env.output.iloc[0:0].copy()

            # ROBUST CURSOR BASELINE
            self._output_cursor = len(self.env.output) if hasattr(self.env, "output") and hasattr(self.env.output, "__len__") else 0
            self._history_cursor = len(self._info.get("state", [])) if isinstance(self._info, dict) else 0

            # FIX: Prevent RallyNet OOD extrapolation drift by isolating match state
            self.env.rally = 0
            self.env.match = 1
            self.env.score_a = 0  # Fixed: Hardcoded to 0 for a fresh episode
            if hasattr(self.env, "score_b"):
                self.env.score_b = 0

    def _begin_event_attempt(self, attempt: int) -> None:
        """Initialize every rally-event cursor for one reset attempt.

        Must run after the attempt's output clearing and before the native reset.  The
        output cursor is taken from the actual post-clear length, which must be zero.
        """
        output = getattr(self.env, "output", None)
        if output is None:
            raise WorkerFault("native_state_error", "CoachAI output frame is missing", fatal=True)
        if not all(hasattr(output, name) for name in ("iloc", "columns", "to_dict")):
            raise WorkerFault(
                "native_state_error", "CoachAI output is not DataFrame-like", fatal=True,
                details={"output_type": type(output).__name__},
            )
        # NOTE: required columns are deliberately NOT checked here.  CoachAI's output frame
        # starts as an empty frame without columns; they appear when the first stroke row is
        # appended, so the check belongs after rows exist (see _collect_rally_events).
        length = int(len(output))
        if length != 0:
            raise WorkerFault(
                "native_state_error", "CoachAI output frame was not empty after clearing", fatal=True,
                details={"attempt": attempt, "length": length},
            )
        self._output_cursor = length
        self._output_cursor_attempt = int(attempt)
        self._history_cursor = 0
        self._event_index = 0
        self._last_ball_round = 0
        self._transition_log = []

    def _read_output_delta(self, attempt: int) -> tuple[Any, int]:
        if attempt != self._output_cursor_attempt:
            raise WorkerFault(
                "native_state_error", "Output cursor belongs to a different reset attempt", fatal=True,
            )
        output = self.env.output
        end = int(len(output))
        start = int(self._output_cursor)
        if end < start:
            raise WorkerFault(
                "native_state_error", "CoachAI output shrank below its committed cursor", fatal=True,
                details={"attempt": attempt, "cursor": start, "current_length": end},
            )
        return output.iloc[start:end].copy(), end

    def _commit_output_cursor(self, attempt: int, new_cursor: int) -> None:
        if attempt != self._output_cursor_attempt:
            raise WorkerFault(
                "native_state_error", "Cannot commit output cursor for a different reset attempt", fatal=True,
            )
        if new_cursor < self._output_cursor:
            raise WorkerFault("native_state_error", "Output cursor cannot move backwards", fatal=True)
        if new_cursor > len(self.env.output):
            raise WorkerFault(
                "native_state_error", "Output cursor cannot exceed current output length", fatal=True,
            )
        self._output_cursor = int(new_cursor)

    def _record_transition(
        self, *, source: str, decision_index: int, output_start: int | None, output_end: int | None,
        policy_action_id: int | None = None, fallback_event_id: str | None = None,
    ) -> None:
        if output_start is None or output_end is None:
            raise WorkerFault(
                "native_state_error", "Native transition did not report its output rows", fatal=True,
            )
        self._transition_log.append(NativeTransitionCapture(
            transition_index=len(self._transition_log),
            source=source,
            decision_index=int(decision_index),
            output_start=int(output_start),
            output_end=int(output_end),
            policy_action_id=policy_action_id,
            fallback_event_id=fallback_event_id,
        ))

    def _collect_rally_events(self, *, terminal: bool) -> list[dict[str, Any]]:
        """Serialize everything new since the last commit, then (only on success) commit.

        Order: read delta -> serialize + verify -> commit cursors.  A failure leaves every
        cursor untouched and is a fatal worker error; the native transition is not retried.
        """
        attempt = self._output_cursor_attempt
        try:
            delta, end = self._read_output_delta(attempt)
            if len(delta) > 0:
                present = {str(column) for column in self.env.output.columns}
                missing = [name for name in REQUIRED_OUTPUT_COLUMNS if name not in present]
                if missing:
                    raise WorkerFault(
                        "native_state_error", "CoachAI output frame lacks required columns", fatal=True,
                        details={"missing_columns": missing},
                    )
            states = self._info.get("state") if isinstance(self._info, dict) else None
            if not isinstance(states, list):
                raise WorkerFault(
                    "native_state_error", "CoachAI info has no native state history list", fatal=True,
                )
            if len(states) < self._history_cursor:
                raise WorkerFault(
                    "native_state_error", "CoachAI native history shrank below its committed cursor",
                    fatal=True,
                    details={"cursor": self._history_cursor, "current_length": len(states)},
                )
            events = serialize_rally_history(
                output_rows=delta.to_dict("records"),
                first_output_position=self._output_cursor,
                history_states=states,
                transitions=tuple(self._transition_log),
                boundary_state=None if terminal else self._state,
                terminal=bool(terminal),
                prior_ball_round=self._last_ball_round,
                event_index_start=self._event_index,
            )
        except WorkerFault:
            raise
        except Exception as exc:
            raise WorkerFault(
                "native_state_error", f"Rally event serialization failed: {type(exc).__name__}: {exc}",
                fatal=True,
            ) from exc
        self._commit_output_cursor(attempt, end)
        self._history_cursor = len(states)
        self._event_index += len(events)
        if events:
            self._last_ball_round = int(events[-1]["ball_round"])
        self._transition_log = []
        return events

    # ---------- Resource shutdown ----------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.rallynet = None
            if self._native_temp is not None:
                self._native_temp.cleanup()
                self._native_temp = None
            original_cwd = getattr(self, "_original_cwd", None)
            if original_cwd is not None:
                os.chdir(original_cwd)