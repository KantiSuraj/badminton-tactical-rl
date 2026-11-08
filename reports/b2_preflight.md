# B2 implementation preflight — repository inspection

**Scope:** inspect the uploaded project archive before Step 2 implementation. No CoachAI+ worker, Gymnasium wrapper or action mapping was added in this preflight.

## Existing project structure

The archive already contains:

- `src/badminton_rl/data/`: ShuttleSet loading, canonical event building, split generation, shot-label mapping and court-coordinate conversion.
- `src/badminton_rl/dynamics/`: shuttle-flight, player motion, reachability, action feasibility and shot-execution code.
- `reports/audit/`: the prior ShuttleSet/physics audit results.
- `configs/third_party_manifest.yaml`: external source pin, model artifact metadata and provenance notes.
- `scripts/setup/run_coachai_baseline.py`: a single-process upstream baseline launcher.

It does **not** contain the Step 2 modules yet: `adapters/`, `env/`, `rl/`, a JSONL worker protocol, `B2GymEnv`, B2 action mapping, or Step 2-specific tests.

The current `ShotType` enum in `src/badminton_rl/dynamics/shot_execution.py` is explicitly described as a placeholder requiring future mapping. It belongs to the project's P1 inverse-physics work and must not be reused as proof of native CoachAI+ action semantics.

## Important boundary mismatches

1. The original `docs/SETUP.md` instructed one virtual environment for the project and CoachAI+. It is now updated to use separate Python 3.12 parent (`.venv`) and worker (`.venv-coachai`) environments.
2. `scripts/setup/run_coachai_baseline.py` runs `BadmintonEnv.RunEnv.main` in-process, changes the working directory, extends `sys.path`, and patches `torch.load` to CPU when CUDA is absent. This is a B0 baseline runner, not the isolated Step 2 worker.
3. `scripts/setup/check_setup.py` accepts any `HEAD` containing the pinned commit as an ancestor; Step 2 requires exact `HEAD == 4fd05934f172eec16c74d05c2902983b64c8521e` and a clean working tree.
4. The setup check currently omits active B2 dependencies including `input_data/All_dataset.csv`, `BadmintonEnv/Agent/INV_weight.pth` and `BadmintonEnv/Agent/BC4Serve_weight.pth`.
5. `docs/SETUP.md` suggests `pip install -e ".[dev]"`, but `pyproject.toml` currently defines no `dev` optional-dependency group. `pyproject.toml` also declares no project runtime dependencies.
6. `src/badminton_rl/data/court_coordinates.py` converts ShuttleSet camera pixels through `court_coordinates_v1` into metres with the net at `y=0`. CoachAI+ uses its own hitter-relative coordinates. These transforms must be connected and tested explicitly at the worker boundary; do not assume the numeric coordinates are interchangeable.
7. The canonical 10-family mapping in `src/badminton_rl/data/shot_label_mapping_v1.yaml` is project-specific and not equivalent by index to CoachAI+'s native action categories.

## Test run in this inspection environment

Command:

```bash
PYTHONPATH=src python -m pytest -q
```

Environment: Python 3.13.5 in the inspection container (not the target Python 3.12 parent or worker runtime).

Result:

```text
169 passed, 18 failed, 12 errors
```

All 30 failures/errors shown by the test run arise when the data pipeline tries to write Parquet and neither `pyarrow` nor `fastparquet` is installed. `requirements.txt` contained `#pyarrow<26`, commented out, even though the event and split pipeline tests require a Parquet engine.

**Preflight correction made:** uncommented `pyarrow<26` in `requirements.txt`. The package was not installed in this inspection environment, so the Parquet-dependent tests have not yet been re-run and are not claimed as passing. After installing project requirements, rerun the full suite before beginning Step 2 implementation.

Recommended command on the project machine:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
PYTHONPATH=src python -m pytest -q
```

Create/activate the distinct Python 3.12 CoachAI worker environment separately; do not install its dependency pins into the Python 3.12 parent.

## Scope decision

Continue only with Milestone 0 until local checkout identity, clean status, active data/checkpoint files, and the native Python 3.12 isolated-worker runtime probe are verified. Do not implement a guessed worker or freeze the canonical-to-native family map yet.


## Local CoachAI+ verification update (2026-11-08)

The user verified that local `HEAD` equals the exact required CoachAI+ SHA `4fd05934f172eec16c74d05c2902983b64c8521e`. Git still reports one untracked generated output, `output_data/output_game_1.csv`; preserve it and move it outside the checkout before claiming a clean working tree.

The six user-supplied SHA-256 values were checked against `configs/third_party_manifest.yaml`: hashes for `target_players_ids.pkl`, `hyperparameters.pkl`, `RallyNet_weight.trc`, `INV_weight.pth`, and `BC4Serve_weight.pth` match the manifest. The `All_dataset.csv` hash is a newly captured local fingerprint because the manifest had no expected hash; it is now recorded as a local observation, not asserted as an upstream reference hash.

The local machine has Python 3.12 and does not have `python3.10`. The user has approved a separate Python 3.12 worker virtual environment, preserving the subprocess boundary. The original CoachAI+ environment metadata declares Python 3.10.12; the project manifest records a prior five-rally CPU baseline run under Python 3.12. The worker compatibility probe remains outstanding.

The native API runtime probe has not yet been run. Milestone 0 remains in progress.


## Runtime decision update (2026-11-08)

The user explicitly approved Python 3.12 for the CoachAI+ worker. Parent and worker must use separate virtual environments (`.venv` and `.venv-coachai`) and separate processes. Install `requirements-coachai.txt` only into `.venv-coachai`. The original upstream Python 3.10.12 declaration is retained as provenance, not treated as the selected runtime. No B2 worker compatibility claim is made until the dedicated runtime probe passes.
