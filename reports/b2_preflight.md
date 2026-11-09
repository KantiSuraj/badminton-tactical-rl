# B2 implementation preflight â€” repository inspection

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

## Latest user verification (2026-11-08)

The user has now completed setup in WSL:

- `.venv-coachai` runs Python 3.12.3 and the upstream CoachAI+ baseline completed five rallies in 9.3 seconds, recording 31 stroke rows.
- The parent `.venv` setup checker completed with `ALL REQUIRED CHECKS PASSED`.
- Both environments are Python 3.12.3 and remain isolated virtual environments.
- The checker found all expected dependency versions, GPU visibility, 104 ShuttleSet set CSVs, required CoachAI+ data/checkpoint files, the exact pinned CoachAI+ commit and a clean checkout.

This confirms the approved Python 3.12 worker environment can execute the native baseline. The dedicated B2 reset/step/terminal runtime probe is still required before implementing the worker protocol or Gymnasium interface.

New individual file: `scripts/setup/probe_coachai_runtime.py`. See `reports/b2_source_audit.md`, Section 9, for its probe scope and output path.


## Runtime probe completed â€” user-provided report (2026-11-08)

The user ran `scripts/setup/probe_coachai_runtime.py` in `.venv-coachai` and supplied the resulting JSON report. The probe status is `PASS`.

- Runtime: Python 3.12.3, PyTorch 2.4.1+cu121, CUDA device `NVIDIA GeForce RTX 4050 Laptop GPU`.
- A-serving: reset returned four values with `launch=True`; A needed to make a service action; three subsequent controlled A decisions used five-value step returns; terminal score delta and `getpoint_player` identified B as the winner.
- B-serving: reset returned four values with `launch=False`; the reset history showed B's serve action already executed; A received the serve; two subsequent controlled A decisions used five-value step returns; terminal score delta and `getpoint_player` identified A as the winner.
- Winner agreement: both sampled terminal rallies showed exactly one side gaining one point and the final record's `getpoint_player` matching the score delta.
- `env.check_winner()` returned `None` in both cases; source confirms it is a game-level 21/30-point test, not a per-rally winner extractor.
- Native rewards varied and are not the project's shared terminal Â±1/0 reward.
- `validate_action()` always applies a `t2c` model-to-native shot-ID conversion before legality checks. Therefore `action_changed=True` can occur with `is_constraint=False`; changed action alone is not a constraint-violation indicator.
- In the sampled actions, `hitting_position` matched the current state shuttle position (`state[3]`).

**B2 v1 service design:** initialize side B as server so its native RallyNet service is performed during `reset()` and the RL-controlled side A starts as receiver. This keeps service outside the frozen canonical action space. The probe demonstrates the behaviour at the pinned commit; add an integration regression test around it.

**Remaining lifecycle issue:** native `reset()` increments `rally` but does not clear `score_a`, `score_b`, the previous-score pointers or accumulated output rows. The upstream `RunEnv.py` intentionally reuses one environment across multiple rallies. The Gymnasium contract requires one rally per episode and the guide proposes neutral or otherwise explicitly sampled score context, so native object reuse must be designed deliberately.

A new individual diagnostic script is now provided: `scripts/setup/probe_coachai_episode_lifecycle.py`. It compares two consecutive rallies in one native environment against a rally in a fresh native environment, recording score/output carry-over and constructor/reset latency. Its output path is `reports/b2_episode_lifecycle_probe.json`.

**Milestone 0 remains in progress.** The basic Python 3.12 runtime, B-serve reset path, external A step path and two sample winner paths are now verified. Repeated-reset policy, deterministic replay, action translation, pure full-joint mask, canonical coordinate mapping, exhaustive terminal branches and data-provenance/leakage audit remain open.

## Latest lifecycle probe results (2026-11-08)

The user-provided lifecycle probe passed. Reusing the same native `BadmintonEnv` for two rallies carried the score from A=1/B=0 into the second reset; the rally counter advanced from 1 to 2 and the output DataFrame accumulated from 5 to 7 rows. A fresh environment started at 0-0 and its first B-serving reset produced an A receive state. Measured fresh construction was 331.495 ms; reset was 171.687 ms. These are single-run measurements, not stable performance targets.

Pinned source inspection explains the result: `reset()` increments the rally and resets `round`, but leaves scores and previous-score pointers untouched; `Record()` relies on those previous-score pointers for the `getpoint_player` field and appends to the environment output DataFrame. The constraint object reads and preprocesses `All_dataset.csv` during construction. Recreating `BadmintonEnv` for every Gym episode therefore reloads constraint data unnecessarily.

The next proposed design is one persistent native environment per worker process with a small, version-pinned adapter that resets score, score pointers, serving side, rally/match/round counters, state and info history before every one-rally reset. The RallyNet agent objects, native constraint object, BaseServe object and cumulative output DataFrame remain alive. This is not yet production-approved; it must pass `scripts/setup/probe_coachai_controlled_reset.py` first.

New individual file: `scripts/setup/probe_coachai_controlled_reset.py`. Run it with `.venv-coachai/bin/python scripts/setup/probe_coachai_controlled_reset.py`. It checks repeated 0-0 B-serving episodes, winner-record agreement, whether run-level objects are reused, cumulative output write/readback, and same-seed trajectory signatures. The script passed Python syntax compilation and a local fake-object reset-helper smoke test; it has not yet been executed against the user's CoachAI+ checkout.


## Controlled-reset verification â€” user-provided report (2026-11-08)

`probe_coachai_controlled_reset.py` completed with `status=PASS`.

- Two consecutive episodes on the same `BadmintonEnv` both started at 0-0 with B serving.
- Both episodes terminated with A winning; the final `getpoint_player` records agreed with the per-rally score deltas.
- `agent_a`, `agent_b`, `badminton_constraint`, `baseServe` and the output DataFrame retained object identity across resets.
- The output CSV contained 10 rows after two rallies and was successfully read back.
- Re-seeding Python, NumPy and Torch with 42 before each episode reproduced equal rounded trajectory signatures in this two-episode test.

**Current architecture decision:** a persistent native environment per worker, with an explicit reset of `score_a`, `score_b`, `pre_score_a`, `pre_score_b`, `serving_player`, `rally`, `match`, `round`, `state` and `info_dict`. Preserve agents, constraints, BaseServe and native output records. This reset adapter is supported by a two-episode runtime test; broader determinism tests and a long-run output/memory policy remain pending.

The next file is `scripts/setup/probe_coachai_action_mask.py`. It tests the proposed canonical/model/native shot-ID mapping, the proposed metric-to-CoachAI cell conversion and the full joint legality checks. The mapping remains a proposal until the user approves treatment of `HALF_SMASH`, `BLOCK` and `CUT`.