# B2 CoachAI+ source audit

**Status: IN PROGRESS â€” pinned-source inspection and one successful runtime rally for each serve mode. Not yet cleared for worker implementation.**

- Project: `badminton-tactical-rl`
- Required upstream pin: `4fd05934f172eec16c74d05c2902983b64c8521e`
- Audit basis: source files at the exact commit above, inspected through their immutable GitHub raw URLs.
- Local checkout verification: user-provided terminal output confirms `HEAD` and the resolved required commit both equal `4fd05934f172eec16c74d05c2902983b64c8521e`. `git status --short` reports one untracked generated file: `output_data/output_game_1.csv`; therefore the checkout is not yet clean.
- Runtime probe: user-provided `b2_runtime_probe_v1` report executed successfully under the isolated Python 3.12.3 worker environment on 2026-11-08. It completed one A-serving rally and one B-serving rally. See Section 10 for the exact tested outcomes and remaining limits.

## 1. Checkout identity and source evidence

The project manifest already records the full expected commit SHA as `4fd05934f172eec16c74d05c2902983b64c8521e` (`configs/third_party_manifest.yaml`). The ZIP does not include the corresponding checkout, therefore the following mandatory local checks remain open:

```bash
COACHAI_ROOT=../external/CoachAI-Plus

git -C "$COACHAI_ROOT" rev-parse HEAD
git -C "$COACHAI_ROOT" rev-parse '4fd05934f172eec16c74d05c2902983b64c8521e^{commit}'
git -C "$COACHAI_ROOT" status --short
```

`HEAD` must equal the full pinned SHA, and `git status --short` must be empty. The existing `scripts/setup/check_setup.py` currently checks whether the pinned commit is an ancestor of `HEAD`; that is weaker than the exact-`HEAD` requirement and must be corrected before the integration is considered verified.

All source links in this report are immutable permalinks to that full SHA:

- [`BadmintonEnv/Environment.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/Environment.py)
- [`BadmintonEnv/RunEnv.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/RunEnv.py)
- [`BadmintonEnv/Utils/Constraint.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/Utils/Constraint.py)
- [`BadmintonEnv/Utils/RewardFunction.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/Utils/RewardFunction.py)
- [`BadmintonEnv/Utils/BaseServe.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/Utils/BaseServe.py)
- [`BadmintonEnv/Agent/RallyNet.py`](https://github.com/KuangDW/CoachAI-Plus/blob/4fd05934f172eec16c74d05c2902983b64c8521e/BadmintonEnv/Agent/RallyNet.py)

## 2. Audit question table

| Guide question | Static source finding | Status |
|---|---|---|
| Environment module/class | `BadmintonEnv/Environment.py`, class `BadmintonEnv` at line 6. `RunEnv.py` imports it as `.Environment.BadmintonEnv` (line 0). | **Static source verified** |
| Constructor signature | `BadmintonEnv(player_agent, opponent_agent, player_name, opponent_name, episodes, is_match, is_constraint, is_have_serve_state, filepath)`; all parameters are required (Environment.py lines 7â€“18). | **Static source verified** |
| Player A/B assignment | `player_agent` is stored as `agent_a`, `opponent_agent` as `agent_b` (Environment.py lines 19â€“28). `RunEnv.main` constructs the player1 agent as `player_agent`, player2 as `opponent_agent` (RunEnv.py lines 19â€“60). | **Static source verified** |
| Is the opponent called internally? | Yes. After A's legal action, `Environment.step` calls `self.agent_b.action(..., launch=False)` (Environment.py lines 278â€“288). When B serves, `reset()` calls `agent_b.action(..., launch=True)` (lines 135â€“143). | **Static source verified** |
| Exact reset tuple | A-serving reset returns `(state, info, False, True)` (Environment.py lines 144â€“151). B-serving reset performs B's serve through `step()` and returns that branch's four values (lines 135â€“143). | **Static source verified; runtime probe pending** |
| Exact step tuple | The normal A-controlled path returns `(state, reward, info, done, False)`; the B-serving launch branch returns `(state, info, done, False)` (Environment.py lines 157â€“223 and 238â€“258). Thus the native API is path-dependent. | **Static source verified; runtime probe pending** |
| Meaning of `launch` | It is a Boolean `is_launch`, not a serialized launch-state object. `True` denotes the service action; after the serve, the returned value is `False` (Environment.py lines 152â€“158, 209â€“223, 354â€“363). | **Static source verified** |
| Who creates serve state? | `is_have_serve_state=True` calls the serving side's `serve_state()`; otherwise reset uses `BaseServe.serve_state()` (Environment.py lines 114â€“129). `BaseServe` reads `./input_data/All_dataset.csv` (BaseServe.py lines 4â€“12) and samples a stored serve state using NumPy RNG (lines 123â€“128). | **Static source verified; seed/runtime probe pending** |
| State fields and units | State is a 4-tuple: previous/opponent shot category, own position, opponent position, shuttle position. RallyNet translates these values using the native `177.5` and `240` normalizers (RallyNet.py lines 256â€“318 and 320â€“338); `Environment.py` flips coordinates when changing hitter perspective (lines 209â€“223 and 278â€“285). | **Field order inferred from source use; exact semantic/unit contract and round-trip probe pending** |
| Native action fields | RallyNet returns `(output_shot, raw_states[-1], output_land, output_move, normalized_tuple)` (RallyNet.py lines 438â€“466): shot ID, hitting position, landing position, recovery/movement target, and a 10-element shot probability vector. | **Static source verified** |
| Native action mask | No public full-joint mask method was found in the inspected environment/constraint path. `validate_action` is a mutating repair routine, not a pure legality predicate. | **Static source verified; adapter design required** |
| Shot constraints | `validate_action` converts the model output shot ID through `t2c`, then checks an empirical distribution keyed by previous shot, hitting block and binned player-y movement (Constraint.py lines 381â€“425). | **Static source verified** |
| Landing constraints | It maps the landing coordinate to a native area and moves an invalid landing to a nearest valid area, with a random fallback if no valid area exists (Constraint.py lines 447â€“466). | **Static source verified** |
| Recovery constraints | It calls `correct_position`; recovery distance is normalized by 177.5 and 240 and capped at `max_total=1.6` (Constraint.py lines 314â€“339 and 468â€“480). | **Static source verified** |
| Native reward meaning | `Reward_function` adds/subtracts constraint and win/loss penalties; it is not simply a terminal Â±1 reward (RewardFunction.py lines 0â€“20). | **Static source verified** |
| Rally winner | On A fault the environment increments `score_b`; on B fault it increments `score_a` (Environment.py lines 227â€“258 and 290â€“325). `info_dict['env_score']` stores `[score_a, score_b]` (lines 398â€“400). `check_winner()` is a game-level 21-point/30-point check (lines 403â€“414), not a per-rally winner function. | **Source path identified; runtime terminal probes required** |
| Constraint data | `BadmintonConstraint` defaults to `./input_data/All_dataset.csv` (Constraint.py lines 5â€“6); `BaseServe` uses the same CSV by default (BaseServe.py lines 4â€“12). | **Static source verified; hash/provenance pending** |
| RallyNet checkpoint/data | `RunEnv.py` reads `input_data/target_players_ids.pkl`, `BadmintonEnv/Agent/hyperparameters.pkl` and loads `RallyNet_weight.trc` (lines 8â€“35 and 37â€“54). RallyNet constructor loads `INV_weight.pth` (RallyNet.py lines 167â€“187). Serving inference loads `BC4Serve_weight.pth` (lines 425â€“435). | **Static source verified; exact local files/hashes/provenance pending** |
| RallyNet history | `RallyNet.action` receives `info` and indexes its `round`, `player`, `state` and `action` history (RallyNet.py lines 387â€“400; feature translation at lines 276â€“318). | **Static source verified** |
| Determinism | Inference is stochastic: latent sampling uses `torch.randn_like`; landing/recovery sample Gaussian-mixture components and Gaussian noise; shot category is sampled from `Categorical` (RallyNet.py lines 411â€“420 and 449â€“465). Serve sampling uses NumPy RNG; constraint fallback uses Python `random`. | **Static source verified; complete RNG control/reproducibility pending** |

## 3. Findings that change the adapter design

### 3.1 Use a worker-owned Boolean `launch` state

The native `launch` value is a Boolean indicating whether the controlled action is a serve. The worker should own it and pass it unchanged to the native `env.step(action, is_launch)` call. It is not an opaque physics object.

### 3.2 Reset must normalize multiple native paths

The native `reset()` is always a four-value return, but the values differ by serving side. If A serves, A is expected to make a service action. If B serves, `reset()` itself calls RallyNet and steps B's service action before returning. This means the worker's `reset` operation must normalize both cases and must handle a possible terminal B-service fault before exposing an observation/mask. The parent must not assume that every `reset()` produces a live decision state.

### 3.3 Candidate service policy: let B serve so A receives

The canonical 10-family action taxonomy intentionally excludes service. The cleanest candidate for B2 v1 is therefore to have B serve and A receive: after constructing the native environment, set `serving_player = "B"` before calling `reset()`. This uses an internal field assignment rather than a documented reset option; the exact behavior must be runtime-tested and logged. It must not be hidden inside a random worker policy.

`LAUNCH_BY_BC` is `True` in RallyNet.py (line 14), and its launch path uses `BC4Serve_weight.pth` (lines 425â€“435). Thus B's service decision is performed by CoachAI+'s existing serve model, not by the RL action space.

### 3.4 One-rally Gym episodes need an explicit score reset policy

`Environment.reset()` increments the rally number but does not reset `score_a`, `score_b`, or `serving_player` (Environment.py lines 111â€“151). Terminal paths update the score. `Record()` calls `reset_for_next_game()` (lines 437â€“483), and the latter resets scores only after `check_winner()` reports a game win and `is_match` is true (lines 403â€“426). Consequently, calling native `reset()` between Gym episodes does not create independent neutral-score rallies. Before implementation, choose and test one explicit policy: reset score/serve fields under worker ownership, reconstruct the environment while reusing loaded model instances, or deliberately treat episodes as sequential points with score context. The choice must agree with the guide's one-rally and reset-context contract.

### 3.5 Do not call `validate_action` to build the mask

The function returns a possibly rewritten action, a penalty/reward and an `isConstraint` flag. It may select a random fallback shot or landing area, change the landing point to a nearest valid area, and clip recovery. Calling it for every candidate would mutate/repair the candidate and can consume RNG; it would not answer whether the original canonical macro-action is jointly legal. Implement a separate deterministic, non-mutating reference legality evaluator from the audited constraint arrays and native court/out checks. Test that every action accepted by the mask is passed through unchanged by the native validator.

### 3.6 Investigate probability-vector indexing before native action construction

`RallyNet.action` returns the shot probability vector using output slots `1..10` (RallyNet.py lines 453â€“459). `validate_action` remaps `action[0]` through `t2c` (Constraint.py lines 381â€“386), and in its correction path indexes that remapped category into the probability vector (lines 412â€“445). The source therefore presents a possible class-order alignment issue when a shot is corrected. Do not paper over it with a guessed one-hot vector. Build a tiny controlled probe that prints (a) sampled model output ID, (b) mapped native category, (c) probability-vector order and (d) the validator's correction result.

### 3.7 Canonical shot taxonomy cannot be frozen by name similarity

The project's own `src/badminton_rl/data/shot_label_mapping_v1.yaml` defines 10 canonical groups:

`CLEAR, DROP, SMASH, HALF_SMASH, DRIVE, PUSH, LIFT, NET_SHOT, BLOCK, CUT`.

CoachAI+'s native action categories instead include short service, clear, push shot, smash, smash defence, drive, net shot, lob, drop and long service (Constraint.py lines 7â€“18). The native action ID bridge is `t2c` at lines 381â€“386. There is no distinct native `HALF_SMASH` or `CUT` category, and `BLOCK`'s source label `return net` does not appear as a native `type_mapping` action class. The fact that source annotation maps Chinese `åˆ‡çƒ` to `Drop` does not prove the separate canonical `CUT` family (`passive drop`) should also collapse to `Drop`.

**Proposed conservative B2 v1 policy for user approval:** map the directly represented families `CLEAR`, `DROP`, `SMASH`, `DRIVE`, `PUSH`, `LIFT` (to native `Lob`), and `NET_SHOT`; mark `HALF_SMASH`, `BLOCK`, and `CUT` explicitly unsupported in B2 v1 rather than silently collapse their semantics. The 1350 action IDs can remain stable, while mask entries for unsupported families remain false. Do not create/freeze `coachai_v1.yaml` until this policy is approved.

## 4. Runtime and reproducibility blockers

1. **Local checkout:** verify exact `HEAD`, tag resolution and clean status on the user's machine.
2. **Python 3.12 worker:** create the separate `.venv-coachai` environment from `requirements-coachai.txt`, record its explicit interpreter path, and prove imports and native behavior using the runtime probe. The user has explicitly approved this runtime change; compatibility remains pending.
3. **Model/data files:** verify every active model/data file exists and hash it. The project manifest contains hashes for several files, but the source path shows the worker also needs `INV_weight.pth` and `BC4Serve_weight.pth`; those hashes must be checked against the user's checkout. `BC_weight.pth` is commented out in the active `RunEnv.main` RallyNet path (RunEnv.py lines 12â€“17), so it should not be required unless a different agent path is selected.
4. **CPU loading:** `RallyNet.__init__` loads `INV_weight.pth` without `map_location` (RallyNet.py lines 184â€“187), and the serving branch loads `BC4Serve_weight.pth` without `map_location` (lines 426â€“429). The existing baseline wrapper patches `torch.load` defaults to CPU when CUDA is unavailable. Reproduce and test the same behavior inside the isolated worker without editing CoachAI+ source.
5. **RNGs:** explicitly seed Python `random`, NumPy and Torch before any serve/model/constraint sampling. Test whether `torchsde` with `bm=None` is reproducible under the pinned runtime. Record any hardware-specific non-determinism.
6. **Output path:** native `close()` writes `self.output` to the constructor's `filepath` (Environment.py lines 485â€“486). The worker must pass a worker-owned writable temporary/output path and must not pollute the external checkout.
7. **Data leakage:** audit `All_dataset.csv`, target-player IDs, hyperparameters and model checkpoint provenance against the frozen match/player splits. Until this is complete, do not claim leakage-free player-disjoint generalization.

## 5. Required runtime probe before worker implementation

Run this inside the configured isolated Python 3.12 worker environment, against the exact pinned checkout. The probe must be developed only after the local checkout checks above pass. It should capture:

- constructor and agent object setup;
- default A-serve reset tuple and B-serve reset tuple;
- one normal A action and its five-value step return;
- a B-serve failure path if one can be induced safely;
- `state` structure and numeric ranges for all four fields;
- `launch` transition `True -> False`;
- `info` keys/lengths through a rally;
- score values before/after terminal outcomes for both winners;
- action before/after `validate_action`, the three constraint branches and RNG calls;
- output file behavior on `close()`.

**Do not implement the real worker until the action translation, non-mutating full action mask, service behavior, terminal winner extraction and observation conversion have passed this probe.**

## 6. Source status summary

| Classification | Items |
|---|---|
| Verified from pinned source (static) | Class, constructor, A/B ownership, internal opponent call, native reset/step branches, Boolean `launch`, five-field native action, constraint mutation behavior, reward function shape, score updates, RallyNet history use and stochastic sampling |
| Verified by runtime probe | Both serve modes reset with four values; B-serving reset executes a B serve and returns an A receive state; externally controlled A steps returned five values in all recorded decisions; both sampled rallies terminated; score-delta winner agreed with `getpoint_player`; tested hitting positions matched the current shuttle state |
| Frozen by project guide | Python 3.12 parent / separate Python 3.12 worker venv split (user-approved); one Gym episode per rally; canonical 10-family action semantics; shared Â±1/0 reward; no edits to CoachAI+ source |
| Proposed, requires approval | Unsupported status for `HALF_SMASH`, `BLOCK`, `CUT`; per-episode score reset policy; fresh native environment per Gym reset vs a controlled reset adapter |
| Unresolved | Repeated-reset score/output lifecycle; independent episode score policy; exhaustive terminal winner branches; deterministic replay; canonical-to-RallyNet-to-CoachAI shot mapping approval; coordinate transform and units; non-mutating full-joint action mask; data leakage provenance |

**Milestone 0 is not yet passed.** The required pinned runtime now works and several API facts are confirmed. The remaining action, coordinate, constraint, episode-reset and determinism questions still block production worker implementation.


## 7. Local verification update â€” user-provided terminal output (2026-11-08)

The user ran the prescribed Git and SHA-256 checks in their WSL environment and provided these results.

### Checkout identity

| Check | Result | Status |
|---|---|---|
| `git rev-parse HEAD` | `4fd05934f172eec16c74d05c2902983b64c8521e` | **Exact pin verified** |
| `git rev-parse '4fd05934f172eec16c74d05c2902983b64c8521e^{commit}'` | `4fd05934f172eec16c74d05c2902983b64c8521e` | **Pinned commit resolves** |
| `git status --short` | `?? output_data/output_game_1.csv` | **Working tree not clean due to one untracked output file** |

No tracked source modification was reported. Do not delete `output_data/output_game_1.csv` merely to clean the checkout; preserve it, then move it outside the CoachAI+ checkout if it is a generated run artifact. Re-run `git status --short` after moving it. The report must not call the checkout clean until that output is no longer reported.

### Model and data fingerprints

| File | User-reported SHA-256 | Manifest comparison |
|---|---|---|
| `input_data/All_dataset.csv` | `3116fd86391e2cec477b513f9e00bc75efc596bf3862779927eee7c923403ba6` | No expected SHA-256 was present in the project's third-party manifest; record as observed fingerprint |
| `input_data/target_players_ids.pkl` | `b6c2e4ea5b58923d20c1a7b51a0ee6f8d83a2958d622183995c271222f51e755` | Matches manifest |
| `BadmintonEnv/Agent/hyperparameters.pkl` | `7e5ab372205f5b37db450e519040bdb358136e4c2b66461a1374b50c43588c7d` | Matches manifest |
| `BadmintonEnv/Agent/RallyNet_weight.trc` | `bed1f112d542903a5c6db898f56a0e6baf3d240abe2879041cc092805e985c7d` | Matches manifest |
| `BadmintonEnv/Agent/INV_weight.pth` | `ae9d0e1837c3515c3507172b060e82808e83840938313523509438d56eff3c60` | Matches manifest |
| `BadmintonEnv/Agent/BC4Serve_weight.pth` | `8036be9ccf012174a51ded00689d494b530f01249ea28ce4e7dceb93653be344` | Matches manifest |

Five expected model/data hashes match exactly. The active-path files listed above are now fingerprinted locally; the observed `All_dataset.csv` fingerprint should be added to the project's manifest after its provenance is confirmed. These hashes establish file identity relative to the recorded manifest values; they do not establish training-split provenance or absence of data leakage.

### Python runtime

The user's WSL environment reports that Python 3.12 is installed and `python3.10` is not available on `PATH`. The original CoachAI+ environment metadata declares Python 3.10.12. The project manifest also records a prior five-rally CPU baseline run under Python 3.12; this is useful compatibility evidence but is not a runtime probe of the new subprocess worker.

**Runtime decision: APPROVED.** The user explicitly approved using Python 3.12 for both parent and worker, while retaining strict process isolation and separate virtual environments. The worker interpreter will be configured explicitly (default setup path: `.venv-coachai/bin/python`) and its packages installed from `requirements-coachai.txt`. CoachAI+ imports remain worker-only. Compatibility is **not yet verified**; the dedicated B2 runtime probe is still mandatory.

### Milestone 0 status

- Exact local CoachAI+ commit: verified.
- Tracked source edits: none reported; one untracked generated output prevents a clean `git status`.
- Active checkpoint/data fingerprints: captured; five compared values match existing manifest entries, and the local `All_dataset.csv` fingerprint is recorded as an observation.
- Worker interpreter: Python 3.12 in a separate `.venv-coachai` environment approved; environment creation/import/runtime probe pending.
- Runtime API probe: **not yet run**.
- Service handling, action construction/mask equivalence, winner extraction and deterministic replay: **not yet runtime-verified**.

Milestone 0 therefore remains **in progress**. Do not begin real worker implementation until the runtime decision is explicit and the required native API probe has passed.

## 8. Latest local verification â€” user-provided terminal output (2026-11-08)

The user's latest WSL terminal output supersedes the earlier untracked-file status above:

- `HEAD` and the resolved pinned commit both equal `4fd05934f172eec16c74d05c2902983b64c8521e`.
- `git status --short` is empty; the working tree is now clean.
- The isolated `.venv-coachai` environment runs Python 3.12.3.
- The CoachAI+ baseline completed five rallies, reported 31 stroke rows and took 9.3 seconds. It wrote output beneath `../external/CoachAI-Plus/output_data`.
- The parent setup checker reports `ALL REQUIRED CHECKS PASSED`.
- The checker validated parent Python 3.12.3, worker Python 3.12.3, the required worker package versions, GPU visibility, all 104 ShuttleSet set CSVs, all required CoachAI+ files, the exact source commit and a clean working tree.
- The six local artifact SHA-256 values were supplied by the user earlier; five compare against existing expected values in the project manifest and match. `All_dataset.csv` remains a locally observed fingerprint because no expected hash was defined in the manifest.

This verifies the user's isolated Python 3.12 environment can import/load/run the upstream baseline entry point for five rallies. It does **not** yet verify the B2-specific runtime contract: forced B-serve reset, A-controlled action stepping, per-rally terminal winner extraction, stable observation serialization or deterministic replay.

## 9. Dedicated probe implementation

A new script is provided at `scripts/setup/probe_coachai_runtime.py`. It is intentionally separate from the production worker and does not modify CoachAI+ source. It:

1. requires the exact pinned commit and a clean checkout;
2. records SHA-256 and file sizes for all six active native data/model files;
3. loads the same two RallyNet agents and hyperparameters as the pinned `RunEnv.py` runner;
4. probes one A-serving rally and one B-serving rally by setting the native `serving_player` before reset;
5. records the native reset tuple, normal step tuple shape, launch flag, compact state/info/action structures, inference latency, constraint validation before/after samples, score delta and final output-record winner;
6. stores a JSON report outside the CoachAI+ checkout under `reports/b2_runtime_probe.json` by default;
7. directs CoachAI+ CSV output to a temporary directory outside the checkout.

The script has passed local syntax compilation in the authoring environment. It has **not** been run against the user's WSL checkout. Milestone 0 remains in progress until the user runs it and the report demonstrates completed A-serve and B-serve paths with consistent terminal winner evidence.


## 10. Runtime probe results â€” user-provided JSON (2026-11-08)

**Input:** `b2_runtime_probe_v1`, status `PASS`, Python 3.12.3, torch 2.4.1+cu121, CUDA device `NVIDIA GeForce RTX 4050 Laptop GPU`, seed 42. The report recorded the exact pinned CoachAI+ commit and the active model/data fingerprints. The user previously confirmed the local checkout is clean and the worker baseline setup passes.

### 10.1 A serves

- Native reset returned four values: `(state, info, done, launch)`.
- `done=False`, `launch=True`, history length 1, no latest action. This leaves the first A action as a service decision, which is outside the canonical 10-family macro-action contract.
- Three externally controlled A decisions followed; every `env.step(action, launch)` returned five values.
- The rally terminated. Score changed from A=0/B=0 to A=0/B=1.
- Final native output row recorded `getpoint_player=Kento MOMOTA`; score delta independently identified side B. These agreed.

### 10.2 B serves

- The probe set `env.serving_player = "B"` before reset.
- Native reset returned four values with `done=False`, `launch=False`. Reset-side `info` had two history states/actions and the latest action was the B service action. The returned state is therefore an A receiving state, not a pre-serve decision state.
- Two externally controlled A decisions followed; every `env.step(action, launch)` returned five values.
- The rally terminated. Score changed from A=0/B=0 to A=1/B=0.
- Final native output row recorded `getpoint_player=CHOU Tien Chen`; score delta independently identified side A. These agreed.

**Integration decision:** B-serve/A-receive is the appropriate initial B2 mode. It keeps service out of the canonical action space. The assignment to `serving_player` is a pinned-source internal attribute rather than a documented reset argument; the runtime test demonstrates it works at this commit. Keep this assumption explicit and cover it with an integration test.

### 10.3 Normal step and native reward

All externally controlled A steps in both runs returned the five-value tuple:

`(next_state, native_reward, next_info, done, next_launch)`.

Observed native rewards were `[0.0, -1.0, -2.0]` in the A-serving run and `[-1.0, 0.0]` in the B-serving run. These values include native constraint/reward semantics and do not represent the shared terminal win/loss reward. The project reward must be computed from a verified terminal winner, not copied from the native reward.

### 10.4 Terminal winner evidence

For both sampled terminal rallies:

- `done=True`;
- exactly one side's score incremented by one;
- the final `env.output` record's `getpoint_player` matched the name assigned to that side;
- `env.check_winner()` returned `None`.

The source defines `check_winner()` as a game-level 21-point/30-point condition (`Environment.py` lines 403â€“414). Do not use it as a rally winner function. For one-rally termination, use the score delta and final record as independent checks; fail closed if they disagree or do not identify exactly one point winner.

**Scope limit:** one completed sample for each serve path is not exhaustive coverage of all terminal violations, faults, invalid actions or serve failure. Add targeted terminal-path tests before declaring winner extraction fully verified.

### 10.5 Action construction clues and validator behaviour

The runtime report observed the five-field action:

`(shot_type, hitting_position, landing_position, recovery_position, shot_probability)`.

For the sampled A actions, `hitting_position` matched `state[3]`, the current shuttle position. This is consistent with RallyNet's source implementation returning `raw_states[-1]` as the hitting position (`RallyNet.py` lines 438â€“465). The adapter must still explicitly validate this equality and shape before constructing a native action.

The report also showed `validate_action()` sometimes changed the action when its returned `is_constraint` flag was false. This is explained by source lines 381â€“387: the function unconditionally applies `t2c` shot-ID conversion before constraint checks. A changed shot ID is therefore not, by itself, evidence that a legality constraint fired.

The mapping table in `Constraint.py` is:

```python
t2c = {
    1: 1, 2: 10, 3: 2, 4: 4, 5: 9,
    6: 8, 7: 6, 8: 7, 9: 3, 10: 5, 11: 11,
}
```

Maintain separate identifiers for canonical ShuttleSet family, RallyNet/model shot ID and CoachAI native shot-category ID. Do not pass native IDs into an API path that expects pre-conversion RallyNet IDs.

### 10.6 Runtime environment

- Both player agents loaded as `BadmintonEnv.Agent.RallyNet.RallyNet` and were in evaluation mode.
- CUDA was available and the probe used `cuda:0`.
- The target-player vocabulary contained 27 players; CHOU Tien Chen was ID 1 and Kento MOMOTA was ID 0.
- The probe used RallyNet to produce sample A actions. Production B2 will replace A's policy action generation with the macro-action adapter; RallyNet remains the native opponent on side B.

### 10.7 What this probe did not verify

- Reproducibility across repeated resets using the same seed.
- Resetting the same native environment object after a terminal rally.
- That score, pre-score counters and output records reset between one-rally Gym episodes.
- All terminal branches and B-serve faults during `reset()`.
- A pure, non-mutating evaluator equivalent to all three native constraint stages.
- Full action-mask legality over all 1,350 joint macro-actions.
- Canonical court transform equivalence and exact coordinate units.
- Shot-family mapping approval and information-loss treatment.
- Constraint/RallyNet data split provenance and leakage.

### 10.8 Milestone status after runtime probe

| Item | Current status |
|---|---|
| Exact CoachAI+ commit and clean checkout | Verified by user terminal output |
| Separate Python 3.12.3 worker venv and dependencies | Verified by user setup checker |
| Five-rally upstream baseline | Runtime passed |
| A-serving and B-serving reset path | Runtime passed for one rally each |
| External A step tuple | Runtime passed for 5 total decisions |
| Per-rally winner evidence | Score delta and `getpoint_player` agreed in both runs |
| Shared reward implementation | Not yet implemented; extraction evidence is promising but terminal cases need broader tests |
| One-rally Gym reset lifecycle | **Open; targeted lifecycle probe required** |
| Native candidate action builder and joint mask | **Open; must not use mutating `validate_action()` as the mask predicate** |
| Full Step 2 worker | Not started; correctly blocked pending remaining contract decisions |

**Conclusion:** the B2 native integration path is viable in the approved Python 3.12 worker environment. Milestone 0 is materially advanced but remains open. The next probe must settle score/output lifecycle and environment reconstruction cost before `B2GymEnv.reset()` is designed.

## 11. Episode lifecycle probe results â€” user-provided JSON (2026-11-08)

**Input:** `b2_episode_lifecycle_probe_v1`, status `PASS`, Python 3.12.3, PyTorch 2.4.1+cu121, CUDA device `cuda:0`, seed 42. The report reverified the pinned CoachAI+ commit and all active model/data file fingerprints.

### 11.1 Reusing one native environment without an episode reset shim

The same `BadmintonEnv` object completed two rallies:

| Metric | Rally 1 | Rally 2 |
|---|---:|---:|
| Score before reset | A=0, B=0 | A=1, B=0 |
| Score after reset | A=0, B=0 | A=1, B=0 |
| Native rally number after reset | 1 | 2 |
| Reset history entries | 2 | 2 |
| A decisions | 2 | 1 |
| Score after rally | A=1, B=0 | A=1, B=1 |
| Output rows added | 5 | 2 |
| Reset latency | 698.006 ms | 171.858 ms |

This confirms that native `reset()` does not create an independent 0-0 rally: it carries the prior match score and increments the rally counter. The `output` DataFrame also accumulates records between rallies.

### 11.2 Constructing a fresh environment

A newly constructed `BadmintonEnv` started with score A=0/B=0, rally 0 before reset, and after B-serving reset had score A=0/B=0, rally 1, two history entries and `launch=False`. Its sampled rally terminated with A winning. It recorded five rows.

Measured costs were 331.495 ms for construction and 171.687 ms for reset. These measurements are from one diagnostic run on the user's RTX 4050 / Python 3.12.3 setup; they are not a latency benchmark.

### 11.3 Resource and output implications from pinned source

Static source inspection at the exact pin establishes:

- `BadmintonEnv.__init__` initializes `score_a`, `score_b`, `pre_score_a`, `pre_score_b`, `rally`, `match`, `round`, `serving_player`, `state`, `info_dict`, `badminton_constraint`, `baseServe` and `output` (`BadmintonEnv/Environment.py`, lines 19â€“60).
- `BadmintonEnv.reset()` increments `rally` and sets `round=1`, but does not reset either score, either `pre_score_*` field, `match`, or `serving_player` (`Environment.py`, lines 110â€“151).
- `info()` rebuilds `info_dict` when `round == 1` (`Environment.py`, lines 364â€“401).
- `Record()` uses `pre_score_a` / `pre_score_b` to derive `getpoint_player`, appends to `self.output`, and `close()` writes the accumulated DataFrame to `filepath` (`Environment.py`, lines 427â€“486).
- `BadmintonConstraint.__init__` reads `All_dataset.csv` and calculates the action-distribution tables (`BadmintonEnv/Utils/Constraint.py`, lines 5â€“18 and 157â€“168). Reconstructing the native environment therefore reconstructs/reloads this constraint object.
- `RallyNet.action()` reconstructs its context from the `info` history passed by the environment and calls `contextualize()` to set `_ctx` for inference (`BadmintonEnv/Agent/RallyNet.py`, lines 387â€“417). Shot and spatial inference use Torch random sampling (`RallyNet.py`, lines 126â€“139 and 415â€“459).

### 11.4 Preferred reset design to validate next

For the one-rally Gymnasium contract, prefer **one persistent native `BadmintonEnv` per worker process plus a version-pinned episode-reset adapter**, instead of reconstructing the environment and reloading constraints for every rally.

Before each native `reset()`, the adapter should test resetting these episode/context fields:

```python
env.score_a = 0
env.score_b = 0
env.pre_score_a = 0
env.pre_score_b = 0
env.serving_player = "B"
env.rally = 0
env.match = episode_id
env.round = 1
env.state = None
env.info_dict = {}
```

Preserve `agent_a`, `agent_b`, `badminton_constraint`, `baseServe` and `output`. `env.reset()` then increments `rally` to 1, performs B's serve internally, and gives A the first receive state. A unique `match=episode_id` keeps accumulated CSV rows distinguishable while `rally=1` and score 0-0 give RallyNet a consistent independent-rally context. `pre_score_*` must be reset together with score fields; otherwise `Record().getPointer()` can fail to identify the next rally winner when the previous score value is encountered again.

**Important status:** this is a proposed adapter, not yet accepted as production-safe. It deliberately writes native instance attributes without modifying upstream source. Run `scripts/setup/probe_coachai_controlled_reset.py` to check neutral scores, correct winner records, output accumulation, resource reuse and same-seed replay before adopting it in the worker. If the probe reveals hidden state carry-over or determinism failure, fall back to a fresh `BadmintonEnv` per Gym reset and measure the cost.

### 11.5 Milestone 0 status after lifecycle probe

Verified:
- The baseline native API works in isolated Python 3.12.3.
- B-serving reset returns an A receiving state and excludes service from the RL action contract.
- Normal A-controlled steps return five values.
- Two sampled terminal winner records agree with the per-rally score delta.
- Native reset carries score/rally state and output records accumulate.
- Fresh environment construction produces a neutral 0-0 state.

Still open:
- Validate the controlled-reset adapter or approve fresh-object reset.
- Verify deterministic repeated-seed transitions with the selected reset strategy.
- Audit all native terminal branches and resolve one-point winner extraction for every supported terminal path.
- Freeze canonical-to-model-to-native shot mapping.
- Verify the court transform, native coordinate units and action-cell geometry.
- Build a non-mutating full joint action legality evaluator equivalent to the native constraints.
- Complete native data/checkpoint provenance and leakage audit.

**Milestone 0 is still in progress.** Do not create the production worker until the reset strategy, action translation, legality, coordinates and winner paths are resolved.


## 12. Controlled episode-reset probe â€” user-provided report (2026-11-08)

**Runtime probe status: PASS.** The user executed `scripts/setup/probe_coachai_controlled_reset.py` against the exact pinned checkout using Python 3.12.3, PyTorch 2.4.1+cu121 and CUDA (`cuda:0`). The report records two consecutive one-rally episodes on the same native `BadmintonEnv` object.

Observed results:

- Both episodes began at score A=0/B=0, with B serving. Native `reset()` returned four values and `launch=False`, so A entered as receiver.
- Both rallies terminated. Each score delta was A=+1/B=0, and the final `getpoint_player` record agreed with side A.
- The same `BadmintonEnv`, `agent_a`, `agent_b`, `badminton_constraint`, `baseServe` and output DataFrame were retained across episodes.
- Each episode added five stroke records. `close()` wrote a 10-row CSV and reading it back returned all 10 rows.
- Re-seeding Python `random`, NumPy and Torch with seed 42 before each episode produced equal discrete/rounded trajectory signatures in this two-episode test.
- Constructor time was 279.652 ms. The two reset calls were 720.445 ms and 122.163 ms respectively. These are diagnostic measurements from one run, not performance guarantees.

The tested reset adapter explicitly sets:

```python
env.score_a = 0
env.score_b = 0
env.pre_score_a = 0
env.pre_score_b = 0
env.serving_player = "B"
env.rally = 0                 # native reset increments this to 1
env.match = episode_match_id
env.round = 1
env.state = None
env.info_dict = {}
```

It preserves `agent_a`, `agent_b`, `badminton_constraint`, `baseServe` and `output`. The successful test also verifies winner-record correctness after resetting the previous-score pointers, which is important because `Record()` obtains `getpoint_player` through those pointers.

**Decision supported by this probe:** use one persistent native environment per worker and a pinned, explicit one-rally reset adapter for B-serving episodes. Do not reconstruct the model and constraint objects on every Gymnasium reset.

**Limits of this evidence:** only two episodes were run, both deliberately reset with the same seed 42, on one machine/runtime and one player pairing. This does not establish reproducibility for arbitrary seeds, fresh worker processes, CPU/GPU parity or other player identities. Add separate seed-sequence and worker-restart tests later. The output DataFrame grows for the lifetime of the worker; for longer training, monitor its memory use and consider a documented periodic flush strategy without losing the native records.

### 12.1 Controlled-reset status

| Item | Status |
|---|---|
| Neutral 0-0 start on each reset | Runtime verified (2 episodes) |
| B serves on each reset | Runtime verified (2 episodes) |
| Score and previous-score pointers reset together | Runtime verified |
| RallyNet and constraint objects reused | Runtime verified |
| BaseServe and output DataFrame retained | Runtime verified |
| Winner record agrees with per-rally score delta | Runtime verified (2 episodes) |
| Same-seed replay within the same persistent process | Runtime verified for seed 42 (2 episodes) |
| Arbitrary seed sequence / fresh-worker replay | Pending |
| ExternalAgentStub replacing unused A-side RallyNet | Pending runtime probe |
| Long-run output/memory policy | Pending design |

### 12.2 Milestone status after controlled-reset probe

The native runtime and one-rally lifecycle portions of the source audit are now verified. Milestone 0 as a whole is **not yet closed**: the canonical-to-model-to-native family mapping, B2 coordinate conversion, non-mutating full joint action mask, forced-miss handling and exhaustive terminal semantics remain unresolved. Do not implement the production worker until these action/transition contracts are frozen.

## 13. Native shot IDs and constraint semantics â€” pinned-source audit

The pinned source makes three distinct shot identifiers necessary:

1. canonical ShuttleSet family ID (the project's stable 0â€“9 order);
2. RallyNet model output ID (sampled from the model's categorical output);
3. CoachAI native shot ID (used by the environment state and native constraint tables).

At `BadmintonEnv/Utils/Constraint.py` lines 381â€“386, `validate_action()` applies this `t2c` conversion before checking the selected category:

```python
t2c = {1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8,
       7: 6, 8: 7, 9: 3, 10: 5, 11: 11}
```

`BadmintonEnv/Environment.py` lines 47â€“59 names native IDs 1â€“11 as Short Serve, Clear, Push Shot, Smash, Smash Defence, Drive, Net Shot, Lob, Drop, Long Serve and Missed Shot. RallyNet's non-service inference masks model categories 3â€“11; model ID 11 returns `None` (missed shot). Normal B2 decisions must use non-service categories and should not select the missed-shot category.

The native legality checks at `Constraint.py` lines 381â€“480 are:

- category legality: `action_distribution_1` indexed by previous native shot, 24-zone hitting position, longitudinal movement bin and native shot ID;
- landing legality: `action_distribution_2` indexed by native shot ID and 24-zone opponent landing area;
- movement legality: `correct_position()` limits the normalized distance from hitting position to recovery position to 1.6.

`action_distribution_3` is initialized but the pinned `validate_action()` does not use it. `validate_action()` may repair shot type, landing point or recovery point, and can use Python `random` fallbacks. A legal-action mask must therefore implement the checks without mutating the candidate; call `validate_action()` only as a test oracle for candidates the mask already marks legal.

The selected action's hitting position is not a fourth policy decision: the runtime probe showed it matches the current shuttle coordinate in `state[3]`. The worker must derive it from the current state. CoachAI's `check_out_violation()` separately rejects an unreachable contact (`>= 1.6` normalized distance from player to shuttle) and out-of-bounds landing coordinates.

## 14. Proposed canonical-to-CoachAI mapping â€” not frozen

The initial semantic proposal is:

| Canonical family | Canonical ID | RallyNet model ID | CoachAI native ID/name | Proposal |
|---|---:|---:|---|---|
| CLEAR | 0 | 3 | 2 â€” Clear | Supported |
| DROP | 1 | 5 | 9 â€” Drop | Supported |
| SMASH | 2 | 4 | 4 â€” Smash | Supported |
| HALF_SMASH | 3 | â€” | â€” | Unsupported |
| DRIVE | 4 | 7 | 6 â€” Drive | Supported |
| PUSH | 5 | 9 | 3 â€” Push Shot | Supported |
| LIFT | 6 | 6 | 8 â€” Lob | Supported |
| NET_SHOT | 7 | 8 | 7 â€” Net Shot | Supported |
| BLOCK | 8 | â€” | â€” | Unsupported |
| CUT | 9 | â€” | â€” | Unsupported by default; lossy DROP alias would need explicit approval |

This table is a **proposal**, not a frozen mapping file. It derives the model/native ID pair from the pinned `t2c` table and native class names. The canonical semantic correspondences are not all exact and require user approval. In particular, CoachAI's native class list does not expose separate `Wrist smash`, `Return Net` or `Passive Drop` action classes. Do not silently map these to `Smash Defence`, `Net Shot` or `Drop`.

If the user approves the conservative proposal, keep all ten canonical family IDs in the 1,350-action codec and mark unsupported family slices permanently false in B2 v1. This preserves the cross-condition action contract while making the reduced B2 action expressivity explicit. P1/P2 may support those families later.

The next diagnostic is `scripts/setup/probe_coachai_action_mask.py`. It checks the proposed metric-to-native cell conversion, builds a non-mutating full joint mask from the pinned constraints, passes every `True` candidate through `validate_action()` as a no-repair oracle, and executes one actual A-controlled native transition. Its mapping remains labelled proposed until the user approves it.