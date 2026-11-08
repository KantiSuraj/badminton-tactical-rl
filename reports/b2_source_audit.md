# B2 CoachAI+ source audit

**Status: PARTIAL — static inspection of the pinned upstream source only. Not cleared for worker implementation.**

- Project: `badminton-tactical-rl`
- Required upstream pin: `4fd05934f172eec16c74d05c2902983b64c8521e`
- Audit basis: source files at the exact commit above, inspected through their immutable GitHub raw URLs.
- Local checkout verification: user-provided terminal output confirms `HEAD` and the resolved required commit both equal `4fd05934f172eec16c74d05c2902983b64c8521e`. `git status --short` reports one untracked generated file: `output_data/output_game_1.csv`; therefore the checkout is not yet clean.
- Runtime probe: **not performed**. Static findings below still require reproduction in the isolated Python 3.12 worker environment.

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
| Constructor signature | `BadmintonEnv(player_agent, opponent_agent, player_name, opponent_name, episodes, is_match, is_constraint, is_have_serve_state, filepath)`; all parameters are required (Environment.py lines 7–18). | **Static source verified** |
| Player A/B assignment | `player_agent` is stored as `agent_a`, `opponent_agent` as `agent_b` (Environment.py lines 19–28). `RunEnv.main` constructs the player1 agent as `player_agent`, player2 as `opponent_agent` (RunEnv.py lines 19–60). | **Static source verified** |
| Is the opponent called internally? | Yes. After A's legal action, `Environment.step` calls `self.agent_b.action(..., launch=False)` (Environment.py lines 278–288). When B serves, `reset()` calls `agent_b.action(..., launch=True)` (lines 135–143). | **Static source verified** |
| Exact reset tuple | A-serving reset returns `(state, info, False, True)` (Environment.py lines 144–151). B-serving reset performs B's serve through `step()` and returns that branch's four values (lines 135–143). | **Static source verified; runtime probe pending** |
| Exact step tuple | The normal A-controlled path returns `(state, reward, info, done, False)`; the B-serving launch branch returns `(state, info, done, False)` (Environment.py lines 157–223 and 238–258). Thus the native API is path-dependent. | **Static source verified; runtime probe pending** |
| Meaning of `launch` | It is a Boolean `is_launch`, not a serialized launch-state object. `True` denotes the service action; after the serve, the returned value is `False` (Environment.py lines 152–158, 209–223, 354–363). | **Static source verified** |
| Who creates serve state? | `is_have_serve_state=True` calls the serving side's `serve_state()`; otherwise reset uses `BaseServe.serve_state()` (Environment.py lines 114–129). `BaseServe` reads `./input_data/All_dataset.csv` (BaseServe.py lines 4–12) and samples a stored serve state using NumPy RNG (lines 123–128). | **Static source verified; seed/runtime probe pending** |
| State fields and units | State is a 4-tuple: previous/opponent shot category, own position, opponent position, shuttle position. RallyNet translates these values using the native `177.5` and `240` normalizers (RallyNet.py lines 256–318 and 320–338); `Environment.py` flips coordinates when changing hitter perspective (lines 209–223 and 278–285). | **Field order inferred from source use; exact semantic/unit contract and round-trip probe pending** |
| Native action fields | RallyNet returns `(output_shot, raw_states[-1], output_land, output_move, normalized_tuple)` (RallyNet.py lines 438–466): shot ID, hitting position, landing position, recovery/movement target, and a 10-element shot probability vector. | **Static source verified** |
| Native action mask | No public full-joint mask method was found in the inspected environment/constraint path. `validate_action` is a mutating repair routine, not a pure legality predicate. | **Static source verified; adapter design required** |
| Shot constraints | `validate_action` converts the model output shot ID through `t2c`, then checks an empirical distribution keyed by previous shot, hitting block and binned player-y movement (Constraint.py lines 381–425). | **Static source verified** |
| Landing constraints | It maps the landing coordinate to a native area and moves an invalid landing to a nearest valid area, with a random fallback if no valid area exists (Constraint.py lines 447–466). | **Static source verified** |
| Recovery constraints | It calls `correct_position`; recovery distance is normalized by 177.5 and 240 and capped at `max_total=1.6` (Constraint.py lines 314–339 and 468–480). | **Static source verified** |
| Native reward meaning | `Reward_function` adds/subtracts constraint and win/loss penalties; it is not simply a terminal ±1 reward (RewardFunction.py lines 0–20). | **Static source verified** |
| Rally winner | On A fault the environment increments `score_b`; on B fault it increments `score_a` (Environment.py lines 227–258 and 290–325). `info_dict['env_score']` stores `[score_a, score_b]` (lines 398–400). `check_winner()` is a game-level 21-point/30-point check (lines 403–414), not a per-rally winner function. | **Source path identified; runtime terminal probes required** |
| Constraint data | `BadmintonConstraint` defaults to `./input_data/All_dataset.csv` (Constraint.py lines 5–6); `BaseServe` uses the same CSV by default (BaseServe.py lines 4–12). | **Static source verified; hash/provenance pending** |
| RallyNet checkpoint/data | `RunEnv.py` reads `input_data/target_players_ids.pkl`, `BadmintonEnv/Agent/hyperparameters.pkl` and loads `RallyNet_weight.trc` (lines 8–35 and 37–54). RallyNet constructor loads `INV_weight.pth` (RallyNet.py lines 167–187). Serving inference loads `BC4Serve_weight.pth` (lines 425–435). | **Static source verified; exact local files/hashes/provenance pending** |
| RallyNet history | `RallyNet.action` receives `info` and indexes its `round`, `player`, `state` and `action` history (RallyNet.py lines 387–400; feature translation at lines 276–318). | **Static source verified** |
| Determinism | Inference is stochastic: latent sampling uses `torch.randn_like`; landing/recovery sample Gaussian-mixture components and Gaussian noise; shot category is sampled from `Categorical` (RallyNet.py lines 411–420 and 449–465). Serve sampling uses NumPy RNG; constraint fallback uses Python `random`. | **Static source verified; complete RNG control/reproducibility pending** |

## 3. Findings that change the adapter design

### 3.1 Use a worker-owned Boolean `launch` state

The native `launch` value is a Boolean indicating whether the controlled action is a serve. The worker should own it and pass it unchanged to the native `env.step(action, is_launch)` call. It is not an opaque physics object.

### 3.2 Reset must normalize multiple native paths

The native `reset()` is always a four-value return, but the values differ by serving side. If A serves, A is expected to make a service action. If B serves, `reset()` itself calls RallyNet and steps B's service action before returning. This means the worker's `reset` operation must normalize both cases and must handle a possible terminal B-service fault before exposing an observation/mask. The parent must not assume that every `reset()` produces a live decision state.

### 3.3 Candidate service policy: let B serve so A receives

The canonical 10-family action taxonomy intentionally excludes service. The cleanest candidate for B2 v1 is therefore to have B serve and A receive: after constructing the native environment, set `serving_player = "B"` before calling `reset()`. This uses an internal field assignment rather than a documented reset option; the exact behavior must be runtime-tested and logged. It must not be hidden inside a random worker policy.

`LAUNCH_BY_BC` is `True` in RallyNet.py (line 14), and its launch path uses `BC4Serve_weight.pth` (lines 425–435). Thus B's service decision is performed by CoachAI+'s existing serve model, not by the RL action space.

### 3.4 One-rally Gym episodes need an explicit score reset policy

`Environment.reset()` increments the rally number but does not reset `score_a`, `score_b`, or `serving_player` (Environment.py lines 111–151). Terminal paths update the score. `Record()` calls `reset_for_next_game()` (lines 437–483), and the latter resets scores only after `check_winner()` reports a game win and `is_match` is true (lines 403–426). Consequently, calling native `reset()` between Gym episodes does not create independent neutral-score rallies. Before implementation, choose and test one explicit policy: reset score/serve fields under worker ownership, reconstruct the environment while reusing loaded model instances, or deliberately treat episodes as sequential points with score context. The choice must agree with the guide's one-rally and reset-context contract.

### 3.5 Do not call `validate_action` to build the mask

The function returns a possibly rewritten action, a penalty/reward and an `isConstraint` flag. It may select a random fallback shot or landing area, change the landing point to a nearest valid area, and clip recovery. Calling it for every candidate would mutate/repair the candidate and can consume RNG; it would not answer whether the original canonical macro-action is jointly legal. Implement a separate deterministic, non-mutating reference legality evaluator from the audited constraint arrays and native court/out checks. Test that every action accepted by the mask is passed through unchanged by the native validator.

### 3.6 Investigate probability-vector indexing before native action construction

`RallyNet.action` returns the shot probability vector using output slots `1..10` (RallyNet.py lines 453–459). `validate_action` remaps `action[0]` through `t2c` (Constraint.py lines 381–386), and in its correction path indexes that remapped category into the probability vector (lines 412–445). The source therefore presents a possible class-order alignment issue when a shot is corrected. Do not paper over it with a guessed one-hot vector. Build a tiny controlled probe that prints (a) sampled model output ID, (b) mapped native category, (c) probability-vector order and (d) the validator's correction result.

### 3.7 Canonical shot taxonomy cannot be frozen by name similarity

The project's own `src/badminton_rl/data/shot_label_mapping_v1.yaml` defines 10 canonical groups:

`CLEAR, DROP, SMASH, HALF_SMASH, DRIVE, PUSH, LIFT, NET_SHOT, BLOCK, CUT`.

CoachAI+'s native action categories instead include short service, clear, push shot, smash, smash defence, drive, net shot, lob, drop and long service (Constraint.py lines 7–18). The native action ID bridge is `t2c` at lines 381–386. There is no distinct native `HALF_SMASH` or `CUT` category, and `BLOCK`'s source label `return net` does not appear as a native `type_mapping` action class. The fact that source annotation maps Chinese `切球` to `Drop` does not prove the separate canonical `CUT` family (`passive drop`) should also collapse to `Drop`.

**Proposed conservative B2 v1 policy for user approval:** map the directly represented families `CLEAR`, `DROP`, `SMASH`, `DRIVE`, `PUSH`, `LIFT` (to native `Lob`), and `NET_SHOT`; mark `HALF_SMASH`, `BLOCK`, and `CUT` explicitly unsupported in B2 v1 rather than silently collapse their semantics. The 1350 action IDs can remain stable, while mask entries for unsupported families remain false. Do not create/freeze `coachai_v1.yaml` until this policy is approved.

## 4. Runtime and reproducibility blockers

1. **Local checkout:** verify exact `HEAD`, tag resolution and clean status on the user's machine.
2. **Python 3.12 worker:** create the separate `.venv-coachai` environment from `requirements-coachai.txt`, record its explicit interpreter path, and prove imports and native behavior using the runtime probe. The user has explicitly approved this runtime change; compatibility remains pending.
3. **Model/data files:** verify every active model/data file exists and hash it. The project manifest contains hashes for several files, but the source path shows the worker also needs `INV_weight.pth` and `BC4Serve_weight.pth`; those hashes must be checked against the user's checkout. `BC_weight.pth` is commented out in the active `RunEnv.main` RallyNet path (RunEnv.py lines 12–17), so it should not be required unless a different agent path is selected.
4. **CPU loading:** `RallyNet.__init__` loads `INV_weight.pth` without `map_location` (RallyNet.py lines 184–187), and the serving branch loads `BC4Serve_weight.pth` without `map_location` (lines 426–429). The existing baseline wrapper patches `torch.load` defaults to CPU when CUDA is unavailable. Reproduce and test the same behavior inside the isolated worker without editing CoachAI+ source.
5. **RNGs:** explicitly seed Python `random`, NumPy and Torch before any serve/model/constraint sampling. Test whether `torchsde` with `bm=None` is reproducible under the pinned runtime. Record any hardware-specific non-determinism.
6. **Output path:** native `close()` writes `self.output` to the constructor's `filepath` (Environment.py lines 485–486). The worker must pass a worker-owned writable temporary/output path and must not pollute the external checkout.
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
| Verified by runtime probe | **None yet** |
| Frozen by project guide | Python 3.12 parent / separate Python 3.12 worker venv split (user-approved); one Gym episode per rally; canonical 10-family action semantics; shared ±1/0 reward; no edits to CoachAI+ source |
| Proposed, requires approval | B-serve/A-receive start; unsupported status for `HALF_SMASH`, `BLOCK`, `CUT`; per-episode score reset policy |
| Unresolved | Exact local checkout cleanliness (one untracked generated CSV remains), Python 3.12 worker environment creation/import probe, runtime tuple/state probes, coordinate transform, mask equivalence, winner terminal probes, RNG reproducibility, all-data leakage audit |

**Milestone 0 is not yet passed.** Static source inspection is useful preparation, but the guide requires the local pinned checkout and a Python 3.12 isolated-worker runtime probe before the worker or action/observation contracts are implemented.


## 7. Local verification update — user-provided terminal output (2026-11-08)

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
