# B2 CoachAI+ worker and parent client

This document describes the first production adapter implementation for the
frozen B2 v1 baseline. It does not implement the Gymnasium wrapper or PPO
collector; those are separate next-layer components.

## 1. Files and responsibilities

| Repository path | Responsibility |
|---|---|
| `src/badminton_rl/adapters/protocol.py` | Standard-library-only JSONL v1 contract, validation and error types |
| `src/badminton_rl/adapters/worker_client.py` | Parent process lifecycle, request correlation, timeouts, stderr capture and mask guard |
| `src/badminton_rl/adapters/coachai_worker.py` | Persistent native CoachAI+ environment, real RallyNet B opponent, canonical action translation, pure action-mask evaluation and fallback execution |
| `configs/action_mappings/coachai_v1.json` | Explicit frozen canonical-family/model/native-shot mapping |
| `tests/test_worker_protocol.py` | Protocol validation regression tests |
| `tests/test_worker_client.py` | Fake-child subprocess lifecycle and failure tests |
| `tests/test_coachai_worker_units.py` | Codec, mapping, seed, classifier, one-shot capture and validator policy unit tests |
| `tests/test_b2_worker_integration.py` | Opt-in tests using the real pinned checkout and RallyNet checkpoint |
| `scripts/setup/smoke_b2_worker.py` | Manual real-worker multi-episode smoke test |

The parent imports only `protocol.py` and `worker_client.py`. The worker imports
CoachAI+, NumPy, pandas and PyTorch inside its own process. Do not import
`coachai_worker.py` from the parent application.

## 2. Runtime assumptions and safety checks

The worker is designed for the following fixed runtime:

- CoachAI+ checkout: `../external/CoachAI-Plus`
- Required commit: `4fd05934f172eec16c74d05c2902983b64c8521e`
- Dedicated interpreter: `.venv-coachai/bin/python`
- Controlled side: A
- Opponent: actual RallyNet model on B
- Serve policy: B serves and A receives
- Native constraints enabled

At startup the worker checks the checked-in `coachai_v1.json` mapping against
the executable action codec and allow-list, then checks the exact Git commit,
a clean CoachAI+ working tree, and the audited SHA-256 values for the
environment's relevant datasets, checkpoints and constraint code. It also
checks the `Environment.py` and `Constraint.py` import origins, native action
names, canonical landing-grid zones and common fallback landing zones. A
mismatch is fatal; the worker does not try to repair or reinterpret an
incompatible checkout.

The worker writes CoachAI+'s native CSV output to a temporary directory and
removes it on shutdown. Fallback events and worker stderr are written outside
the CoachAI+ checkout.

## 3. Protocol v1

Each stdin line is one UTF-8 JSON request. Each stdout line is exactly one JSON
response. Standard output is reserved for protocol messages; diagnostics go
to stderr. One request is outstanding at a time per client.

Every request has exactly these fields:

```json
{
  "protocol_version": 1,
  "request_id": "w1234-r00000001",
  "op": "reset",
  "payload": {"seed": 42, "episode_index": 0}
}
```

Supported operations:

| Operation | Payload | Purpose |
|---|---|---|
| `health` | `{}` | Return runtime identity, source hashes, device and mapping versions |
| `reset` | `seed` required; `episode_index`, `episode_id`, `options` optional | Start an independent B-serves/A-receives rally |
| `step` | `{"action_id": integer}` | Execute one parent-selected A macro-action, followed internally by RallyNet B |
| `close` | `{}` | Flush native output and close the worker |

Successful responses contain `{protocol_version, request_id, ok: true,
payload}`. Error responses contain `{protocol_version, request_id, ok: false,
error: {code, message, fatal, details?}}`. The client validates the request ID
and protocol version before accepting a response.

### Reset payload

```json
{"seed": 42, "episode_index": 0, "options": {"server": "B"}}
```

Only `server: "B"` is currently supported. The worker manually neutralizes
`score_a`, `score_b`, `pre_score_a`, and `pre_score_b`, clears rally history,
and calls native `reset()`. It preserves the native environment, RallyNet,
constraint object, BaseServe object and accumulated output frame across rallies.

### Reset/step state payload

The worker returns:

- `observation`: A-side-local CoachAI native coordinates, player positions,
  shuttle position, previous native shot ID, score and rally-history length;
- `action_mask`: exactly 1,350 Boolean entries;
- `action_mask_sha256`: stable digest of the flattened mask;
- `ready_for_policy`: true only when at least one action is valid;
- `terminated` / `truncated`: mutually exclusive terminal flags;
- `native_reward_trace`: diagnostics only; not the B2 reward;
- `fallback_events`: events executed internally during this operation;
- `terminal`: score-delta winner and B2 `+1/-1` reward when terminal.

A non-policy state must be terminal or truncated and must have an all-false
mask. A policy-ready state must have at least one true mask entry. The parent
client enforces both rules before exposing the state to a caller.

## 4. Canonical action mapping

A canonical action is `(shot_family, landing_cell, recovery_cell)`. The flat
integer is:

```text
action_id = (family_id * 15 + landing_cell_id) * 9 + recovery_cell_id
```

There are 10 canonical family slots, 15 opponent-side landing cells and 9
A-side recovery cells, giving 1,350 possible IDs. B2 v1 supports these families:

| Family ID | Family | RallyNet/model shot ID | CoachAI native shot ID |
|---:|---|---:|---:|
| 0 | CLEAR | 3 | 2 |
| 1 | DROP | 5 | 9 |
| 2 | SMASH | 4 | 4 |
| 4 | DRIVE | 7 | 6 |
| 5 | PUSH | 9 | 3 |
| 6 | LIFT | 6 | 8 |
| 7 | NET_SHOT | 8 | 7 |

`HALF_SMASH` (3), `BLOCK` (8), and `CUT` (9) keep their original canonical
slots but have no valid B2 policy actions. Do not alias them to another shot
family. The code keeps canonical family IDs, RallyNet model shot IDs and
CoachAI native shot IDs as separate namespaces.

## 5. Action mask and normal policy transition

Mask computation is pure with respect to `validate_action()` and random state.
For each reachable native state it intersects:

1. CoachAI's shot-type distribution indexed by previous native shot, hit zone
   and longitudinal movement bin;
2. native landing-zone legality for the selected shot;
3. fixed-point recovery cells under `correct_position()`;
4. CoachAI's native out/reach precheck.

Only full shot/landing/recovery triples are exposed as true entries. The worker
constructs the native five-field action from the selected macro-action and
executes it through the actual `env.step()`. It captures the single A-side
`validate_action()` call and requires normal policy actions to pass unchanged.
The native environment invokes the B RallyNet agent internally; the parent must
not simulate B separately.

## 6. Empty-mask fallback contract

Fallbacks are environment-generated transitions, not policy actions. The three
stable reason codes are:

- `NATIVE_CONSTRAINT_EMPTY`: the native allowed-shot set is empty;
- `B2_TAXONOMY_GAP`: native legal shots exist but none maps to a supported B2
  family (for example native shot ID 5 / model shot ID 10);
- `NO_JOINT_ACTION_AFTER_GEOMETRY`: a supported native shot exists but no
  complete macro-action survives the joint landing/recovery geometry checks.

`CONTACT_UNREACHABLE` uses the native miss/fault path and is logged as
`b2_unplayable_state`; it is not assigned a fake macro-action. A contact point
outside CoachAI's 24-zone system and unknown native-only categories are fatal
compatibility errors, not silently repaired cases.

For each fallback event the worker derives a private deterministic RNG seed
from `(worker_seed, episode_index, decision_index, fallback_counter)` using
BLAKE2b. It uses that `random.Random` instance for carrier selection and scopes
CoachAI's `Constraint` module's `random` binding to that instance only during
the matching A-side validator call. The binding is restored in `finally`.
Nonmatching calls, including B-side validation, delegate to the original
validator with the original module RNG. RNG snapshots are compared immediately
around A validation; RallyNet's later Torch randomness is not misclassified as
fallback RNG consumption.

The worker captures the actual action returned by the native validator during
`env.step()` and verifies that the first new output row with `player == "A"`
matches it. For `NATIVE_CONSTRAINT_EMPTY`, the pinned validator is expected to
correct the carrier shot to a native ID in 2..9 and zero the probability vector.
For taxonomy and geometry fallback, the validator must leave the carrier
unchanged. Every successful event is appended to `reports/b2_fallback_events.jsonl`.
Failed fallback attempts are written with `event_type:
"b2_empty_mask_fallback_failed"` before the worker aborts.

Every fallback event has `policy_action_id: null`, no log probability, and
`policy_selected: false`. It is never added to PPO action or log-probability
buffers. The fallback's effect on the native trajectory and terminal score is
still preserved. Empty masks are resolved inside the worker before a new
policy decision is exposed; an all-false policy mask is never forwarded.

The existing geometry audit did not find a natural geometry-empty context on
the current 3x5/3x3 grid. Keep the reason code as a defensive invariant and
regression-test its classifier/builder through controlled fixtures. Do not
claim natural geometry-empty contexts have been observed unless a later runtime
scan finds one.

## 7. Reward and terminal semantics

The B2 reward is computed only from score delta relative to the current rally's
neutral baseline:

- non-terminal transition: `0.0`;
- terminal A win: `+1.0`;
- terminal B win: `-1.0`.

CoachAI's native scalar reward and the validator reward are retained only as
diagnostics. The worker cross-checks `getpoint_player` from the latest native
record when present. A terminal result that does not add exactly one point to
one side is treated as a native-state error.

## 8. Parent-side lifecycle and retry rules

Create one `WorkerClient` for each native environment instance. It owns one
subprocess and serializes all requests using a lock. `stdout` is consumed by a
reader thread and `stderr` is drained continuously to a log file and a bounded
in-memory tail.

The client rejects masked actions locally. It never retries `step()` after a
timeout, crash, broken pipe or protocol-corruption event because the native
state may already have advanced. The worker is invalidated. A fresh child may
be started only by a subsequent explicit `reset()`, which starts a new rally.
`close()` sends the close operation when possible, then terminates/kills the
child if needed; it is idempotent. Prefer a context manager to guarantee
shutdown.

Do not begin PPO vectorization yet. Start with one worker and validate the
complete lifecycle; each additional environment means another native process
and another RallyNet model instance, with corresponding RAM/VRAM cost.

## 9. Install and run

From the WSL project root:

```bash
source .venv/bin/activate
python -m pip install -e .

# Fast, dependency-light adapter regressions
PYTHONPATH=src pytest -q \
  tests/test_worker_protocol.py \
  tests/test_coachai_worker_units.py \
  tests/test_worker_client.py

# Set the verified external checkout and isolated worker interpreter
export COACHAI_ROOT="$PWD/../external/CoachAI-Plus"
export B2_WORKER_PYTHON="$PWD/.venv-coachai/bin/python"

# Real RallyNet, real native environment, two sequential rallies in one worker
python scripts/setup/smoke_b2_worker.py --episodes 2 --device auto

# Opt-in tests against the actual pinned CoachAI+ and RallyNet checkpoint
PYTHONPATH=src pytest -q tests/test_b2_worker_integration.py
```

To run on CPU, pass `--device cpu`. To force CUDA, pass `--device cuda`. The
smoke runner uses the first valid action in the mask; it is only a transition
and process-lifecycle smoke policy, not a tactical or learning baseline.

## 10. Parent application usage

```python
from pathlib import Path

from badminton_rl.adapters import WorkerClient, WorkerConfig

project_root = Path.cwd()
config = WorkerConfig.from_project(project_root)

with WorkerClient(config) as worker:
    health = worker.start()
    state = worker.reset(seed=42, episode_index=0)

    while state["ready_for_policy"]:
        # Replace this with your masked policy. Never choose an ID where mask[id] is false.
        action_id = next(i for i, valid in enumerate(state["action_mask"]) if valid)
        state = worker.step(action_id)

    # `state` is terminal or truncated. Read state['terminal'] when terminated.
```

The `WorkerClient` itself checks masks and lifecycle state but does not define
a Gymnasium `spaces.Dict`, a Gymnasium `Env`, observation normalization,
reward aggregation outside one rally, or a MaskablePPO rollout buffer. Those
belong to the next adapter layer.

## 11. Test coverage and known limits

The local adapter test suite covers:

- all 1,350 action codec round trips and invalid ID rejection;
- JSONL UTF-8, size, shape, NaN/Infinity, version, request ID and field checks;
- process startup, health, reset, successful step, terminal state and idempotent
  close;
- local masked-action rejection without sending a request;
- no retry after step timeout/crash, explicit reset restart and request
  correlation failure;
- deterministic fallback seed derivation and all three reason classifiers;
- one-shot A-state validation capture, nonmatching delegation and RNG-binding
  restoration even when the validator raises;
- special native-empty validator correction versus unchanged native-allowed
  fallback/normal actions.

The real-runtime tests are intentionally skipped unless `COACHAI_ROOT` and
`B2_WORKER_PYTHON` are set. They load the real RallyNet checkpoint and test
repeated episodes in one worker plus same-seed initialization in two fresh
processes. This execution environment cannot execute those tests because it
does not include the user's WSL CoachAI+ checkout or `.venv-coachai`; the first
local live run remains an acceptance gate. The previous standalone fallback
probe passed its controlled three-reason runtime cases, but that probe used a
stub B agent to isolate the A transition. Do not confuse it with the full real
RallyNet worker test.

Not included yet: Gymnasium wrapper, MaskablePPO collector verification,
real-RallyNet fault-injection tests that force all three fallback branches,
and an observed naturally occurring geometry-empty state. These are explicit
follow-on tasks, not silently assumed to have passed.
