# B2 production fallback regression suite

## Scope

`tests/test_b2_production_fallback.py` runs the actual `CoachAIWorker` class in
fresh `.venv-coachai` subprocesses. Each subprocess calls `worker.initialize()`
and verifies the pinned CoachAI+ commit, the real RallyNet checkpoint hash, and
that `env.agent_b is worker.rallynet` before testing fallback paths.

The suite directly exercises the production methods:

- `CoachAIWorker._advance_to_policy_state()`
- `CoachAIWorker._execute_fallback()`
- `CoachAIWorker._build_fallback_carrier()`
- `CoachAIWorker._execute_a_transition()`
- `OneShotValidationCapture`

It does not replace RallyNet with a stub. It wraps the real B agent's `action`
method only to count calls and determine whether the returned action should
reach CoachAI+'s native B-side validator. The original method is delegated to
unchanged and restored after each controlled transition.

## Controlled cases

| Reason | Fixture source | Expected provenance |
|---|---|---|
| `NATIVE_CONSTRAINT_EMPTY` | Previously observed reachable native state; real mask evaluator | `runtime_mask_evaluation` |
| `B2_TAXONOMY_GAP` | Audit-confirmed previous shot 4 / hit zone 5 / movement bin 1 context; real mask evaluator | `runtime_mask_evaluation` |
| `NO_JOINT_ACTION_AFTER_GEOMETRY` | Audit-confirmed previous shot 1 / hit zone 11 / movement bin 2 context | `synthetic_fault_injection` |

For the geometry case, the test first computes the real mask and asserts that
it is nonempty. It then changes only the evaluated mask to empty and sets the
geometry reason code. This is deliberately fault injection, not a claim that
the currently audited 3x5 landing / 3x3 recovery grid naturally produces an
empty joint action set.

The post-fallback evaluator is allowed to inject a single stop bit only if the
real next-state mask is empty. This isolates one fallback transition per case
and prevents the regression test from recursively testing unrelated fallback
states. The transition into that next state, including real RallyNet's B action,
is native.

## Assertions

For each case the test verifies:

- exactly one matching A-side `validate_action()` call is captured;
- the real RallyNet `agent_b.action()` is invoked exactly once;
- if B's returned action is natively valid, one nonmatching validator call is
  delegated; if B returns `None` or an invalid action, no B validator call is
  expected;
- CoachAI+'s actual A output record matches the captured corrected native action;
- A-side validator RNG isolation is preserved and the original module RNG
  binding and validator method are restored;
- score deltas are either zero for a continuing rally or exactly one point for
  the terminal winner, with the corresponding B2 reward `0`, `+1` or `-1`;
- fallback events contain no policy action ID or log probability and do not
  increment the policy decision counter;
- geometry fallback telemetry explicitly identifies `synthetic_fault_injection`.

The complete three-case sequence is run twice in independent subprocesses.
The test compares the event seed, state signature, fallback carrier selection,
and corrected native five-field action for each reason code.

## Run

From the project root, using the parent `.venv` for pytest:

```bash
source .venv/bin/activate
export COACHAI_ROOT="$PWD/../external/CoachAI-Plus"
export B2_WORKER_PYTHON="$PWD/.venv-coachai/bin/python"

PYTHONPATH=src pytest -vv -s tests/test_b2_production_fallback.py
```

The test is marked `integration` and skips when either environment variable is
missing. Each child process initializes the real checkpoint; expect it to take
longer than the dependency-light unit suite and to require sufficient GPU memory
when `B2_DEVICE=auto` selects CUDA.

Optional overrides:

```bash
export B2_PLAYER_NAME="CHOU Tien Chen"
export B2_OPPONENT_NAME="Kento MOMOTA"
export B2_DEVICE=cpu   # or auto / cuda
```

## Interpretation

A passing suite provides production-method evidence for all three reason codes
under the real pinned worker and RallyNet opponent. It does not demonstrate that
`NO_JOINT_ACTION_AFTER_GEOMETRY` occurs naturally; it explicitly tests the branch
with fault injection. It also does not test Gymnasium reward aggregation or the
MaskablePPO collector, which remain later integration layers.
