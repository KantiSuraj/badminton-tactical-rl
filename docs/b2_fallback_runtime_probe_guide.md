# B2 fallback runtime probe — run guide

**Status:** Diagnostic harness only. It does not modify the CoachAI+ checkout, production worker, Gymnasium environment, or PPO collector.

## Repository placement

Copy the two files individually:

```text
badminton-tactical-rl/
├── scripts/
│   └── setup/
│       └── probe_b2_fallback_runtime.py
└── docs/
    └── b2_fallback_runtime_probe_guide.md
```

## Run

From the project root, with the isolated CoachAI+ Python 3.12 environment:

```bash
.venv-coachai/bin/python scripts/setup/probe_b2_fallback_runtime.py
```

The probe verifies the pinned CoachAI+ commit and `Constraint.py` hash before importing the native environment. It constructs a controlled environment with a stub B agent that returns `None` after A's transition. This allows the test to exercise the real `BadmintonConstraint.validate_action()` and `BadmintonEnv.step()` code without loading RallyNet or running a full rally.

Outputs:

```text
reports/b2_fallback_runtime_probe.json
reports/b2_fallback_events_probe.jsonl
```

Pure-Python helper tests can be run without the CoachAI+ checkout:

```bash
.venv-coachai/bin/python scripts/setup/probe_b2_fallback_runtime.py --self-test
```

## Three test cases

| Reason code | Fixture | Expected native behaviour |
|---|---|---|
| `NATIVE_CONSTRAINT_EMPTY` | Previous shot 1, hit zone 11, movement bin 1 | Input a disallowed carrier ID; the pinned validator chooses native ID 2–9 using the scoped fallback RNG and zeros its probability vector |
| `B2_TAXONOMY_GAP` | Previous shot 4, hit zone 5, movement bin 1 | Native ID 5 (`Smash Defence`) is the sole allowed category; its inverse model ID 10 is passed to the native API |
| `NO_JOINT_ACTION_AFTER_GEOMETRY` | Previous shot 1, hit zone 11, movement bin 2 | Use a native-allowed mapped shot with a native landing zone and continuous recovery point outside the policy-grid action record |

The geometry case is fault-injected when the deterministic grid scan finds a feasible canonical-grid action. The report explicitly marks this with `reason_source = synthetic_fault_injection`; it must not be interpreted as evidence that CoachAI+ naturally produced such a state.

## What the probe verifies

1. A deterministic event seed is derived from worker seed, episode index, decision index and fallback counter using a stable digest (not Python's process-randomized `hash()`).
2. The fallback uses its own `random.Random` instance. The `Utils.Constraint.random` binding is replaced only while the matching A-state call to `validate_action()` executes, then restored in `finally`.
3. A one-shot capture wrapper records the actual `(checked_action, native_validator_reward, corrected_flag)` returned from the single validator invocation inside `Environment.step()`.
4. Non-matching states are delegated without the fallback RNG scope. A fake B-state call in the unit test verifies this boundary; the controlled environment uses a no-return B stub, so it does not execute a real B-side validator call.
5. The actual returned native action agrees with the A-side row written to `env.output`.
6. Python, NumPy, Torch CPU and (when available) Torch CUDA global RNG states are unchanged across the fallback transition.
7. The event is written separately from policy actions, with null policy action ID/log-probability and `policy_selected = false`.
8. B2 reward is derived only from the score delta: A win `+1`, B win `-1`, otherwise `0`. CoachAI+'s validator reward is recorded as telemetry only.
9. Same-worker replay and two fresh Python-process replays produce the same corrected native action for the same event identity and seed.
10. The canonical 3×5 landing and 3×3 recovery grids are scanned against the pinned native tables and recovery projection rule to see whether a *natural* geometry-empty context exists under those exact grids.

## Important scope limitations

This is a native-runtime boundary probe, not production integration. In particular, it does **not** yet test:

- the actual CoachAI+ RallyNet agent's hidden RNG state or post-fallback action sequence;
- Gymnasium `reset()` / `step()` auto-advancement through consecutive fallback states;
- a real MaskablePPO call site or rollout buffer;
- the production worker's JSONL transport or lifecycle;
- a full rally in which a real B RallyNet response follows the fallback.

Those require the current worker and Gymnasium source files. Do not treat the current harness's empty local policy buffers as proof that a production PPO collector has been integrated correctly.

## Interpreting `NO_JOINT_ACTION_AFTER_GEOMETRY`

The deterministic geometry scan is important. For the current fixed grids, every mapped family appears to have at least one landing-grid zone accepted by its native landing table, and recovery feasibility can be checked over all 24 hit-zone centres. If the scan reports zero natural geometry-empty contexts, retain this reason code as a defensive invariant, but do not manufacture a production event to exercise it. Its test fixture remains explicitly synthetic.

If the scan finds natural examples, inspect the reported context and family counts before using the branch in the worker.

## Promotion criteria

Only after this probe passes should the fallback logic be ported into the isolated worker. Production integration still needs tests that:

- auto-advance internal native transitions until a valid policy decision or terminal state;
- never call the policy with an all-false mask;
- never add a fallback action or log-probability to PPO's sampled-action records;
- attach the fallback event and internal-transition count to Gymnasium `info`;
- preserve B2's rally-outcome reward independently of native constraint penalties;
- verify replay with the real RallyNet worker and across fresh worker processes.

The canonical 10 × 15 × 9 macro-action space must not gain a fallback action or synthetic valid bit.
