# B2 Empty-Action-Mask Fallback Contract

**Status:** Design contract for B2 v1; not yet integrated into the worker or Gymnasium environment.

**Purpose:** Define what the environment does when the proposed B2 policy mask contains no legal macro-action. The fallback is an environment-generated transition, not an action selected by the policy.

## 1. Non-negotiable invariants

1. The policy action space remains the canonical `10 x 15 x 9` macro-action space. Do not add a `FALLBACK` family, sentinel action ID, or synthetic legal bit.
2. If the mask is all false, do not call `policy.predict()` / `model.predict()` with that all-false mask.
3. Do not fabricate a policy action ID, policy log-probability, or policy-selected macro-action for the fallback transition.
4. Log fallback events separately from policy decisions.
5. Use a deterministic RNG owned by the isolated CoachAI+ worker. Record the fallback event seed and the selected/corrected native action.
6. Execute through the pinned native environment so CoachAI+ owns the transition, opponent response, score update, and rally record.
7. B2's externally defined rally reward remains `+1` for A's rally win, `-1` for B's rally win, and `0` otherwise. Do not leak CoachAI+'s internal constraint penalty into the B2 reward contract.
8. Never relax the native category, landing, contact, or movement constraints to make the mask non-empty.

## 2. Empty-mask reasons must be classified before fallback

At the current state, compute and retain:

- contact reachability / out-of-zone status;
- the native shot-category vector for `(previous native shot ID, hit zone, y-movement bin)`;
- the subset of that vector represented by the proposed B2 family mapping;
- landing-zone compatibility for the candidate native shot IDs;
- recovery-cell compatibility.

Classify an empty mask as one of:

| Reason | Meaning | Handling |
|---|---|---|
| `contact_unreachable` | Current shuttle contact violates CoachAI+'s reach limit | Do not invent a shot. Use the native miss/terminal path and log `unplayable_state`; no macro-action is emitted. |
| `contact_out_of_zone` | Hit point cannot be classified into the 24 native zones | Treat as a state/coordinate contract failure; fail loudly in development. Do not silently fall back. |
| `native_shot_set_empty` | Native `action_distribution_1` contains no allowed shot category | Trigger the pinned native validator's documented empty-vector fallback using the worker-owned RNG. Capture the returned corrected action. |
| `native_nonempty_b2_unmapped` | Native categories exist, but none maps to one of the seven proposed B2 families | Choose a category only from the native-allowed non-service IDs using the worker RNG; convert that native ID back to its RallyNet/model input ID for the native API. |
| `landing_grid_empty` | A proposed shot category is allowed, but no centre in the 3x5 B2 landing grid survives the native landing table | Use a native-valid landing point outside the policy grid; never mark an invalid B2 landing cell as legal. |
| `recovery_grid_empty` | Shot and landing are legal, but no centre in the 3x3 recovery grid passes the movement constraint | Use a native recovery point obtained by the pinned `correct_position()` rule; do not add a false macro-action bit. |
| `joint_mask_empty_other` | No candidate survives the full combination of checks | Record full diagnostics and use the native-valid fallback construction only if a reachable contact and valid native shot/landing/recovery can be constructed. Otherwise use `unplayable_state`. |

The table-only audit in `audit_coachai_constraint_table.py` determines the first three category-level conditions over all 1,440 contexts. It does not prove that a continuous coordinate or the discretized landing/recovery grids always admit a joint action.

## 3. Native category and landing behavior established from the pinned source

The pinned `BadmintonEnv/Utils/Constraint.py` has ten native output categories (IDs 1–10) and stores `action_distribution_1` as 10 previous-shot IDs x 24 hit zones x 6 movement bins x 10 output categories. The three-way native shot constraint is the intersection of previous-shot, longitudinal-movement, and hitting-zone category lists.

The native `validate_action()` converts the RallyNet/model ID to a native ID first. If the selected category is disallowed and the entire native category vector is empty, it selects `random.choice(range(2, 10))`, clears the probability vector to ten zeros, and flags a constraint correction. The same method then processes landing and recovery corrections. This behavior is native fallback behavior, not a legal B2 macro-action.

The proposed B2 mapping covers native IDs `{2, 3, 4, 6, 7, 8, 9}`. Native ID `5` (`Smash Defence`) is not represented by a proposed B2 family. Do not silently map it to SMASH, DRIVE, or another family.

The pinned landing table has a useful property for the native empty-vector fallback: the intersection of landing zones accepted by every native fallback category 2 through 9 is `{9, 10, 11, 16}`. The audit script recomputes this from `action_distribution_2`; do not hard-code it without asserting the table result at worker startup.

An independent recomputation of the pinned source's three static category lists (`valid_type_to_type`, `valid_y_to_type`, and `hit_loc_to_type`) gives the following expected audit totals. The runtime audit script must confirm these against the imported `action_distribution_1` before they are treated as the checkout's verified result:

| Context classification | Expected contexts | Share of 1,440 |
|---|---:|---:|
| Native shot vector empty | 529 | 36.74% |
| Proposed B2 supported-family subset empty | 552 | 38.33% |
| Native vector non-empty, but no proposed B2 family available | 23 | 1.60% |
| At least one proposed B2 family available | 888 | 61.67% |

The 23 contexts in the third row permit only native shot ID `5` (`Smash Defence`), which has no proposed B2 macro-action mapping. This is why the fallback contract must distinguish `native_shot_set_empty` from `native_nonempty_b2_unmapped`; the native validator's empty-vector fallback only applies to the first case.

## 4. Fallback action construction

This is an internal carrier action used only to invoke the native step/validator. It is **not** in the B2 action space and must never be returned as a policy action.

For a reachable, in-zone state:

- derive hitting position from the current shuttle coordinate (`state[3]`);
- use a centre point from a landing zone validated against `action_distribution_2` for the chosen native category; for native fallback IDs 2–9, choose from their recomputed common-valid zones `{9, 10, 11, 16}`;
- use the current player position (`state[1]`) as a conservative recovery candidate, then independently verify `correct_position(hit, recovery)` leaves it unchanged; if it does not, use the projected result and verify it;
- if the native shot vector is empty, pass a harmless in-bounds carrier model ID so the pinned validator executes its own empty-vector fallback;
- if the native vector is non-empty but has no B2-mapped family, select only from native-allowed IDs 2–9 with the worker RNG, and use the inverse model/native mapping for the input expected by `env.step()`;
- allow `env.step()` to perform the native validation exactly once. Capture the **returned corrected native action tuple** from that validation call; do not call `validate_action()` once for logging and then call `env.step()` (that would consume randomness twice and may produce a different action).

The exact returned native tuple is `(native_shot_id, hitting_position, landing_position, recovery_position, shot_probability)`. Record every field. Also record the validator's native reward and correction flag for diagnostics, but ignore the native reward when computing B2's externally specified rally reward.

If the native category vector is empty, CoachAI+'s validator randomly chooses a native shot ID from 2 through 9. The worker RNG must control that choice. The returned action after any landing/recovery correction—not the carrier action—must be the value recorded as `corrected_native_action`.

## 5. RNG ownership and reproducibility

- The worker owns a dedicated `random.Random` instance; never seed or consume the parent Gymnasium process's RNG for this fallback.
- Derive a fallback-event seed deterministically from the run seed, episode/rally index, decision index, and fallback-event counter. Store the integer seed in the event record.
- Scope the native `Utils.Constraint` module's random calls to this worker-owned RNG during the fallback transition. Restore the module binding afterward. The worker must process one environment transition at a time; do not run this scoped override concurrently in multiple threads.
- Include the actual corrected native action in the event record so replay/audit does not depend only on the seed.
- Test same-worker same-seed replay and a fresh-worker replay separately. Do not claim cross-device bitwise determinism without testing it.

## 6. Capturing the corrected native action exactly once

Do not call native `validate_action()` separately before `env.step()` to discover the corrected action: `Environment.step()` calls it again and would consume a second fallback random draw.

Instead, install a narrow one-shot capture wrapper around the worker's `badminton_constraint.validate_action` call:

1. Arm capture for the current A-controlled state signature immediately before `env.step(carrier_action, launch)`.
2. Delegate to the original `validate_action(state, action)` exactly once.
3. Capture its returned `(checked_action, native_reward, is_constraint)` tuple only for the matching A state, then disarm.
4. Let subsequent B-side validation calls pass through unmodified.
5. Restore the original bound method in `finally`.
6. Assert that the capture occurred. If it did not, mark the fallback event as failed; do not report an action that was not actually passed into the native transition.

The existing `env.output` record can be retained as an independent cross-check of the A-side native shot type and corrected coordinates.

## 7. Event schema

Store fallback records separately, for example `reports/b2_fallback_events.jsonl` or a worker telemetry stream. Do not place them in the policy action trajectory as if they were sampled from the policy.

Recommended record:

```json
{
  "event_type": "b2_empty_mask_fallback",
  "fallback_event_id": "episode-4-decision-2",
  "episode_seed": 45,
  "fallback_seed": 42004502,
  "episode_index": 4,
  "decision_index": 2,
  "state_signature_sha256": "...",
  "empty_mask_reason": "native_shot_set_empty",
  "native_context": {
    "previous_native_shot_id": 1,
    "hit_zone": 11,
    "y_movement_bin": 1,
    "native_allowed_shot_ids": []
  },
  "policy_action_id": null,
  "policy_action_log_probability": null,
  "policy_selected": false,
  "fallback_input_model_shot_id": 3,
  "corrected_native_action": {
    "native_shot_id": 7,
    "hitting_position": [43.8, -214.0],
    "landing_position": [36.75, 195.0],
    "recovery_position": [6.0, -149.7],
    "shot_probability": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "native_validator_reward": -10.0,
  "native_validator_corrected": true,
  "executed_by_native_environment": true
}
```

The numerical values above are illustrative placeholders except for the schema shape; always write actual runtime values. For `native_nonempty_b2_unmapped`, the fallback selection method and actual native-allowed IDs must also be recorded.

## 8. Gymnasium / PPO boundary

A fallback is an environment-generated transition with no policy action. Therefore:

- the policy must only be queried at states with at least one valid B2 macro-action;
- `reset()` and `step()` must internally advance fallback states until the next policy-decision state or rally termination;
- expose fallback events and the number of internally advanced native transitions in `info`, but do not expose a fallback action ID in the policy observation or action mask;
- ensure the rollout collector does not fabricate a policy log-probability or store the fallback as a sampled PPO action.

Internal auto-advancement changes the number of native transitions between policy decisions. For B2 v1's sparse rally-outcome objective, `gamma=1.0` makes reward aggregation straightforward. If a discount factor below 1 is selected, implement and test semi-Markov discounting (`gamma ** internally_advanced_transitions`) or a custom rollout collector; do not silently treat a multi-transition advance as one ordinary discounted step.

A rally that terminates during reset-time fallback before the policy ever receives a decision state must be recorded as a no-policy rally. Do not invent a policy action to fill it.

## 9. Required tests before enabling this path

1. A synthetic empty native shot vector triggers one seeded native fallback and captures one corrected tuple.
2. Repeating the same fallback seed and state in the same worker produces the same corrected native action.
3. A native-nonempty / B2-unmapped context selects only from the native-allowed IDs and does not trigger the native empty-vector fallback.
4. The fallback target zone is checked against all native fallback IDs 2–9 at worker startup.
5. The recovery candidate is verified against `correct_position()` and the contact-reachability check.
6. The native validator is called exactly once for the controlled A action; B-side validation remains unaffected.
7. The event has `policy_action_id = null`, `policy_selected = false`, and is absent from policy action samples/log-probability buffers.
8. The policy is never invoked with an all-false mask.
9. The returned corrected native action agrees with the A-side `env.output` record.
10. B2 reward remains exactly `+1/-1/0` from rally winner; native constraint penalties are telemetry only.
11. Same-seed replay is tested within one worker and across fresh worker processes.

## 10. Implementation order

1. Run `audit_coachai_constraint_table.py` and inspect the full 1,440-context report.
2. Confirm exact counts for native-empty and B2-empty contexts, plus the common landing-zone intersection.
3. Implement the fallback action builder and one-shot native-action capture in the isolated worker.
4. Add unit tests for the nine failure/edge classes above before connecting the path to the Gymnasium environment.
5. Add Gymnasium auto-advance and PPO collector semantics only after the worker-level fallback is tested.

Do not freeze the canonical-to-native mapping merely because the fallback exists. The family mapping remains a separate semantic decision.
