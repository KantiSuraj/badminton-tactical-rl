# B2 v1 Reward Integrity and Anti-Reward-Hacking Contract

**Status:** Design gate for the Gymnasium wrapper and MaskablePPO integration  
**Scope:** One rally per Gymnasium episode; controlled side A; CoachAI+ RallyNet controls B; B serves and A receives.

This document deliberately freezes the initial objective before implementing the Gymnasium wrapper. The goal is not merely to make the reward function simple. The transition boundary, forced fallback semantics, watchdog behavior, reset edge cases, logging, and training configuration must all preserve the same objective.

## 1. The objective

B2 v1 optimizes the probability that controlled side A wins a rally against the pinned CoachAI+ RallyNet opponent.

There is one reward source:

\[
R_{\text{rally}} =
\begin{cases}
+1 & \text{verified winner is A},\\
-1 & \text{verified winner is B},\\
0 & \text{rally is not yet terminal}.
\end{cases}
\]

There is no dense shaping reward in B2 v1. In particular, do not reward or penalize distance to a target, shot diversity, shot-family choice, landing-cell choice, recovery-cell choice, rally length, movement, or fallback frequency. These are either proxies for the objective or simulator implementation details and can become exploitable objectives of their own.

The winner must be established from the rally score delta and cross-checked against CoachAI+'s final rally record. The worker already implements score-delta winner resolution. The parent must independently validate the returned metadata rather than trusting an arbitrary scalar.

## 2. One Gymnasium step is one policy decision epoch

A policy-ready state is a worker state with at least one valid action in its 1,350-entry mask.

For one call to `GymnasiumEnv.step(action_id)`, the worker may execute this sequence:

1. The policy-selected A action.
2. CoachAI+'s internal B/RallyNet response.
3. Zero or more environment-generated transitions needed to resolve an empty policy mask, including the three documented fallback reasons and the `CONTACT_UNREACHABLE` native miss/fault path.
4. Stop at the next policy-ready A state or at a verified terminal rally outcome.

The entire sequence is one **decision-epoch transition** from the policy's perspective. It creates exactly one policy action, one policy log-probability, and at most one returned Gymnasium reward.

The number of native transitions is recorded as `internal_native_transitions`. It is diagnostic duration information, not a multiplier for reward and not a number of PPO timesteps.

### Reward aggregation rule

The worker's B2 reward for the completed decision epoch is the rally outcome:

- If the rally remains unresolved: return exactly `0.0`.
- If any transition in the internal sequence resolves the rally for A: return exactly `+1.0`.
- If any transition in the internal sequence resolves the rally for B: return exactly `-1.0`.

Since B2 v1 has no intermediate shaping rewards, this is equivalent to summing the B2 reward of each internal native transition: all non-terminal internal transitions contribute zero, and exactly one terminal transition contributes one `+1` or `-1` reward.

**Do not implement the aggregation by summing the worker's `native_reward_trace`, summing event-log fields, or adding a fallback event's `b2_rally_reward_from_score_delta` to the top-level worker reward.** Those values are audit evidence, not separate reward payments. The terminal result is the sole authoritative reward, and the parent validates it against the score delta.

A fallback that ends the rally therefore gives the preceding policy decision its one terminal rally reward. The fallback itself remains absent from PPO's action and log-probability buffers.

## 3. Forced fallback transitions are not policy decisions

For every `b2_empty_mask_fallback` event:

- `policy_action_id` must be null;
- `policy_action_log_probability` must be null;
- `policy_selected` must be false;
- `policy_invoked` must be false;
- `all_false_mask_passed_to_policy` must be false;
- no policy action counter is incremented;
- no policy log-probability is created;
- the actual corrected native action, fallback reason, deterministic event seed, and state signature remain in the audit log.

The action mask must be nonempty whenever a decision is exposed to MaskablePPO. An all-false mask is never passed to the policy. A policy-selected action that is not valid under the current mask is an error; do not silently replace it with a fallback or another action.

### Can a policy deliberately induce fallbacks?

Potentially, a policy can choose an action whose consequences make a later state empty-mask. The fallback mechanism must not be rewarded as a desirable tactic merely because it happened. B2 v1 therefore does not add a fallback bonus or fallback penalty. Instead, the wrapper and evaluation logs must expose fallback counts, reason codes, preceding policy actions, state signatures, and rally outcomes.

This is an explicit simulator-exploit audit. Run matched-seed evaluations and inspect policies with unusually high fallback rates. If a policy learns to steer into a fallback because the simulator's automatic behavior is exploitable, the affected environment behavior must be repaired or the experiment declared invalid; do not patch it with an arbitrary reward penalty.

## 4. Reset-time terminal rallies have no policy owner

The worker can internally advance from reset to a policy-ready state. In the edge case where reset-time automatic transitions resolve the rally before A receives a policy decision, there is no preceding policy action to which the terminal reward can correctly be attributed.

The Gymnasium wrapper must not:

- fabricate a dummy policy action;
- return a reward from `reset()` (Gymnasium reset has no reward return slot);
- attach that outcome to the first action of a different rally;
- silently discard the outcome from experiment metrics.

Instead, the wrapper should record the complete verified result as a `pre_policy_outcome`, then obtain another fresh rally until it reaches a valid policy-ready state. The reset `info` must report the skipped pre-policy outcomes and their seeds/reasons. Bound the number of consecutive reset skips; exceeding the bound is an environment-integrity failure, not a normal episode result.

Training metrics must count pre-policy outcomes separately from policy-controlled rally outcomes. Evaluation reports must show both (a) policy-controlled rally win rate and (b) all-rally outcome rate including pre-policy outcomes. This prevents reset-time environment behavior from being hidden.

## 5. Watchdog limits are not neutral truncations

The current worker implementation has `max_decisions=64` and returns `truncated=True, reward=0.0` if the rally remains unresolved at the limit. That is unsafe for the objective: a policy could learn to prolong a rally until the cap and avoid receiving a loss. A truncated bootstrap can also assign positive continuation value to stalling.

For B2 v1, a decision limit is a **safety watchdog only**, not a badminton game rule and not a reward-bearing episode ending.

Required behavior:

- The selected watchdog limit must be justified using observed rally-length statistics plus CoachAI rollout stress tests.
- If the limit is reached before natural terminal resolution, the worker must raise a fatal, stable error code such as `policy_decision_limit_exceeded`.
- The worker process/episode is invalidated; the Gymnasium wrapper must propagate the failure.
- The training run must stop and preserve its last valid checkpoint. Do not catch the failure, drop only that episode, assign reward zero, call it a truncation, or automatically restart and continue training.
- The same fail-closed rule applies to `max_internal_transitions` exhaustion, worker timeouts, worker crashes, and protocol violations.

A natural rally terminal reached on the watchdog boundary is still a valid terminal rally and receives its verified `+1` or `-1` reward. The watchdog only fires when the rally remains unresolved.

B2 v1 has no ordinary Gymnasium time-limit truncation. For all accepted transitions, `truncated` must be false. `terminated` is true only after the rally winner is resolved. An external `TimeLimit` wrapper must not be applied.

## 6. Discounting must match the objective

The task objective is rally win probability, not speed of resolution. With a single terminal `+1/-1` reward, `gamma < 1` makes the magnitude of the outcome depend on how many policy decisions it took to resolve the rally. In particular, delaying a negative terminal outcome makes its discounted penalty smaller.

Therefore the initial MaskablePPO configuration should use:

```python
gamma = 1.0
```

This makes the episode return exactly the rally outcome, independent of rally length. The GAE lambda remains an estimator setting, not a reward term. Do not add reward normalization, reward clipping, or reward scaling in the initial experiment.

## 7. Parent-side reward verification

The parent must call a pure validator on every worker `step` response before returning anything to Gymnasium.

For a non-terminal response:

- `terminated is False`;
- `truncated is False`;
- `terminal is None`;
- `reward == 0.0` exactly;
- there is a next policy-ready state and a nonempty action mask.

For a terminal response:

- `terminated is True`;
- `truncated is False`;
- terminal metadata is present;
- `winner_side` is exactly `A` or `B`;
- `score_after - score_before` is exactly one point for the declared winner and zero for the other side;
- `score_delta` agrees with the independently recomputed delta;
- `terminal.b2_rally_reward` agrees with the winner;
- top-level `worker.reward` agrees with the winner-derived reward;
- top-level score agrees with `terminal.score_after`;
- no policy-ready action mask is exposed after termination.

Any disagreement raises `RewardIntegrityError` and aborts the episode. Never guess a winner from native reward, event order, or an incomplete score record.

The helper in `src/badminton_rl/env/rewards.py` implements this validation and intentionally ignores native reward values.

## 8. Reward-hacking threat model and required audits

| Potential exploit or accounting error | Required defense |
|---|---|
| Prolong rally to reach decision cap and avoid a loss | Fatal watchdog error; stop training; no neutral truncation |
| Receive fallback reward twice (event plus terminal payload) | Terminal result is authoritative; event values are consistency checks only |
| Treat native CoachAI reward as B2 objective | Keep `native_reward_trace` diagnostic-only |
| Treat a fallback as a sampled action | Null policy action/log-probability; no PPO buffer entry |
| Select a masked/illegal action and receive a repaired action | Reject the action; no silent correction or reward |
| Trigger fallback because its random behavior is advantageous | Log reason/state/preceding action; monitor rates; matched-seed counterfactual audit |
| Reset-time rally ends before first action and reward is lost or misattributed | Record pre-policy outcome separately; never attach to next rally's first action |
| Early terminal metadata conflicts with score | Cross-check score delta and CoachAI final rally record; fail closed |
| Worker crash/timeout is converted to a zero-reward episode | Propagate fatal error and abort training |
| Discounting rewards short rallies irrespective of win probability | `gamma=1.0` |
| Dense proxy reward creates a loophole | No shaping in B2 v1 |
| Reward normalization/clipping changes the intended return | Disable in the initial experiment |

This is a threat model, not a proof that every simulator exploit is impossible. It makes suspicious behavior observable and prevents the known accounting paths from becoming alternate objectives.

## 9. Required tests before PPO training

1. Pure reward table: non-terminal `0`, A terminal win `+1`, B terminal win `-1`.
2. Reject terminal metadata with missing/unknown winner.
3. Reject score delta that disagrees with winner.
4. Reject terminal reward that disagrees with winner.
5. Reject top-level score that disagrees with terminal score.
6. Reject non-terminal nonzero reward or any terminal metadata.
7. Reject all `truncated=True` worker step responses in B2 v1.
8. Confirm arbitrarily large positive/negative values in `native_reward_trace` do not change B2 reward.
9. Confirm multiple automatic fallback transitions still produce exactly one terminal reward.
10. Confirm fallback events have null policy action and log probability, and are never inserted into policy rollout data.
11. Confirm fallback terminal event audit reward matches, but is not added to, top-level reward.
12. Confirm duplicate fallback terminal reward claims are rejected.
13. Confirm reset-time terminal outcome is logged separately and never assigned to a policy action.
14. Confirm the policy decision watchdog causes a fatal failure, not `truncated=True` or reward zero.
15. Confirm internal-transition watchdog, timeout, crash and protocol failures abort rather than return ordinary experience.
16. Confirm invalid masked actions raise and are not silently replaced.
17. Confirm a second `step()` after terminal is rejected.
18. Confirm every valid rally produces exactly one terminal reward and the sum of Gymnasium rewards for that rally is exactly `+1` or `-1`.
19. Confirm the training configuration uses `gamma=1.0` and no reward normalization/clipping.
20. Run adversarial policies that maximize action count and fallback frequency; any watchdog or reward-integrity event invalidates that training/evaluation run.

## 10. Metrics to retain in every run

At minimum, log:

- policy-controlled completed rallies, wins, losses and win rate;
- pre-policy outcomes and their separate win/loss counts;
- reward sum per completed policy-controlled rally;
- policy decision count distribution (median, p90, p99, maximum);
- native transition count distribution;
- fallback count/rate by reason;
- fallback outcomes by reason and preceding policy action;
- invalid or masked action attempts (must be zero);
- reward-integrity failures (must be zero);
- watchdog failures (must be zero in a valid run);
- truncation count (must be zero in B2 v1);
- worker timeouts/crashes/protocol failures (must be zero in a valid run);
- worker commit, action-map hash, checkpoint hash, episode seed and policy seed.

A high mean return alone is not an acceptance criterion. The primary performance metric is rally win rate on held-out episode seeds, accompanied by the integrity metrics above.