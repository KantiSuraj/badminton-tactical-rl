# B2 reward-integrity deliverables

These files establish the B2 v1 reward contract before the Gymnasium wrapper is added.

## Copy into the project

- `src/badminton_rl/env/__init__.py` → the same path in the project.
- `src/badminton_rl/env/rewards.py` → the same path in the project.
- `tests/test_b2_reward_integrity.py` → the same path in the project.
- `docs/B2_REWARD_INTEGRITY.md` → the same path in the project.

The worker change is supplied separately as a patch:

- `patches/worker_decision_watchdog_fatal.patch`

From the project root, first verify and then apply it:

```bash
git apply --check /path/to/worker_decision_watchdog_fatal.patch
git apply /path/to/worker_decision_watchdog_fatal.patch
```

The patch moves the decision watchdog check until after the worker has resolved the entire policy-decision epoch, including automatic fallbacks. A natural terminal on the boundary is accepted; an unresolved rally raises a fatal `policy_decision_limit_exceeded` error instead of returning a neutral truncation.

## Run the pure reward tests

From the project root, with the project editable-installed or `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src pytest -q tests/test_b2_reward_integrity.py
```

These tests need only Python and pytest. They do not need Gymnasium, PyTorch, or CoachAI+.
