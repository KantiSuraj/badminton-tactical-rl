"""Native validator interception and RNG isolation for the B2 CoachAI+ worker.

NumPy and PyTorch are never imported here: the worker passes the module
objects in, so importing this file does not load the heavyweight stack.
This module must not import `coachai_worker_core`.
"""
from __future__ import annotations

import random
from typing import Any

from badminton_rl.adapters.coachai_worker_utils import (
    WorkerFault,
    _state_signature,
    actions_equal,
)


class OneShotValidationCapture:
    """Capture exactly one A-side validator result; optionally scope fallback RNG.

    The wrapper is installed only around a single native `env.step`.  B-side
    validator calls see a nonmatching A-state signature and delegate to the
    original method with the native module RNG untouched.  RNG snapshots are
    taken immediately around the matching validator call, before RallyNet's B
    inference can consume its ordinary process RNG streams.
    """

    def __init__(self, *, constraint: Any, constraint_module: Any, target_state: Any,
                 torch_module: Any, np_module: Any, event_rng: random.Random | None = None,
                 target_action: Any = None):
        self.constraint = constraint
        self.constraint_module = constraint_module
        self.target_signature = _state_signature(target_state)
        self.target_action = target_action
        self.torch = torch_module
        self.np = np_module
        self.event_rng = event_rng
        self.original = constraint.validate_action
        self.captured_result: Any = None
        self.matching_calls = 0
        self.nonmatching_calls = 0
        self.rng_isolation: dict[str, bool] | None = None
        self.installed = False

    def __enter__(self) -> "OneShotValidationCapture":
        def wrapper(state_arg: Any, action_arg: Any) -> Any:
            matches = _state_signature(state_arg) == self.target_signature
            if (
                matches
                and self.matching_calls == 1
                and self.target_action is not None
                and not actions_equal(action_arg, self.target_action)
            ):
                # B's local-frame state can occasionally collide with A's state signature
                # on symmetric rallies, but B's RallyNet action will not match A's
                # proposed one-hot carrier/policy action.
                matches = False
            if not matches:
                self.nonmatching_calls += 1
                return self.original(state_arg, action_arg)
            if self.matching_calls:
                raise WorkerFault(
                    "native_state_error",
                    "More than one native validator call matched the controlled A state in one transition",
                    fatal=True,
                )
            self.matching_calls += 1
            before = self._snapshot_rng()
            previous_random_binding = self.constraint_module.random
            try:
                if self.event_rng is not None:
                    self.constraint_module.random = self.event_rng
                result = self.original(state_arg, action_arg)
            finally:
                self.constraint_module.random = previous_random_binding
                after = self._snapshot_rng()
                self.rng_isolation = self._compare_rng(before, after)
            self.captured_result = result
            return result

        self.constraint.validate_action = wrapper
        self.installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.installed:
            self.constraint.validate_action = self.original
            self.installed = False

    def _snapshot_rng(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": self.np.random.get_state(),
            "torch_cpu": self.torch.random.get_rng_state().clone(),
        }
        if self.torch.cuda.is_available():
            snapshot["torch_cuda"] = [state.clone() for state in self.torch.cuda.get_rng_state_all()]
        else:
            snapshot["torch_cuda"] = None
        return snapshot

    def _compare_rng(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
        np_before, np_after = before["numpy"], after["numpy"]
        numpy_same = (
            np_before[0] == np_after[0]
            and self.np.array_equal(np_before[1], np_after[1])
            and np_before[2:] == np_after[2:]
        )
        torch_same = bool(self.torch.equal(before["torch_cpu"], after["torch_cpu"]))
        before_cuda, after_cuda = before["torch_cuda"], after["torch_cuda"]
        cuda_same = True
        if before_cuda is not None or after_cuda is not None:
            cuda_same = (
                before_cuda is not None and after_cuda is not None
                and len(before_cuda) == len(after_cuda)
                and all(self.torch.equal(a, b) for a, b in zip(before_cuda, after_cuda))
            )
        python_same = before["python"] == after["python"]
        return {
            "python_global_rng_unchanged": python_same,
            "numpy_global_rng_unchanged": bool(numpy_same),
            "torch_cpu_rng_unchanged": torch_same,
            "torch_cuda_rng_unchanged": bool(cuda_same),
            "all_observed_global_rngs_unchanged": bool(python_same and numpy_same and torch_same and cuda_same),
        }