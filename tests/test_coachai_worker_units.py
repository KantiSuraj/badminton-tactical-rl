"""Pure B2 worker contract tests; no CoachAI checkout or model is required."""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from badminton_rl.adapters.coachai_worker import (
    FAMILY_MAP,
    MODEL_TO_NATIVE,
    NATIVE_FALLBACK_IDS,
    N_ACTIONS,
    REASON_GEOMETRY_EMPTY,
    REASON_NATIVE_EMPTY,
    REASON_TAXONOMY_GAP,
    CoachAIWorker,
    OneShotValidationCapture,
    _build_candidate,
    _landing_centre_m,
    _metres_to_native,
    _recovery_centre_m,
    classify_empty_mask_reason,
    decode_action,
    derive_fallback_seed,
    encode_action,
)
from badminton_rl.adapters.coachai_worker import WorkerFault


def test_canonical_codec_roundtrips_all_1350_actions():
    seen = set()
    for action_id in range(N_ACTIONS):
        parts = decode_action(action_id)
        assert encode_action(*parts) == action_id
        seen.add(parts)
    assert len(seen) == N_ACTIONS == 10 * 15 * 9
    assert encode_action(np.int64(2), np.int64(3), np.int64(4)) == encode_action(2, 3, 4)
    assert decode_action(np.int64(23)) == decode_action(23)


@pytest.mark.parametrize("parts", [
    (-1, 0, 0), (10, 0, 0), (0, -1, 0), (0, 15, 0), (0, 0, 9),
    (True, 0, 0), (0, 1.0, 0), (0, 0, False),
])
def test_codec_rejects_invalid_components(parts):
    with pytest.raises(ValueError):
        encode_action(*parts)


@pytest.mark.parametrize("action_id", [-1, 1350, True, 1.5, "4"])
def test_decode_rejects_invalid_action_id(action_id):
    with pytest.raises(ValueError):
        decode_action(action_id)


def test_fallback_seed_is_repeatable_and_coordinates_change_it():
    a = derive_fallback_seed(42, 4, 1, 0)
    assert a == derive_fallback_seed(42, 4, 1, 0)
    assert a != derive_fallback_seed(42, 4, 1, 1)
    assert a != derive_fallback_seed(42, 5, 1, 0)
    assert 0 <= a < 2**64


def test_empty_mask_reason_classifier_priority_and_all_three_codes():
    mapped = {2, 3, 4, 6, 7, 8, 9}
    assert classify_empty_mask_reason(
        contact_reachable=False, contact_zone_valid=False,
        native_allowed_ids=[], mapped_native_ids=mapped,
    ) == "CONTACT_OUT_OF_ZONE"
    assert classify_empty_mask_reason(
        contact_reachable=False, contact_zone_valid=True,
        native_allowed_ids=[], mapped_native_ids=mapped,
    ) == "CONTACT_UNREACHABLE"
    assert classify_empty_mask_reason(
        contact_reachable=True, contact_zone_valid=True,
        native_allowed_ids=[], mapped_native_ids=mapped,
    ) == REASON_NATIVE_EMPTY
    assert classify_empty_mask_reason(
        contact_reachable=True, contact_zone_valid=True,
        native_allowed_ids=[5], mapped_native_ids=mapped,
    ) == REASON_TAXONOMY_GAP
    assert classify_empty_mask_reason(
        contact_reachable=True, contact_zone_valid=True,
        native_allowed_ids=[3, 7], mapped_native_ids=mapped,
    ) == REASON_GEOMETRY_EMPTY
    assert classify_empty_mask_reason(
        contact_reachable=True, contact_zone_valid=True,
        native_allowed_ids=[10, 11], mapped_native_ids=mapped,
    ) == "UNSUPPORTED_NATIVE_CATEGORY_SET"


def test_action_mapping_preserves_separate_family_model_and_native_ids():
    assert set(FAMILY_MAP) == {0, 1, 2, 4, 5, 6, 7}
    assert set(FAMILY_MAP) | {3, 8, 9} == set(range(10))
    assert [FAMILY_MAP[i]["model_id"] for i in sorted(FAMILY_MAP)] == [3, 5, 4, 7, 9, 6, 8]
    assert [FAMILY_MAP[i]["native_id"] for i in sorted(FAMILY_MAP)] == [2, 9, 4, 6, 3, 8, 7]
    assert MODEL_TO_NATIVE[10] == 5  # native-only Smash Defence taxonomy gap
    assert NATIVE_FALLBACK_IDS == tuple(range(2, 10))



def test_checked_in_mapping_config_matches_executable_worker_contract(tmp_path):
    import shutil

    source_root = Path(__file__).resolve().parents[1]
    target = tmp_path / "configs" / "action_mappings" / "coachai_v1.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(source_root / "configs" / "action_mappings" / "coachai_v1.json", target)
    worker = CoachAIWorker.__new__(CoachAIWorker)
    worker.project_root = tmp_path
    worker._action_mapping_hash = ""
    worker._verify_action_mapping_file()
    assert len(worker._action_mapping_hash) == 64


def test_worker_fails_closed_when_mapping_config_drifts(tmp_path):
    import json

    target = tmp_path / "configs" / "action_mappings" / "coachai_v1.json"
    target.parent.mkdir(parents=True)
    document = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "action_mappings" / "coachai_v1.json").read_text()
    )
    document["supported_families"][0]["coachai_native_shot_id"] = 99
    target.write_text(json.dumps(document), encoding="utf-8")
    worker = CoachAIWorker.__new__(CoachAIWorker)
    worker.project_root = tmp_path
    with pytest.raises(WorkerFault, match="does not match the worker's frozen contract"):
        worker._verify_action_mapping_file()

def test_candidate_uses_five_field_native_action_and_one_hot_model_probability():
    action = _build_candidate(8, (1.0, -2.0), (3.0, 4.0), (5.0, -6.0))
    assert len(action) == 5
    assert action[0] == 8
    assert action[1:4] == ((1.0, -2.0), (3.0, 4.0), (5.0, -6.0))
    assert tuple(action[4]) == tuple([0.0] * 7 + [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        _build_candidate(11, (0, 0), (0, 0), (0, 0))


def test_grid_cell_centres_are_distinct_and_convert_to_native_units():
    landing = [_metres_to_native(_landing_centre_m(i)) for i in range(15)]
    recovery = [_metres_to_native(_recovery_centre_m(i)) for i in range(9)]
    assert len(set(landing)) == 15
    assert len(set(recovery)) == 9
    assert all(y > 0 for _, y in landing)
    assert all(y < 0 for _, y in recovery)
    assert landing[0] == pytest.approx((-85.0, 33.0))
    assert recovery[0] == pytest.approx((-85.0, -55.0))


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _FakeTensor(self.value)


class _FakeTorchRandom:
    def __init__(self):
        self.state = 0

    def get_rng_state(self):
        return _FakeTensor(self.state)


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    def __init__(self):
        self.random = _FakeTorchRandom()
        self.cuda = _FakeCuda()

    @staticmethod
    def equal(left, right):
        return left.value == right.value


class _FakeNumpyRandom:
    @staticmethod
    def get_state():
        return ("fake", [1, 2, 3], 4, 5, 6)


class _FakeNumpy:
    random = _FakeNumpyRandom()

    @staticmethod
    def array_equal(a, b):
        return list(a) == list(b)


class _FakeConstraint:
    def validate_action(self, state, action):
        return action, 0.0, False


def _tiny_state(x=1.0):
    return [1, [1.0, 2.0], [3.0, 4.0], [x, 6.0]]


def test_one_shot_capture_scopes_rng_only_for_matching_a_state_and_restores_binding():
    constraint = _FakeConstraint()
    module = SimpleNamespace(random=random)
    original_random = module.random
    capture = OneShotValidationCapture(
        constraint=constraint, constraint_module=module, target_state=_tiny_state(),
        torch_module=_FakeTorch(), np_module=_FakeNumpy(), event_rng=random.Random(100),
    )
    with capture:
        constraint.validate_action(_tiny_state(x=2.0), ("B",))  # delegated, not scoped
        assert module.random is original_random
        result = constraint.validate_action(_tiny_state(), ("A",))
        assert result == (("A",), 0.0, False)
        assert module.random is original_random
    assert capture.matching_calls == 1
    assert capture.nonmatching_calls == 1
    assert capture.rng_isolation["all_observed_global_rngs_unchanged"] is True
    assert constraint.validate_action.__func__ is _FakeConstraint.validate_action


def test_one_shot_capture_restores_native_random_binding_when_validator_raises():
    class BrokenConstraint:
        def validate_action(self, state, action):
            raise RuntimeError("validator exploded")

    constraint = BrokenConstraint()
    module = SimpleNamespace(random=random)
    original_random = module.random
    capture = OneShotValidationCapture(
        constraint=constraint, constraint_module=module, target_state=_tiny_state(),
        torch_module=_FakeTorch(), np_module=_FakeNumpy(), event_rng=random.Random(2),
    )
    with pytest.raises(RuntimeError, match="exploded"):
        with capture:
            constraint.validate_action(_tiny_state(), ("A",))
    assert module.random is original_random
    assert constraint.validate_action.__func__ is BrokenConstraint.validate_action


def test_native_empty_validator_is_the_only_allowed_corrective_mutation():
    proposal = _build_candidate(3, (1, 2), (3, 4), (5, 6))
    corrected = _build_candidate(7, (1, 2), (3, 4), (5, 6))
    corrected = (7, corrected[1], corrected[2], corrected[3], [0.0] * 10)
    CoachAIWorker._validate_captured_a_action(
        fallback_reason=REASON_NATIVE_EMPTY,
        proposed_action=proposal,
        checked_action=corrected,
        validator_reward=-11.0,
        validator_corrected=True,
    )

    for reason in (None, REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY):
        with pytest.raises(WorkerFault):
            CoachAIWorker._validate_captured_a_action(
                fallback_reason=reason,
                proposed_action=proposal,
                checked_action=corrected,
                validator_reward=-11.0,
                validator_corrected=True,
            )


def test_taxonomy_and_geometry_fallbacks_must_pass_validator_unchanged():
    action = _build_candidate(8, (1, 2), (3, 4), (5, 6))
    # CoachAI's validator maps model shot 8 to native shot 7 before returning.
    native_action = (7, action[1], action[2], action[3], action[4])
    for reason in (REASON_TAXONOMY_GAP, REASON_GEOMETRY_EMPTY, None):
        CoachAIWorker._validate_captured_a_action(
            fallback_reason=reason,
            proposed_action=action,
            checked_action=native_action,
            validator_reward=0.0,
            validator_corrected=False,
        )
