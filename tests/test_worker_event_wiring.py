"""Wiring of rally-event collection into reset()/step() with a fake native env.

The real CoachAI+ environment is exercised by tests/test_b2_worker_integration.py in the
worker interpreter; this test only checks the worker-side control flow and needs NumPy.
"""
from __future__ import annotations

import copy

import pytest

np = pytest.importorskip("numpy")

from tests.b2_trace_fixtures import C_ROWS, C_STATES  # noqa: E402
from badminton_rl.adapters.coachai_worker_utils import N_ACTIONS, encode_action  # noqa: E402
from tests.b2_test_helpers import COLUMNS, FakeFrame, make_worker  # noqa: E402


class FakeEnv:
    """Attempt 0: rally ends during reset (no policy decision). Attempt 1: normal serve."""

    def __init__(self, serve_rows=True, fresh=False):
        self.serve_rows = serve_rows
        # fresh = a brand-new native frame (no columns yet); otherwise stale rows from a prior rally
        self.output = FakeFrame([]) if fresh else FakeFrame(COLUMNS, [C_ROWS[0]] * 7)
        self.score_a = self.score_b = self.pre_score_a = self.pre_score_b = 0
        self.round = 1
        self.match = 1
        self.state = None
        self.info_dict = {}
        self.serving_player = "B"
        self.reset_calls = 0
        self.lengths_seen_by_reset = []

    def reset(self):
        self.lengths_seen_by_reset.append(len(self.output))
        self.reset_calls += 1
        if self.reset_calls == 1:  # no-policy rally: B's serve faults, A wins the point
            if self.serve_rows:
                self.output.append(C_ROWS[0])
            self.score_a += 1
            return C_STATES[1], {"state": copy.deepcopy(C_STATES[:2])}, True, False
        self.output.append(C_ROWS[0])
        return copy.deepcopy(C_STATES[2]), {"state": copy.deepcopy(C_STATES[:2])}, False, False


def build(tmp_path, serve_rows=True, fresh=False):
    worker = make_worker(tmp_path, None)
    worker.np = np
    worker.env = FakeEnv(serve_rows=serve_rows, fresh=fresh)
    worker._episode_seed = 0
    worker._seed_everything = lambda seed: None

    def advance(**kwargs):
        worker._mask = np.zeros((N_ACTIONS,), dtype=np.bool_)
        worker._mask[encode_action(0, 3, 4)] = True
        worker._ready = True
        return 0

    worker._advance_to_policy_state = advance
    worker._observation = lambda: {"stub": True}
    return worker


def test_discarded_retry_attempt_never_leaks_events_into_the_accepted_one(tmp_path):
    worker = build(tmp_path)
    payload = worker.reset({"seed": 5, "options": {"max_reset_retries": 2}})
    env = worker.env
    assert env.lengths_seen_by_reset == [0, 0]  # output cleared before *each* attempt
    assert payload["status"] == "ready_for_policy" and payload["ready_for_policy"] is True
    events = payload["rally_events_delta"]
    assert [(e["event_index"], e["decision_index"], e["ball_round"], e["actor"]) for e in events] == [
        (0, 0, 1, "B")]
    assert events[0]["controller"] == "RALLYNET" and events[0]["event_type"] == "STROKE"
    assert worker._output_cursor == 1 and worker._output_cursor_attempt == 1
    assert len(env.output) == 1 and payload["reset_output_rows_added"] == 1


def test_reset_response_keeps_every_existing_field(tmp_path):
    payload = build(tmp_path).reset({"seed": 5, "options": {"max_reset_retries": 2}})
    for key in ("status", "episode_id", "observation", "action_mask", "action_mask_sha256",
                "ready_for_policy", "terminated", "truncated", "reset_return_length",
                "reset_output_rows_added", "internal_native_transitions", "native_reward_trace",
                "fallback_events", "terminal", "no_policy_rally", "score"):
        assert key in payload
    assert payload["fallback_events"] == [] and payload["terminal"] is None


def test_final_no_policy_attempt_reports_only_its_own_events(tmp_path):
    worker = build(tmp_path)
    payload = worker.reset({"seed": 5, "options": {"max_reset_retries": 0}})
    assert payload["status"] == "terminal_without_policy"
    assert payload["terminal"]["winner_side"] == "A"
    assert [(e["ball_round"], e["actor"]) for e in payload["rally_events_delta"]] == [(1, "B")]
    assert worker.env.lengths_seen_by_reset == [0]  # the 7 stale rows never reached this attempt


def test_a_reset_that_produces_no_output_rows_is_fatal(tmp_path):
    from badminton_rl.adapters.coachai_worker_utils import WorkerFault

    worker = build(tmp_path, serve_rows=False)
    with pytest.raises(WorkerFault) as info:
        worker.reset({"seed": 5, "options": {"max_reset_retries": 0}})
    assert info.value.fatal and info.value.code == "native_state_error"


def test_very_first_reset_on_a_fresh_native_frame_succeeds(tmp_path):
    """Regression for the real-worker failure: no columns exist before the first native reset."""
    worker = build(tmp_path, fresh=True)
    payload = worker.reset({"seed": 5, "options": {"max_reset_retries": 2}})
    assert payload["status"] == "ready_for_policy"
    assert [(e["ball_round"], e["actor"]) for e in payload["rally_events_delta"]] == [(1, "B")]