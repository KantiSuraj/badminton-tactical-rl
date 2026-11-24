"""Reset/step cursor lifecycle for rally-event collection.

Uses a tiny DataFrame-like fake so the parent test environment needs no pandas/NumPy
(the real frame lives only in the worker interpreter).
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tests.b2_test_helpers import COLUMNS, FakeFrame, make_worker
from tests.b2_trace_fixtures import C_ROWS, C_STATES
from badminton_rl.adapters.coachai_history_serializer import REQUIRED_OUTPUT_COLUMNS
from badminton_rl.adapters.coachai_worker_utils import WorkerFault, encode_action

def begin(worker, attempt=0):
    worker._clear_native_output()
    worker._begin_event_attempt(attempt)


def serve_ready(worker, attempt=0):
    """Worker state right after a reset: serve row in the output, A's contact state pending."""
    worker.env.output.append(C_ROWS[0])
    worker._info = {"state": copy.deepcopy(C_STATES[:2])}
    worker._state = copy.deepcopy(C_STATES[2])
    worker._record_transition(source="SERVE", decision_index=0, output_start=0, output_end=1)


# ------------------------------------------------------------ initialization
def test_cursor_is_zero_after_clearing_a_stale_20_row_output(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS, [C_ROWS[0]] * 20))
    begin(worker)
    assert len(worker.env.output) == 0
    assert worker._output_cursor == 0
    assert (worker._history_cursor, worker._event_index, worker._last_ball_round) == (0, 0, 0)


def test_cursor_is_zero_for_an_already_empty_output(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    assert worker._output_cursor == 0


def test_required_columns_survive_clearing(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS, [C_ROWS[0]] * 3))
    begin(worker)
    assert worker.env.output.columns == COLUMNS
    assert set(REQUIRED_OUTPUT_COLUMNS) <= set(worker.env.output.columns)


def test_missing_output_is_fatal(tmp_path):
    worker = make_worker(tmp_path, None)
    worker.env = SimpleNamespace()  # no `output` attribute at all
    with pytest.raises(WorkerFault) as info:
        begin(worker)
    assert info.value.code == "native_state_error" and info.value.fatal


@pytest.mark.parametrize("bad", [[], {"a": 1}, "frame", 7])
def test_non_dataframe_output_is_fatal(tmp_path, bad):
    worker = make_worker(tmp_path, bad)
    with pytest.raises(WorkerFault) as info:
        begin(worker)
    assert info.value.code == "native_state_error" and info.value.fatal


def test_fresh_native_frame_without_columns_is_accepted_at_reset(tmp_path):
    """Regression: CoachAI's output frame has no columns until the first row is appended."""
    worker = make_worker(tmp_path, FakeFrame([]))
    begin(worker)
    assert worker._output_cursor == 0 and worker.env.output.columns == []


def test_missing_required_columns_is_fatal_once_rows_exist(tmp_path):
    columns = [c for c in COLUMNS if c != "landing_x"]
    worker = make_worker(tmp_path, FakeFrame(columns))
    begin(worker)  # empty frame: nothing to check yet
    serve_ready(worker)
    worker.env.output.columns = columns
    for row in worker.env.output.rows:
        row.pop("landing_x", None)
    with pytest.raises(WorkerFault) as info:
        worker._collect_rally_events(terminal=False)
    assert "required columns" in str(info.value) and worker._output_cursor == 0


def test_output_that_cannot_be_cleared_is_fatal(tmp_path):
    class _Keep:
        def __init__(self, frame):
            self.frame = frame

        def __getitem__(self, key):  # "clearing" returns the rows unchanged
            return self.frame

    class Stubborn(FakeFrame):
        @property
        def iloc(self):
            return _Keep(self)

    worker = make_worker(tmp_path, Stubborn(COLUMNS, [C_ROWS[0]]))
    with pytest.raises(WorkerFault) as info:
        begin(worker)
    assert "not empty after clearing" in str(info.value)


# ------------------------------------------------------------ delta / commit
def test_reset_delta_contains_exactly_the_generated_rows(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS, [C_ROWS[0]] * 9))
    begin(worker)
    before = len(worker.env.output)
    worker.env.output.append(*[C_ROWS[0]] * 5)
    delta, end = worker._read_output_delta(0)
    assert len(worker.env.output) - before == 5 == len(delta) and end == 5


def test_second_attempt_starts_clean_and_old_rows_are_unreachable(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker, attempt=0)
    worker.env.output.append(C_ROWS[0])
    worker._record_transition(source="SERVE", decision_index=0, output_start=0, output_end=1)
    begin(worker, attempt=1)
    assert worker._output_cursor == 0 and len(worker.env.output) == 0
    assert worker._transition_log == [] and worker._event_index == 0
    delta, end = worker._read_output_delta(1)
    assert len(delta) == 0 and end == 0
    with pytest.raises(WorkerFault):
        worker._read_output_delta(0)  # attempt 0's cursor can no longer be used


def test_output_shrinking_below_the_committed_cursor_is_fatal(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    serve_ready(worker)
    worker._collect_rally_events(terminal=False)
    assert worker._output_cursor == 1
    worker.env.output = FakeFrame(COLUMNS)  # rows vanished
    with pytest.raises(WorkerFault) as info:
        worker._read_output_delta(0)
    assert "shrank" in str(info.value)


def test_cursor_cannot_move_backwards_or_past_the_output(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    worker.env.output.append(C_ROWS[0], C_ROWS[1])
    worker._commit_output_cursor(0, 2)
    with pytest.raises(WorkerFault):
        worker._commit_output_cursor(0, 1)
    with pytest.raises(WorkerFault):
        worker._commit_output_cursor(0, 3)
    with pytest.raises(WorkerFault):
        worker._commit_output_cursor(5, 2)


# ------------------------------------------------------------ collection
def test_failed_serialization_does_not_advance_any_cursor(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    serve_ready(worker)
    worker.env.output.rows[0]["landing_x"] = float("nan")  # corrupt the serve row
    with pytest.raises(WorkerFault) as info:
        worker._collect_rally_events(terminal=False)
    assert info.value.fatal
    assert (worker._output_cursor, worker._history_cursor, worker._event_index,
            worker._last_ball_round) == (0, 0, 0, 0)
    assert len(worker._transition_log) == 1  # nothing was consumed


def test_successful_serialization_advances_every_cursor(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    serve_ready(worker)
    events = worker._collect_rally_events(terminal=False)
    assert [e["ball_round"] for e in events] == [1]
    assert worker._output_cursor == 1 and worker._history_cursor == 2
    assert worker._event_index == 1 and worker._last_ball_round == 1
    assert worker._transition_log == []


def test_step_events_continue_the_indexing_and_ignore_committed_rows(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    serve_ready(worker)
    worker._collect_rally_events(terminal=False)

    worker.env.output.append(C_ROWS[1], C_ROWS[2])
    worker._info = {"state": copy.deepcopy(C_STATES)}
    worker._state = None
    worker._record_transition(source="POLICY", decision_index=1, output_start=1, output_end=3,
                              policy_action_id=encode_action(0, 3, 4))
    events = worker._collect_rally_events(terminal=True)
    assert [e["ball_round"] for e in events] == [2, 3]
    assert [e["event_index"] for e in events] == [1, 2]
    assert worker._output_cursor == 3 and worker._history_cursor == 4 and worker._last_ball_round == 3


def test_collection_without_a_native_history_list_is_fatal(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    serve_ready(worker)
    worker._info = {}
    with pytest.raises(WorkerFault):
        worker._collect_rally_events(terminal=False)
    assert worker._output_cursor == 0


def test_transition_without_reported_rows_is_fatal(tmp_path):
    worker = make_worker(tmp_path, FakeFrame(COLUMNS))
    begin(worker)
    with pytest.raises(WorkerFault):
        worker._record_transition(source="FALLBACK", decision_index=1, output_start=None,
                                  output_end=None, fallback_event_id="x")