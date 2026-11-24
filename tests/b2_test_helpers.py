"""Shared fakes for the worker-side tests (no pandas/NumPy needed).

Keep this file in the same folder as b2_trace_fixtures.py.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from tests.b2_trace_fixtures import C_ROWS
from badminton_rl.adapters.coachai_worker_core import CoachAIWorker

COLUMNS = list(C_ROWS[0].keys())


class _ILoc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, int):  # single row access (used when resolving the terminal)
            return dict(self._frame.rows[key])
        assert isinstance(key, slice), "serializer input must be sliced positionally with .iloc"
        return FakeFrame(self._frame.columns, self._frame.rows[key])


class FakeFrame:
    def __init__(self, columns, rows=()):
        self.columns = list(columns)
        self.rows = [dict(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    @property
    def iloc(self):
        return _ILoc(self)

    def copy(self):
        return FakeFrame(self.columns, self.rows)

    def to_dict(self, orient):
        assert orient == "records"
        return [dict(row) for row in self.rows]

    def append(self, *rows):
        if not self.columns and rows:  # like pandas: columns appear with the first row
            self.columns = list(rows[0].keys())
        self.rows.extend(dict(row) for row in rows)


def make_worker(tmp_path, output):
    args = argparse.Namespace(
        project_root=tmp_path, coachai_root=tmp_path, expected_coachai_commit="x",
        player_name="CHOU Tien Chen", opponent_name="Kento MOMOTA", device="cpu",
        worker_seed=1, max_decisions=64, max_internal_transitions=64,
        fallback_events=tmp_path / "events.jsonl",
    )
    worker = CoachAIWorker(args)
    worker.env = SimpleNamespace(output=output)
    return worker