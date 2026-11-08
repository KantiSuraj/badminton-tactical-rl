"""Split integrity — no player overlap, no match leakage, deterministic under seed."""
from pathlib import Path

import pandas as pd
import pytest


def _events(tmp_path):
    # Build from the same synthetic fixture used by test_data_events.
    from tests.test_data_events import _make_fake_dataset
    from badminton_rl.data.build_events import build_events

    return build_events(root=_make_fake_dataset(tmp_path), output_path=tmp_path / "out")


def _synthetic_splits(events, seed=42):
    from badminton_rl.data.splits import build_splits

    # The tiny fixture cannot satisfy the real test_player sample-size gate.
    # Disable only that gate; all identity and leakage invariants remain active.
    return build_splits(
        events,
        seed=seed,
        enforce_minimum_test_player_size=False,
    )


def test_both_players_seen_or_both_held_out(tmp_path):
    splits, manifest = _synthetic_splits(_events(tmp_path), seed=42)
    seen = set(manifest["player_partition"]["seen"])
    held = set(manifest["player_partition"]["held_out"])

    assert (
    (splits["split"] != "excluded")
    | splits["exclusion_reason"].notna()
    ).all()
    assert set(splits["player_partition"]) <= {"seen", "held_out", "mixed", "invalid"}

    for row in splits.itertuples(index=False):
        if row.player_partition == "seen":
            assert row.split in {"train", "val", "test_match"}
            assert row.player_a in seen and row.player_b in seen
        elif row.player_partition == "held_out":
            assert row.split == "test_player"
            assert row.player_a in held and row.player_b in held


def test_mixed_player_matches_excluded(tmp_path):
    splits, _ = _synthetic_splits(_events(tmp_path), seed=42)
    mixed = splits[splits["player_partition"] == "mixed"]
    assert not mixed.empty
    assert (mixed["split"] == "excluded").all()
    assert (mixed["exclusion_reason"] == "mixed_player_partition").all()


def test_invalid_identity_matches_excluded(tmp_path):
    events = _events(tmp_path)
    # Remove all recoverable A/B identities for m1 while preserving its events.
    mask = events["match_id"].astype(str) == "m1"
    events.loc[mask, "player_role"] = "X"
    events.loc[mask, "hitter_id"] = None
    events.loc[mask, "receiver_id"] = None

    splits, _ = _synthetic_splits(events, seed=42)
    invalid = splits[splits["match_id"] == "m1"]
    assert len(invalid) == 1
    assert invalid.iloc[0]["player_partition"] == "invalid"
    assert invalid.iloc[0]["split"] == "excluded"
    assert invalid.iloc[0]["exclusion_reason"] == "missing_identity"


def test_no_player_partition_overlap(tmp_path):
    splits, manifest = _synthetic_splits(_events(tmp_path), seed=42)
    seen = set(manifest["player_partition"]["seen"])
    held = set(manifest["player_partition"]["held_out"])
    observed = set(splits["player_a"].dropna()) | set(splits["player_b"].dropna())
    assert seen.isdisjoint(held)
    assert seen | held == observed


def test_no_match_leakage(tmp_path):
    splits, _ = _synthetic_splits(_events(tmp_path), seed=42)
    assert splits["match_id"].is_unique


def test_all_event_matches_covered_exactly_once(tmp_path):
    events = _events(tmp_path)
    splits, _ = _synthetic_splits(events, seed=42)
    assert set(splits["match_id"]) == set(events["match_id"].astype(str))
    assert len(splits) == events["match_id"].nunique()


def test_same_seed_logical_equality(tmp_path):
    events = _events(tmp_path)
    a, manifest_a = _synthetic_splits(events, seed=42)
    b, manifest_b = _synthetic_splits(events, seed=42)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert manifest_a["content_hash_sha256"] == manifest_b["content_hash_sha256"]


def test_different_seed_changes_player_partition_for_fixture(tmp_path):
    events = _events(tmp_path)
    _, manifest_42 = _synthetic_splits(events, seed=42)
    _, manifest_7 = _synthetic_splits(events, seed=7)
    held_42 = set(manifest_42["player_partition"]["held_out"])
    held_7 = set(manifest_7["player_partition"]["held_out"])
    assert held_42 != held_7


def test_minimum_test_player_gate_is_strict_by_default(tmp_path):
    from badminton_rl.data.splits import build_splits

    events = _events(tmp_path)
    with pytest.raises(RuntimeError, match="test_player below configured minimum"):
        build_splits(events, seed=42)


def test_small_fixture_records_minimum_warning_when_gate_disabled(tmp_path):
    splits, manifest = _synthetic_splits(_events(tmp_path), seed=42)
    validation = manifest["test_player_validation"]
    assert validation["enforced"] is False
    assert validation["passed"] is False
    assert validation["warning"] is not None
    assert validation["actual_matches"] == int((splits["split"] == "test_player").sum())


def test_duplicate_event_ids_rejected(tmp_path):
    from badminton_rl.data.splits import build_splits

    events = _events(tmp_path)
    events.loc[events.index[-1], "event_id"] = events.loc[events.index[0], "event_id"]
    with pytest.raises(ValueError, match="duplicate event_id"):
        build_splits(events, enforce_minimum_test_player_size=False)


def test_missing_match_id_rejected(tmp_path):
    from badminton_rl.data.splits import build_splits

    events = _events(tmp_path)
    events.loc[events.index[0], "match_id"] = pd.NA
    with pytest.raises(ValueError, match="null match_id"):
        build_splits(events, enforce_minimum_test_player_size=False)
