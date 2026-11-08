"""Build deterministic player-held-out and match-held-out ShuttleSet splits.

The split has two independent stages:

1. Partition unique players into ``seen`` (80%) and ``held_out`` (20%).
2. Classify each match:
   * both players seen      -> train / val / test_match (70 / 15 / 15)
   * both players held_out  -> test_player
   * one seen, one held_out -> excluded
   * missing/ambiguous IDs  -> excluded

The minimum-size requirement for ``test_player`` is a production-data gate.
Synthetic unit tests can disable that gate explicitly while still testing the
split assignments and invariants. No player or match is silently reassigned to
satisfy a minimum.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .schemas import SPLIT_SCHEME_VERSION


_REQUIRED_EVENT_COLUMNS = frozenset(
    {"match_id", "player_role", "hitter_id", "receiver_id", "event_id", "rally_id"}
)
_SPLIT_ORDER = ("train", "val", "test_match", "test_player", "excluded")
_PARTITION_ORDER = ("seen", "held_out", "mixed", "invalid")


def _clean_player_id(value: Any) -> str | None:
    """Normalize a player identifier's surrounding whitespace, not its case."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimum event-table contract and return a normalized copy."""
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")

    missing = sorted(_REQUIRED_EVENT_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(f"events is missing required columns: {missing}")
    if events.empty:
        raise ValueError("cannot build splits from an empty events table")

    out = events.copy()
    if out["match_id"].isna().any():
        raise ValueError("events contains null match_id values")

    out["match_id"] = out["match_id"].astype("string").str.strip()
    if out["match_id"].isna().any() or out["match_id"].eq("").any():
        raise ValueError("events contains empty match_id values")

    # Event IDs are the source-row identity. Duplicate IDs usually mean the
    # event table was assembled incorrectly and should not be split silently.
    if out["event_id"].isna().any() or out["event_id"].astype("string").str.strip().eq("").any():
        raise ValueError("events contains null or empty event_id values")
    if out["rally_id"].isna().any() or out["rally_id"].astype("string").str.strip().eq("").any():
        raise ValueError("events contains null or empty rally_id values")
    if out["event_id"].duplicated().any():
        duplicates = out.loc[out["event_id"].duplicated(), "event_id"].head(5).tolist()
        raise ValueError(f"events contains duplicate event_id values: {duplicates}")

    # Player names are normalized only for whitespace. Case variants are
    # detected separately below rather than silently merged.
    for col in ("hitter_id", "receiver_id"):
        out[col] = out[col].map(_clean_player_id).astype("object")

    out["player_role"] = out["player_role"].astype("string").str.strip()
    return out


def _match_players(events: pd.DataFrame) -> pd.DataFrame:
    """Resolve each match's A/winner and B/loser identities from event roles.

    A hitter or B receiver identifies player A. B hitter or A receiver
    identifies player B. Both directions must independently resolve to exactly
    one identity, and the union must contain exactly two distinct players.
    """
    rows: list[dict[str, Any]] = []

    for match_id, group in events.groupby("match_id", sort=True, dropna=False):
        a_candidates = set(
            group.loc[group["player_role"].eq("A"), "hitter_id"].dropna().tolist()
        )
        a_candidates.update(
            group.loc[group["player_role"].eq("B"), "receiver_id"].dropna().tolist()
        )

        b_candidates = set(
            group.loc[group["player_role"].eq("B"), "hitter_id"].dropna().tolist()
        )
        b_candidates.update(
            group.loc[group["player_role"].eq("A"), "receiver_id"].dropna().tolist()
        )

        all_players = (
            set(group["hitter_id"].dropna().tolist())
            | set(group["receiver_id"].dropna().tolist())
        )

        valid = (
            len(a_candidates) == 1
            and len(b_candidates) == 1
            and len(all_players) == 2
        )
        if valid:
            player_a = next(iter(a_candidates))
            player_b = next(iter(b_candidates))
            valid = player_a != player_b and {player_a, player_b} == all_players
        else:
            player_a = player_b = None

        if not valid:
            # Do not let partially resolved candidates from an invalid match
            # contaminate the global player partition.
            player_a = player_b = None
            rows.append(
                {
                    "match_id": str(match_id),
                    "player_a": player_a,
                    "player_b": player_b,
                    "identity_status": "missing_identity",
                }
            )
        else:
            rows.append(
                {
                    "match_id": str(match_id),
                    "player_a": player_a,
                    "player_b": player_b,
                    "identity_status": "ok",
                }
            )

    return pd.DataFrame(
        rows,
        columns=["match_id", "player_a", "player_b", "identity_status"],
    )


def _check_player_name_collisions(players_per_match: pd.DataFrame) -> None:
    """Reject identifiers that differ only by case after whitespace trimming."""
    names = pd.concat(
        [players_per_match["player_a"], players_per_match["player_b"]],
        ignore_index=True,
    ).dropna()

    variants: dict[str, set[str]] = {}
    for value in names.unique().tolist():
        name = str(value)
        variants.setdefault(name.casefold(), set()).add(name)

    collisions = {
        key: sorted(values)
        for key, values in variants.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(
            "player identifiers differ only by case; resolve them in the source "
            f"metadata before splitting: {collisions}"
        )


def _allocate_seen_match_splits(
    match_ids: list[str], seed: int,
) -> dict[str, str]:
    """Deterministically assign eligible matches approximately 70/15/15."""
    shuffled = sorted(match_ids)
    # Separate RNG stream from the player partition. Adding/removing players
    # will not perturb the match-level allocation for a fixed seed and match set.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    # Guard against independent rounding ever allocating more than n rows.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    result: dict[str, str] = {}
    for index, match_id in enumerate(shuffled):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test_match"
        result[match_id] = split
    return result


def build_splits(
    events: pd.DataFrame,
    seed: int = 42,
    *,
    enforce_minimum_test_player_size: bool = True,
    min_test_player_matches: int = 2,
    min_test_player_strokes: int = 100,
) -> tuple[pd.DataFrame, dict]:
    """Build the split table and a reproducibility/validation manifest.

    Production default: require at least ``min_test_player_matches`` held-out
    matches and ``min_test_player_strokes`` strokes in ``test_player``.
    Synthetic tests should pass ``enforce_minimum_test_player_size=False``;
    this disables only the sample-size gate, not any split-integrity checks.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    for name, value in (
        ("min_test_player_matches", min_test_player_matches),
        ("min_test_player_strokes", min_test_player_strokes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(enforce_minimum_test_player_size, bool):
        raise TypeError("enforce_minimum_test_player_size must be bool")

    data = _validate_events(events)
    players_per_match = _match_players(data)
    if players_per_match["match_id"].duplicated().any():
        raise ValueError("internal error: more than one identity row for a match")

    _check_player_name_collisions(players_per_match)

    all_names = pd.concat(
        [players_per_match["player_a"], players_per_match["player_b"]],
        ignore_index=True,
    ).dropna()
    unique_players = sorted({str(name) for name in all_names.tolist()})
    if len(unique_players) < 2:
        raise ValueError(
            f"need at least two identifiable players to split; found {len(unique_players)}"
        )

    # ---- 80/20 player partition ----
    player_rng = np.random.default_rng(int(seed))
    permutation = player_rng.permutation(len(unique_players))
    n_held_out = max(1, int(round(0.20 * len(unique_players))))
    n_held_out = min(n_held_out, len(unique_players) - 1)
    held_out = {unique_players[i] for i in permutation[:n_held_out]}
    seen = set(unique_players) - held_out
    if seen & held_out or seen | held_out != set(unique_players):
        raise RuntimeError("player partition invariant violated")

    # ---- classify matches ----
    per_match: list[dict[str, Any]] = []
    seen_eligible: list[str] = []

    for row in players_per_match.itertuples(index=False):
        match_id = str(row.match_id)
        player_a, player_b = row.player_a, row.player_b

        if row.identity_status != "ok":
            per_match.append(
                {
                    "match_id": match_id,
                    "player_a": player_a,
                    "player_b": player_b,
                    "split": "excluded",
                    "player_partition": "invalid",
                    "exclusion_reason": "missing_identity",
                }
            )
        elif {player_a, player_b} <= held_out:
            per_match.append(
                {
                    "match_id": match_id,
                    "player_a": player_a,
                    "player_b": player_b,
                    "split": "test_player",
                    "player_partition": "held_out",
                    "exclusion_reason": None,
                }
            )
        elif {player_a, player_b} <= seen:
            seen_eligible.append(match_id)
        else:
            per_match.append(
                {
                    "match_id": match_id,
                    "player_a": player_a,
                    "player_b": player_b,
                    "split": "excluded",
                    "player_partition": "mixed",
                    "exclusion_reason": "mixed_player_partition",
                }
            )

    split_of = _allocate_seen_match_splits(seen_eligible, int(seed))
    identity_lookup = players_per_match.set_index("match_id")
    for match_id in sorted(seen_eligible):
        identity = identity_lookup.loc[match_id]
        per_match.append(
            {
                "match_id": match_id,
                "player_a": identity["player_a"],
                "player_b": identity["player_b"],
                "split": split_of[match_id],
                "player_partition": "seen",
                "exclusion_reason": None,
            }
        )

    splits = pd.DataFrame(per_match)
    if splits.empty:
        raise ValueError("no matches were available to classify")

    splits["split_scheme_version"] = SPLIT_SCHEME_VERSION
    splits["seed"] = int(seed)

    # ---- match-level observation counts ----
    counts = (
        data.groupby("match_id", sort=True)
        .agg(
            n_strokes=("event_id", "size"),
            n_rallies=("rally_id", "nunique"),
        )
        .reset_index()
    )
    counts["match_id"] = counts["match_id"].astype(str)

    splits = splits.merge(
        counts,
        on="match_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not splits["_merge"].eq("both").all():
        missing_matches = splits.loc[splits["_merge"] != "both", "match_id"].tolist()
        raise RuntimeError(f"split rows have no matching events: {missing_matches}")
    splits = splits.drop(columns="_merge")
    splits[["n_strokes", "n_rallies"]] = splits[["n_strokes", "n_rallies"]].astype("int64")

    splits = splits[
        [
            "match_id", "player_a", "player_b", "split", "player_partition",
            "split_scheme_version", "seed", "n_strokes", "n_rallies",
            "exclusion_reason",
        ]
    ].sort_values("match_id", kind="stable").reset_index(drop=True)

    # ---- structural invariants ----
    if splits["match_id"].duplicated().any():
        raise RuntimeError("match appears in more than one split row")
    expected_matches = set(data["match_id"].unique().tolist())
    actual_matches = set(splits["match_id"].tolist())
    if actual_matches != expected_matches:
        raise RuntimeError(
            "split match coverage mismatch: "
            f"missing={sorted(expected_matches - actual_matches)}, "
            f"extra={sorted(actual_matches - expected_matches)}"
        )
    if not set(splits["split"].unique()) <= set(_SPLIT_ORDER):
        raise RuntimeError("unexpected split label generated")
    if not set(splits["player_partition"].unique()) <= set(_PARTITION_ORDER):
        raise RuntimeError("unexpected player_partition label generated")

    # Every test_player match must contain two held-out participants; all
    # train/val/test_match rows must contain two seen participants.
    for row in splits.itertuples(index=False):
        if row.player_partition == "held_out":
            if row.split != "test_player" or row.player_a not in held_out or row.player_b not in held_out:
                raise RuntimeError(f"held-out match classification invariant failed: {row.match_id}")
        elif row.player_partition == "seen":
            if row.split not in {"train", "val", "test_match"} or row.player_a not in seen or row.player_b not in seen:
                raise RuntimeError(f"seen match classification invariant failed: {row.match_id}")
        elif row.player_partition == "mixed":
            if row.split != "excluded" or row.exclusion_reason != "mixed_player_partition":
                raise RuntimeError(f"mixed match classification invariant failed: {row.match_id}")
        elif row.player_partition == "invalid":
            if row.split != "excluded" or row.exclusion_reason != "missing_identity":
                raise RuntimeError(f"invalid match classification invariant failed: {row.match_id}")

    # ---- production sample-size gate ----
    test_player = splits.loc[splits["split"].eq("test_player")]
    test_player_matches = int(len(test_player))
    test_player_strokes = int(test_player["n_strokes"].sum())
    minimum_passed = (
        test_player_matches >= min_test_player_matches
        and test_player_strokes >= min_test_player_strokes
    )
    minimum_message = None
    if not minimum_passed:
        minimum_message = (
            "test_player below configured minimum: "
            f"matches={test_player_matches}/{min_test_player_matches}, "
            f"strokes={test_player_strokes}/{min_test_player_strokes}. "
            "Do not change the player partition or seed silently; inspect the "
            "dataset and justify any methodological change."
        )
        if enforce_minimum_test_player_size:
            raise RuntimeError(minimum_message)

    # ---- manifest ----
    split_counts = splits["split"].value_counts().to_dict()
    partition_counts = splits["player_partition"].value_counts().to_dict()
    split_strokes = splits.groupby("split", sort=True)["n_strokes"].sum().to_dict()
    split_rallies = splits.groupby("split", sort=True)["n_rallies"].sum().to_dict()

    # Explicitly include zero-count categories to make manifests comparable.
    split_counts = {name: int(split_counts.get(name, 0)) for name in _SPLIT_ORDER}
    partition_counts = {name: int(partition_counts.get(name, 0)) for name in _PARTITION_ORDER}
    split_strokes = {name: int(split_strokes.get(name, 0)) for name in _SPLIT_ORDER}
    split_rallies = {name: int(split_rallies.get(name, 0)) for name in _SPLIT_ORDER}

    exclusion_reasons = (
        splits.loc[splits["split"].eq("excluded"), "exclusion_reason"]
        .value_counts()
        .to_dict()
    )
    manifest: dict[str, Any] = {
        "split_scheme_version": SPLIT_SCHEME_VERSION,
        "seed": int(seed),
        "method": {
            "player_partition": {"seen_fraction_target": 0.80, "held_out_fraction_target": 0.20},
            "seen_match_partition": {
                "train_fraction_target": 0.70,
                "val_fraction_target": 0.15,
                "test_match_fraction_target": 0.15,
                "allocation": "rounded_counts_after_seeded_shuffle",
            },
        },
        "player_partition": {
            "seen": sorted(seen),
            "held_out": sorted(held_out),
            "n_seen": int(len(seen)),
            "n_held_out": int(len(held_out)),
            "actual_seen_fraction": float(len(seen) / len(unique_players)),
            "actual_held_out_fraction": float(len(held_out) / len(unique_players)),
            "n_unique_identifiable_players": int(len(unique_players)),
        },
        "match_counts": {
            "total": int(len(splits)),
            "by_split": split_counts,
            "by_player_partition": partition_counts,
        },
        "stroke_counts_by_split": split_strokes,
        "rally_counts_by_split": split_rallies,
        "test_player_validation": {
            "enforced": bool(enforce_minimum_test_player_size),
            "passed": bool(minimum_passed),
            "minimum_matches": int(min_test_player_matches),
            "minimum_strokes": int(min_test_player_strokes),
            "actual_matches": test_player_matches,
            "actual_strokes": test_player_strokes,
            "warning": minimum_message,
        },
        "exclusion_reasons": {
            str(reason): int(count) for reason, count in sorted(exclusion_reasons.items())
        },
        "content_hash_sha256": hashlib.sha256(
            splits.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest(),
    }
    return splits, manifest


def write_splits(splits: pd.DataFrame, manifest: dict, out_dir: Path) -> None:
    """Write splits.parquet and splits_manifest.yaml into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out_dir / "splits.parquet", index=False)
    (out_dir / "splits_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
