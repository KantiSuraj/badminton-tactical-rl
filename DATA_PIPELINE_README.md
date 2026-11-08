# Badminton Tactical RL — Data Pipeline and Split Reproduction

This file is the working runbook for the ShuttleSet data pipeline completed before the Gymnasium environment work.

It records the commands used to validate the data pipeline and the two frozen evaluation protocols:

| Protocol | Seed | Primary evaluation split | Purpose |
|---|---:|---|---|
| Match-disjoint | `42` | `test_match` | Unseen matches involving players represented in the training-player partition |
| Player-disjoint | `434` | `test_player` | Matches between players held out from the training-player partition |

**Do not combine metrics from these two protocols.** They have different player partitions and different training sets. Use the same protocol-specific split for P1, P2 and P3 comparisons.

---

## 1. Repository and environment

Run all commands from the repository root (`badminton-tactical-rl`). The examples below assume the project is at `~/project-sem/badminton-tactical-rl` and the virtual environment is `.venv`.

```bash
cd ~/project-sem/badminton-tactical-rl
source .venv/bin/activate
```

Check that the expected data and source files exist:

```bash
python - <<'PY'
from pathlib import Path

required = [
    Path("data/raw/shuttleset/set/match.csv"),
    Path("data/raw/shuttleset/set/homography.csv"),
    Path("src/badminton_rl/data/build_events.py"),
    Path("src/badminton_rl/data/splits.py"),
    Path("tests/test_data_events.py"),
    Path("tests/test_data_splits.py"),
]

missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing required files:\n- " + "\n- ".join(missing))

print("Repository and ShuttleSet paths look present.")
PY
```

If your repository is not at that location, change only the `cd` command. Do not move the dataset or alter the Python source paths just to match this example.

---

## 2. Run the tests

### Data-pipeline tests

```bash
python -m pytest -q tests/test_data_events.py tests/test_data_splits.py
```

### Entire test suite

```bash
python -m pytest -q
```

The event tests include real-dataset checks when `data/raw/shuttleset/set` exists. The real-data checks expect the current local snapshot to contain **36,484 events and 3,683 rallies**. The upstream README reports a slightly different stroke total; do not silently change the local test expectation to hide that discrepancy.

The split tests use a small synthetic fixture. They explicitly disable only the production `test_player` minimum-size gate because a tiny synthetic fixture cannot provide a meaningful held-out-player test set. The production split command below keeps the gate enabled for the player-disjoint protocol.

---

## 3. Rebuild the canonical event table (only when needed)

The event builder is `badminton_rl.data.build_events.build_events`. It reads ShuttleSet from `data/raw/shuttleset/set` and, when `output_path` is supplied, writes:

```text
data/processed/shuttleset/events/
├── events.parquet
└── events_manifest.yaml
```

Run this from the repository root:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import yaml

from badminton_rl.data.build_events import build_events

SHUTTLESET_ROOT = Path("data/raw/shuttleset/set")
EVENTS_DIR = Path("data/processed/shuttleset/events")

if not (SHUTTLESET_ROOT / "match.csv").exists():
    raise FileNotFoundError(f"ShuttleSet root not found: {SHUTTLESET_ROOT.resolve()}")

# Build the canonical per-stroke table and write Parquet + YAML manifest.
events = build_events(root=SHUTTLESET_ROOT, output_path=EVENTS_DIR)

print(f"Events:  {len(events):,}")
print(f"Matches: {events['match_id'].nunique():,}")
print(f"Rallies: {events['rally_id'].nunique():,}")
print(f"Parquet: {EVENTS_DIR / 'events.parquet'}")
print(f"Manifest:{EVENTS_DIR / 'events_manifest.yaml'}")

# Guard against accidentally building from the wrong dataset snapshot.
assert len(events) == 36_484, f"Unexpected event count: {len(events)}"
assert events["match_id"].nunique() == 44
assert events["rally_id"].nunique() == 3_683

manifest_path = EVENTS_DIR / "events_manifest.yaml"
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
print("\nEvent manifest provenance:")
print(yaml.safe_dump(manifest["provenance"], sort_keys=False, allow_unicode=True))
print("Event-table checks passed.")
PY
```

**Do not rebuild the event table every time you regenerate a split.** Rebuild it only if the source dataset, loader, coordinate transform, taxonomy mapping or event-builder implementation changes. The existing `events.parquet` is the input to both split protocols.

To inspect the saved event manifest without rebuilding anything:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path("data/processed/shuttleset/events/events_manifest.yaml")
if not path.exists():
    raise FileNotFoundError(path)
print(yaml.safe_dump(yaml.safe_load(path.read_text(encoding="utf-8")),
                     sort_keys=False, allow_unicode=True))
PY
```

---

## 4. Generate and save both frozen split protocols

### Where the seeds are set

The seeds are arguments to `build_splits()` in `src/badminton_rl/data/splits.py`. **Do not edit `splits.py` to change them.** Set them in the `PROTOCOLS` list in the script below:

- Match-disjoint: `seed=42`
- Player-disjoint: `seed=434`

The script writes each split to a different directory. This prevents one protocol from overwriting the other.

### Run the split builder

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

from badminton_rl.data.splits import build_splits, write_splits

EVENTS_PATH = Path("data/processed/shuttleset/events/events.parquet")
SPLITS_ROOT = Path("data/processed/shuttleset/splits")

# These are the two frozen evaluation protocols. Change a seed only as an
# explicit methodology change, then preserve the new seed in the output path
# and manifest. Do not choose a seed based on RL model performance.
PROTOCOLS = [
    {
        "name": "match_disjoint_seed_42",
        "seed": 42,
        "enforce_minimum_test_player_size": False,
        "primary_test_split": "test_match",
    },
    {
        "name": "player_disjoint_seed_434",
        "seed": 434,
        "enforce_minimum_test_player_size": True,
        "primary_test_split": "test_player",
    },
]

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"Canonical events file not found: {EVENTS_PATH}. "
        "Run Section 3 first."
    )

events = pd.read_parquet(EVENTS_PATH)
print(f"Loaded {len(events):,} events from {EVENTS_PATH}")

for protocol in PROTOCOLS:
    name = protocol["name"]
    seed = protocol["seed"]
    primary_test_split = protocol["primary_test_split"]

    # For seed 42, test_player is not the target of this protocol; the strict
    # player-test minimum is therefore disabled. The test_match split remains
    # the primary evaluation set. For seed 434, the player-test minimum is
    # enforced because test_player is the primary evaluation set.
    splits, manifest = build_splits(
        events,
        seed=seed,
        enforce_minimum_test_player_size=protocol[
            "enforce_minimum_test_player_size"
        ],
    )

    # Structural and leakage checks before writing anything.
    assert splits["match_id"].is_unique, f"Duplicate match rows in {name}"
    assert set(splits["match_id"]) == set(events["match_id"].astype(str)), (
        f"Not every event match is represented exactly once in {name}"
    )

    seen = set(manifest["player_partition"]["seen"])
    held_out = set(manifest["player_partition"]["held_out"])
    assert seen.isdisjoint(held_out)

    for row in splits.itertuples(index=False):
        if row.split in {"train", "val", "test_match"}:
            assert row.player_a in seen and row.player_b in seen, (
                f"Seen/held-out player leakage in {name}, match {row.match_id}"
            )
        elif row.split == "test_player":
            assert row.player_a in held_out and row.player_b in held_out, (
                f"Held-out player leakage in {name}, match {row.match_id}"
            )

    assert int((splits["split"] == primary_test_split).sum()) > 0, (
        f"Primary test split {primary_test_split!r} is empty in {name}"
    )

    out_dir = SPLITS_ROOT / name
    write_splits(splits, manifest, out_dir)

    split_counts = splits["split"].value_counts().to_dict()
    stroke_counts = splits.groupby("split")["n_strokes"].sum().to_dict()

    print(f"\n{'=' * 68}")
    print(f"PROTOCOL: {name}")
    print(f"Seed: {seed}")
    print(f"Primary evaluation split: {primary_test_split}")
    print(f"Held-out players: {manifest['player_partition']['n_held_out']}")
    print(f"Seen players: {manifest['player_partition']['n_seen']}")
    print("Match counts:", split_counts)
    print("Stroke counts:", stroke_counts)
    print("Test-player validation:", manifest["test_player_validation"])
    print(f"Saved Parquet: {out_dir / 'splits.parquet'}")
    print(f"Saved manifest: {out_dir / 'splits_manifest.yaml'}")

print("\nBoth split protocols were generated and passed structural checks.")
PY
```

### Why the minimum-size gate differs between these two calls

- **Seed 42 / match-disjoint:** `test_match` is the primary evaluation set. Its incidental `test_player` split contains only one match, so the player-test minimum is disabled for this protocol. The manifest records that the player-test minimum was not enforced and did not pass. This is expected; do not use that one match as a meaningful player-disjoint result.
- **Seed 434 / player-disjoint:** `test_player` is the primary evaluation set, so the production minimum-size gate remains enabled. This split should satisfy the minimum of at least 2 matches and 100 strokes.

The split writer creates these separate outputs:

```text
data/processed/shuttleset/splits/
├── match_disjoint_seed_42/
│   ├── splits.parquet
│   └── splits_manifest.yaml
└── player_disjoint_seed_434/
    ├── splits.parquet
    └── splits_manifest.yaml
```

Each manifest stores its seed, player partition, split counts, stroke/rally counts, validation status and a hash of the split table. The directory name and manifest together identify the protocol. Re-running the split builder overwrites the two files inside those same named directories.

---

## 5. Re-open and validate the saved split artifacts

This command reads the files written in Section 4. It does not rebuild the event table or the splits.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import yaml

EVENTS_PATH = Path("data/processed/shuttleset/events/events.parquet")
SPLITS_ROOT = Path("data/processed/shuttleset/splits")
events = pd.read_parquet(EVENTS_PATH)

protocols = [
    ("match_disjoint_seed_42", 42, "test_match"),
    ("player_disjoint_seed_434", 434, "test_player"),
]

for name, expected_seed, primary_split in protocols:
    out_dir = SPLITS_ROOT / name
    splits_path = out_dir / "splits.parquet"
    manifest_path = out_dir / "splits_manifest.yaml"
    if not splits_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing split outputs in {out_dir}")

    splits = pd.read_parquet(splits_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["seed"] == expected_seed
    assert splits["seed"].nunique() == 1
    assert int(splits["seed"].iloc[0]) == expected_seed
    assert splits["match_id"].is_unique
    assert set(splits["match_id"]) == set(events["match_id"].astype(str))
    assert int((splits["split"] == primary_split).sum()) > 0

    seen = set(manifest["player_partition"]["seen"])
    held_out = set(manifest["player_partition"]["held_out"])
    assert seen.isdisjoint(held_out)

    for row in splits.itertuples(index=False):
        if row.split in {"train", "val", "test_match"}:
            assert row.player_a in seen and row.player_b in seen
        elif row.split == "test_player":
            assert row.player_a in held_out and row.player_b in held_out

    print(f"\n{name}")
    print(f"  seed: {manifest['seed']}")
    print(f"  primary test: {primary_split}")
    print(f"  held-out players: {manifest['player_partition']['n_held_out']}")
    print("  match counts:", splits["split"].value_counts().to_dict())
    print("  primary-test matches:", int((splits["split"] == primary_split).sum()))
    print("  primary-test strokes:", int(splits.loc[splits["split"] == primary_split, "n_strokes"].sum()))
    print("  manifest hash:", manifest["content_hash_sha256"])
    print("  validation: passed")

print("\nSaved artifact validation completed.")
PY
```

For the current data snapshot, the expected match counts are:

| Split | Seed 42: match-disjoint | Seed 434: player-disjoint |
|---|---:|---:|
| `train` | 22 | 15 |
| `val` | 5 | 3 |
| `test_match` | 5 | 3 |
| `test_player` | 1 | 10 |
| `excluded` | 11 | 13 |

The seed-434 player-disjoint test has five held-out players, all represented in its 10 test matches. Those matches are concentrated in five distinct player pairings, with one pairing appearing five times. Report this concentration when interpreting player-disjoint metrics; 10 matches do not mean 10 independent player pairings.

---

## 6. How to use the split files downstream

When constructing training datasets or evaluation jobs, load the selected split file and filter by its `split` column. For example:

```python
from pathlib import Path
import pandas as pd

EVENTS_PATH = Path("data/processed/shuttleset/events/events.parquet")
MATCH_SPLIT_PATH = Path(
    "data/processed/shuttleset/splits/match_disjoint_seed_42/splits.parquet"
)

# For the player-disjoint experiment, use instead:
# PLAYER_SPLIT_PATH = Path(
#     "data/processed/shuttleset/splits/player_disjoint_seed_434/splits.parquet"
# )

events = pd.read_parquet(EVENTS_PATH)
splits = pd.read_parquet(MATCH_SPLIT_PATH)

train_match_ids = set(splits.loc[splits["split"] == "train", "match_id"])
val_match_ids = set(splits.loc[splits["split"] == "val", "match_id"])
test_match_ids = set(splits.loc[splits["split"] == "test_match", "match_id"])

train_events = events[events["match_id"].astype(str).isin(train_match_ids)].copy()
val_events = events[events["match_id"].astype(str).isin(val_match_ids)].copy()
test_events = events[events["match_id"].astype(str).isin(test_match_ids)].copy()

print(len(train_events), len(val_events), len(test_events))
```

For player-disjoint evaluation, change `MATCH_SPLIT_PATH` to the player-disjoint path and use `split == "test_player"` as the primary test set. Do not train on `excluded` rows or on any test rows.

---

## 7. What is frozen, and what is not

Frozen for the current experiments:

- Canonical event schema and event-builder implementation already present in the repository.
- Match-disjoint protocol: seed `42`, primary test split `test_match`.
- Player-disjoint protocol: seed `434`, primary test split `test_player`.
- 80/20 seen-versus-held-out player partition.
- 70/15/15 allocation of eligible seen-player matches to train/validation/test-match.
- Mixed-player matches excluded from both protocols.

The exploratory 1,000-seed sweep was used only to inspect split feasibility. Previous terminal outputs reported different passing-seed totals (622 and 300), so **do not cite either percentage as a project result**. The final split Parquet files and their manifests are the reproducibility record for the selected protocols. There is no reason to run the 1,000-seed sweep again during normal development.

Do not change a seed, split proportion, minimum gate or player assignment silently. If the dataset or splitting implementation changes, regenerate both protocols, inspect the manifests and document the change before model training.

---

## 8. Next milestone: Gymnasium environment

The dataset pipeline and split artifacts are inputs to training and evaluation; they are not dependencies of the Gymnasium API itself. Once the commands above pass and both split directories exist, proceed to the environment implementation:

1. Define the environment state and observation space.
2. Define the macro-action `(shot_type, landing_cell, recovery_cell)`.
3. Implement action decoding and feasibility checks.
4. Connect shuttle-flight and player-movement dynamics.
5. Implement rally transitions, terminal conditions and rewards.
6. Implement and test `reset()` and `step()` against Gymnasium's API.

The data splits must be used consistently when training and evaluating policies, but they do not need to be modified to begin implementing the environment.
