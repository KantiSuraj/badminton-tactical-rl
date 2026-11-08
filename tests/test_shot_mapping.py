"""Mapping YAML integrity — no real data needed."""
from pathlib import Path
import yaml

_YML = Path(__file__).parents[1] / "src/badminton_rl/data/shot_label_mapping_v1.yaml"
_SPEC = yaml.safe_load(_YML.read_text(encoding="utf-8"))

def test_mapping_version_present():
    assert _SPEC["mapping_id"] == "shot_label_mapping_v1"
    assert _SPEC["version"] == 1

def test_all_19_english_labels_present():
    labels = {e["english"] for e in _SPEC["raw_labels"]}
    assert len(labels) == 19

def test_passive_drop_has_two_chinese_spellings():
    entry = next(e for e in _SPEC["raw_labels"] if e["english"] == "passive drop")
    assert set(entry["chinese"]) == {"過度切球", "過渡切球"}

def test_every_label_has_mapping_status():
    for e in _SPEC["raw_labels"]:
        assert e["mapping_status"] in {"mapped", "unmapped", "ambiguous"}

def test_unknown_is_unmapped():
    entry = next(e for e in _SPEC["raw_labels"] if e["english"] == "unknown")
    assert entry["family"] is None
    assert entry["mapping_status"] == "unmapped"

def test_all_10_families_have_mappings():
    families_with_labels = {e["family"] for e in _SPEC["raw_labels"] if e["family"]}
    expected = {"CLEAR","DROP","SMASH","HALF_SMASH","DRIVE","PUSH","LIFT","NET_SHOT","BLOCK","CUT"}
    assert families_with_labels == expected

def test_family_ids_match_schema():
    from badminton_rl.data.schemas import FAMILY_TO_ID
    for name, meta in _SPEC["families"].items():
        assert FAMILY_TO_ID[name] == meta["id"]

def test_no_ambiguous_labels_currently():
    # must be empty before calibration uses the mapping
    assert _SPEC["ambiguous"] == []