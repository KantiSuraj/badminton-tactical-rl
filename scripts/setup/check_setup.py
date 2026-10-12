"""Check that the environment, data and external repos are set up. Run from the project root:

    python scripts/setup/check_setup.py [--coachai-root ../external/CoachAI-Plus]

Prints OK / FAIL per check and exits non-zero if a required check fails.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

PINNED = {  # module -> required version prefix (from requirements-coachai.txt)
    "numpy": "1.26", "pandas": "2.2", "sklearn": "1.5", "scipy": "1.14", "torch": "2.4", "torchsde": "0.2",
}
COACHAI_PLUS_SHA = "4fd05934f172eec16c74d05c2902983b64c8521e"
COACHAI_FILES = [
    "BadmintonEnv/Environment.py", "BadmintonEnv/RunEnv.py", "input_data/target_players_ids.pkl",
    "BadmintonEnv/Agent/hyperparameters.pkl", "BadmintonEnv/Agent/RallyNet_weight.trc",
    "BadmintonEnv/Agent/BC_weight.pth",
]
failed = []


def report(ok: bool, what: str, hint: str = "", required: bool = True) -> None:
    print(f"{'OK  ' if ok else ('FAIL' if required else 'WARN')}  {what}" + ("" if ok or not hint else f"\n      -> {hint}"))
    if not ok and required:
        failed.append(what)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coachai-root", type=Path, default=Path("../external/CoachAI-Plus"))
    ap.add_argument("--shuttleset-root", type=Path, default=Path("data/raw/shuttleset/set"))
    a = ap.parse_args()

    v = sys.version_info
    report((v.major, v.minor) == (3, 10), f"Python {v.major}.{v.minor}.{v.micro} (CoachAI+ is pinned to 3.10)",
           "create the venv with python3.10 (Ubuntu 22.04 has it)", required=False)

    for mod, want in PINNED.items():
        try:
            got = importlib.import_module(mod).__version__
            report(got.startswith(want), f"{mod} {got} (pinned {want}.x)", "pip install -r requirements-coachai.txt")
        except Exception as e:  # noqa: BLE001
            report(False, f"import {mod}: {type(e).__name__}", "pip install -r requirements-coachai.txt")

    try:
        import torch
        report(torch.cuda.is_available(), "PyTorch sees the GPU (optional; CPU is fine for this project)",
               "update the Windows NVIDIA driver; WSL2 uses it", required=False)
    except Exception:  # noqa: BLE001
        pass

    try:
        import badminton_rl  # noqa: F401
        report(True, "badminton_rl is importable")
    except Exception as e:  # noqa: BLE001
        report(False, f"import badminton_rl: {e}", 'pip install -e ".[dev]"')

    csvs = list(a.shuttleset_root.glob("*/set*.csv")) if a.shuttleset_root.exists() else []
    report(len(csvs) == 104, f"ShuttleSet CSVs in {a.shuttleset_root}: {len(csvs)} (expected 104)",
           "scripts/setup/fetch_external.sh")

    root = a.coachai_root
    report(root.exists(), f"CoachAI+ checkout at {root}", "docs/SETUP.md step 5 (fork and clone)")
    if root.exists():
        for f in COACHAI_FILES:
            report((root / f).exists(), f"CoachAI+ file {f}")
        try:
            sha = subprocess.check_output(["git", "-C", str(root), "merge-base", "--is-ancestor", COACHAI_PLUS_SHA, "HEAD"],
                                          stderr=subprocess.STDOUT)
            report(True, f"CoachAI+ checkout contains the audited commit {COACHAI_PLUS_SHA[:7]}")
        except Exception:  # noqa: BLE001
            report(False, f"CoachAI+ checkout contains the audited commit {COACHAI_PLUS_SHA[:7]}",
                   "git -C <fork> fetch upstream && git -C <fork> checkout -B baseline-pinned " + COACHAI_PLUS_SHA[:7],
                   required=False)

    print("\nALL REQUIRED CHECKS PASSED" if not failed else f"\n{len(failed)} REQUIRED CHECK(S) FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
