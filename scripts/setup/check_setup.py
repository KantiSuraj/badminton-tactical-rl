"""Check the parent environment, isolated CoachAI worker environment and data/repo setup.

Run from the project root with the Python 3.12 parent environment active:

    python scripts/setup/check_setup.py

The worker interpreter defaults to .venv-coachai/bin/python (Linux/WSL) or
.venv-coachai/Scripts/python.exe (Windows). Override it with --worker-python.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

COACHAI_PLUS_SHA = "4fd05934f172eec16c74d05c2902983b64c8521e"
COACHAI_FILES = [
    "BadmintonEnv/Environment.py",
    "BadmintonEnv/RunEnv.py",
    "input_data/All_dataset.csv",
    "input_data/target_players_ids.pkl",
    "BadmintonEnv/Agent/hyperparameters.pkl",
    "BadmintonEnv/Agent/RallyNet_weight.trc",
    "BadmintonEnv/Agent/INV_weight.pth",
    "BadmintonEnv/Agent/BC4Serve_weight.pth",
]
WORKER_DISTRIBUTIONS = {
    "numpy": "1.26",
    "pandas": "2.2",
    "scikit-learn": "1.5",
    "scipy": "1.14",
    "torch": "2.4",
    "torchsde": "0.2",
}

failed: list[str] = []


def report(ok: bool, what: str, hint: str = "", *, required: bool = True) -> None:
    status = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"{status}  {what}" + ("" if ok or not hint else f"\n      -> {hint}"))
    if not ok and required:
        failed.append(what)


def default_worker_python() -> Path:
    if os.name == "nt":
        return Path(".venv-coachai/Scripts/python.exe")
    return Path(".venv-coachai/bin/python")


def inspect_worker(worker_python: Path) -> None:
    """Inspect worker dependencies in a subprocess, never importing them in the parent."""
    report(worker_python.is_file(), f"CoachAI worker interpreter exists at {worker_python}",
           "Create .venv-coachai with Python 3.12 and install requirements-coachai.txt")
    if not worker_python.is_file():
        return

    probe_code = r'''
import importlib.metadata as metadata
import json
import sys

expected = {
    "numpy": "1.26",
    "pandas": "2.2",
    "scikit-learn": "1.5",
    "scipy": "1.14",
    "torch": "2.4",
    "torchsde": "0.2",
}
versions = {}
for distribution, prefix in expected.items():
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        print(json.dumps({"error": str(exc), "distribution": distribution}))
        raise SystemExit(2)
    versions[distribution] = actual
    if not actual.startswith(prefix):
        print(json.dumps({"error": "version mismatch", "distribution": distribution,
                          "expected_prefix": prefix, "actual": actual}))
        raise SystemExit(3)

import torch
print(json.dumps({"python": list(sys.version_info[:3]), "versions": versions,
                  "cuda_available": bool(torch.cuda.is_available())}))
'''
    try:
        completed = subprocess.run(
            [str(worker_python), "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report(False, f"CoachAI worker dependency probe: {type(exc).__name__}: {exc}",
               "Check the worker interpreter path and its installed packages")
        return

    if completed.returncode != 0:
        detail = (completed.stdout.strip() or completed.stderr.strip())[-1200:]
        report(False, "CoachAI worker dependency probe failed", detail)
        return

    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        report(False, "CoachAI worker dependency probe returned invalid output",
               completed.stdout[-1200:] or completed.stderr[-1200:])
        return

    version = result.get("python", [])
    report(len(version) >= 2 and version[:2] == [3, 12],
           f"CoachAI worker Python {'.'.join(map(str, version))}",
           "Recreate .venv-coachai using python3.12 -m venv .venv-coachai")

    versions = result.get("versions", {})
    for distribution, prefix in WORKER_DISTRIBUTIONS.items():
        actual = versions.get(distribution, "missing")
        report(actual.startswith(prefix),
               f"worker {distribution} {actual} (pinned {prefix}.x)",
               ".venv-coachai/bin/python -m pip install -r requirements-coachai.txt")

    report(bool(result.get("cuda_available")),
           "CoachAI worker PyTorch sees the GPU (optional; CPU is supported)",
           "Check the Windows NVIDIA driver and WSL2 GPU support", required=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coachai-root", type=Path, default=Path("../external/CoachAI-Plus"))
    parser.add_argument("--shuttleset-root", type=Path, default=Path("data/raw/shuttleset/set"))
    parser.add_argument("--worker-python", type=Path, default=default_worker_python())
    args = parser.parse_args()

    # Parent requirements are deliberately independent of CoachAI's older NumPy/Pandas pins.
    parent_version = sys.version_info
    report(parent_version[:2] == (3, 12),
           f"Parent Python {parent_version.major}.{parent_version.minor}.{parent_version.micro}",
           "Create/activate the parent .venv using python3.12 -m venv .venv")

    try:
        import badminton_rl  # noqa: F401
        report(True, "badminton_rl is importable in the parent environment")
    except Exception as exc:  # noqa: BLE001
        report(False, f"import badminton_rl in parent: {type(exc).__name__}: {exc}",
               "Activate .venv and run python -m pip install -e .")

    inspect_worker(args.worker_python)

    csvs = list(args.shuttleset_root.glob("*/set*.csv")) if args.shuttleset_root.exists() else []
    report(len(csvs) == 104,
           f"ShuttleSet CSVs in {args.shuttleset_root}: {len(csvs)} (expected 104)",
           "Run bash scripts/setup/fetch_external.sh")

    root = args.coachai_root
    report(root.is_dir(), f"CoachAI+ checkout exists at {root}",
           "See docs/SETUP.md Part F (fork and clone)")
    if root.is_dir():
        for relative_path in COACHAI_FILES:
            report((root / relative_path).is_file(), f"CoachAI+ file {relative_path}")

        try:
            actual_head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            report(False, f"could not read CoachAI+ Git HEAD: {exc}",
                   "Check that the path is a valid Git checkout")
        else:
            report(actual_head == COACHAI_PLUS_SHA,
                   f"CoachAI+ HEAD {actual_head} (required exact pin {COACHAI_PLUS_SHA})",
                   f"git -C {root} checkout --detach {COACHAI_PLUS_SHA}")

        try:
            status = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            report(False, f"could not read CoachAI+ Git status: {exc}")
        else:
            clean = not status.strip()
            report(clean, "CoachAI+ working tree is clean",
                   "Preserve generated outputs, move them outside the checkout, then rerun git status")

    print("\nALL REQUIRED CHECKS PASSED" if not failed else f"\n{len(failed)} REQUIRED CHECK(S) FAILED")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
