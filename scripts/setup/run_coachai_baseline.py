"""Run the UNTOUCHED CoachAI+ baseline (benchmark B0) without editing the fork.

CoachAI+ cannot be started the way its README suggests. This wrapper does what the authors' own
API does plus one fix, and leaves the fork's files exactly as upstream:

  1. puts the fork root AND <fork>/BadmintonEnv on sys.path (Environment.py does `from Utils...`)
  2. changes into the fork root (it reads ./input_data/... and ./BadmintonEnv/Agent/... by relative path)
  3. if PyTorch cannot see a GPU, makes `torch.load` default to map_location="cpu"
     (hyperparameters.pkl and INV_weight.pth are CUDA-saved and upstream loads them with no
     map_location, so it crashes without a GPU). With a GPU this patch is skipped.

Usage (venv active, from the project root):
    python scripts/setup/run_coachai_baseline.py
    python scripts/setup/run_coachai_baseline.py --root ../external/CoachAI-Plus \
        --player "CHOU Tien Chen" --opponent "Kento MOMOTA"
It simulates 5 rallies (the value hard-coded in upstream RunEnv.py) and writes
<fork>/output_data/output_game_1.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../external/CoachAI-Plus"))
    ap.add_argument("--player", default="CHOU Tien Chen")
    ap.add_argument("--opponent", default="Kento MOMOTA")
    a = ap.parse_args()

    root = a.root.resolve()
    if not (root / "BadmintonEnv" / "RunEnv.py").exists():
        sys.exit(f"CoachAI+ not found at {root}. See docs/SETUP.md part F.")
    sys.path[:0] = [str(root), str(root / "BadmintonEnv")]
    os.chdir(root)

    import torch

    if not torch.cuda.is_available():
        print("[wrapper] no GPU visible to PyTorch: torch.load will default to the CPU")
        _orig_load = torch.load

        def _cpu_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return _orig_load(*args, **kwargs)

        torch.load = _cpu_load

    from BadmintonEnv.RunEnv import main as run

    t0 = time.time()
    rows = run(a.player, a.opponent)
    print(f"[wrapper] done in {time.time() - t0:.1f}s; {len(rows)} stroke rows; output in {root / 'output_data'}")


if __name__ == "__main__":
    main()
