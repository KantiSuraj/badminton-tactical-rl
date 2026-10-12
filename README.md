# badminton-tactical-rl

Physics-aware tactical RL environment for badminton singles, built on CoachAI+ and ShuttleSet.
Start with **`docs/SETUP.md`** (step-by-step, written for Windows 11 + WSL2).

| Folder | What it is |
|---|---|
| `src/badminton_rl/dynamics/` | shuttle flight, player motion, reachability, shot execution, action masks (placeholders, being calibrated) |
| `src/badminton_rl/data/` | ShuttleSet loader and court-coordinate conversion |
| `scripts/audit/` | programs that audit the data and produce `reports/audit/*` |
| `scripts/setup/` | fetch pinned external repos, check your setup |
| `reports/audit/AUDIT.md` | what the audit found |
| `configs/third_party_manifest.yaml` | commits, licences and checksums of everything external |
| `tests/` | `pytest -q` |

Third-party code is **never copied into this repo**. It lives next to it in `../external/` (see `docs/SETUP.md`).
