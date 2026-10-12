# Setup guide (Windows 11 + WSL2), baby steps

> **What was tested, and what was not.** The steps in parts D-F were executed on a Linux machine with Python 3.12, no GPU and the pinned
> package versions: `pip install -r requirements-coachai.txt` resolves cleanly, all 161 tests pass, the data fetch script copies 104 CSVs, and the
> baseline wrapper simulates five rallies in about 5 s. **Not tested:** WSL2 itself, Ubuntu 22.04 / Python 3.10, and a visible GPU.
> The Windows-side steps (part A) and the GitHub steps (C3, F2-F4) are written from standard procedure, not run here.

You will end with this layout. Everything lives in the **Linux** filesystem (`~`), never under `/mnt/c` (it is slow and breaks file permissions).

```
~/projects/
├── badminton-tactical-rl/        ← YOUR repo (git, GitHub). All your code lives here.
│   ├── .venv/                    ← the Python environment (not in git)
│   ├── data/raw/shuttleset/set/  ← ShuttleSet CSVs (not in git)
│   └── src/ scripts/ tests/ ...
└── external/                     ← other people's code, next to your repo, never inside it
    ├── CoachAI-Projects/         ← read-only clone (source of the ShuttleSet data)
    └── CoachAI-Plus/             ← YOUR FORK of the baseline environment (you edit this on your own branches)
```

Two rules. **(1)** Never copy folders from one repo into another; your code *calls* external code through adapters.
**(2)** Pin every external repo to a commit; they are recorded in `configs/third_party_manifest.yaml`.

---
## Part A. One-time machine setup

**A1. Install WSL2 with Ubuntu 22.04.** Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu-22.04
```
Reboot if asked, open "Ubuntu 22.04" from the Start menu, and create a Linux username and password. Check it is version 2:
```powershell
wsl -l -v        # Ubuntu-22.04 should show VERSION 2
```

**A2. GPU (optional).** Update the NVIDIA driver in Windows. Inside Ubuntu, `nvidia-smi` should list your RTX 4050. CPU is enough for this project's RL; the GPU only matters later for pose/shuttle tracking.

**A3. Tools inside Ubuntu:**
```bash
sudo apt update
sudo apt install -y git unzip build-essential python3-pip python3.10-venv
python3 --version          # must print 3.10.x (CoachAI+ is pinned to 3.10)
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

**A4. GitHub SSH key** (so pushing works without passwords):
```bash
ssh-keygen -t ed25519 -C "you@example.com"      # press Enter 3 times
cat ~/.ssh/id_ed25519.pub                        # copy the whole line
```
GitHub → Settings → SSH and GPG keys → New SSH key → paste. Then `ssh -T git@github.com` should greet you by username.

## Part B. Make the folders
```bash
mkdir -p ~/projects/external
cd ~/projects
```

## Part C. Bring the existing work into your repo
**C1.** Download `badminton-tactical-rl.zip` from the chat to your Windows Downloads folder, then:
```bash
cp /mnt/c/Users/<WindowsUser>/Downloads/badminton-tactical-rl.zip ~/projects/
cd ~/projects && unzip badminton-tactical-rl.zip && cd badminton-tactical-rl
```
**C2.** Make it a git repo and make the first commit:
```bash
git init -b main
git add -A
git commit -m "Initial import: dynamics modules, ShuttleSet loader, audit scripts"
```
**C3.** On github.com create a **new empty private repository** named `badminton-tactical-rl` (no README, no .gitignore), then:
```bash
git remote add origin git@github.com:<your-user>/badminton-tactical-rl.git
git push -u origin main
```
**C4.** Sanity check: `git ls-files | wc -l` should be small (about 35 files). If you see thousands, something that should be ignored got added.

## Part D. The Python environment (one venv for your repo and CoachAI+)
```bash
cd ~/projects/badminton-tactical-rl
python3.10 -m venv .venv
source .venv/bin/activate          # do this in EVERY new terminal; the prompt shows (.venv)
pip install -U pip
pip install -r requirements-coachai.txt     # big download (PyTorch + CUDA libraries, roughly 3 GB to download and 5.5 GB on disk); be patient
pip install -e ".[dev]"                     # installs YOUR package so `import badminton_rl` works
```
These are the exact versions CoachAI+ pins. Do **not** upgrade them.

## Part E. Get the ShuttleSet data
```bash
bash scripts/setup/fetch_external.sh        # use `bash`; downloads lose the "executable" flag
pytest -q                                   # expect: 161 passed
```
The script clones `CoachAI-Projects` into `../external/` at the audited commit and copies the 104 stroke CSVs to `data/raw/shuttleset/set/`.
If it prints "UNEXPECTED FILE COUNT", upstream changed: re-run the audit before trusting any number.

## Part F. Fork CoachAI+ and run the untouched baseline (the week-1 gate)
**F1. Licence first.** The CoachAI+ repo has **no licence file**, so by default nobody has permission to redistribute or modify it. Private research use of a fork is the usual practice, but email the authors (NYCU ADSL lab) or ask your supervisor for permission before publishing anything derived from it. Note the answer in `configs/third_party_manifest.yaml`.

**F2. Fork.** Open https://github.com/KuangDW/CoachAI-Plus and click **Fork** (top right) → create the fork under your account.

**F3. Clone your fork** (not the original) into `external/` and link the original as `upstream`:
```bash
cd ~/projects/external
git clone git@github.com:<your-user>/CoachAI-Plus.git
cd CoachAI-Plus
git remote add upstream https://github.com/KuangDW/CoachAI-Plus.git
git fetch upstream
```
**F4. Pin the audited commit on a branch you will never edit:**
```bash
git checkout -B baseline-pinned 4fd05934f172eec16c74d05c2902983b64c8521e
git push -u origin baseline-pinned
```
This branch is your untouched reference (benchmark "B0"). Your changes go on a *different* branch created later (`git checkout -b tactical-rl-adapter`).

**F5. Run it** (venv active; run from YOUR repo, not from the fork):
```bash
cd ~/projects/badminton-tactical-rl
python scripts/setup/run_coachai_baseline.py
```
Use this wrapper instead of `python -m BadmintonEnv.RunEnv`: the upstream command **fails** with `No module named 'Utils'`. The wrapper fixes three problems *without editing the fork*: (1) `Environment.py` imports `Utils...` so `BadmintonEnv/` must also be on the Python path, (2) it reads `./input_data/...` by relative path so the working directory must be the fork root, (3) its saved weights and `hyperparameters.pkl` are CUDA tensors loaded without `map_location`, so without a visible GPU it crashes; the wrapper then loads them on the CPU.

You should see five rallies print (`rally : 1`, `score: [...]`, `round: ...`), then `[wrapper] done in ... s`, and a new `output_game_1.csv` in `external/CoachAI-Plus/output_data/`. Results differ between runs (the opponents sample). Keep that file out of the fork's git status:
```bash
echo "output_data/output_game_1.csv" >> ~/projects/external/CoachAI-Plus/.git/info/exclude
```
If it crashes, copy the **full error** to me.

**F6. Check everything at once:**
```bash
cd ~/projects/badminton-tactical-rl
python scripts/setup/check_setup.py          # OK everywhere (GPU may be WARN)
```

## Part G. How the pieces connect (without copying anything)
| Piece | Where | Role | You edit it? |
|---|---|---|---|
| `badminton-tactical-rl` | `~/projects/` | your environment, physics, data, RL | **yes** |
| `CoachAI-Plus` (your fork) | `external/` | baseline simulator + pretrained opponents | only on your own branches |
| `CoachAI-Projects` | `external/` | source of ShuttleSet CSVs | no |

Your code reaches CoachAI+ through one **adapter** file (`adapters/coachai_env_adapter.py`, next step) that: finds the fork via a `COACHAI_PLUS_ROOT` setting, switches into the fork's folder while it runs (because of the relative paths), and converts its non-Gymnasium `reset`/`step` (whose `step` returns a different number of values depending on the path) to Gymnasium's. If its dependencies ever clash with yours, the adapter talks to it through a separate process instead. The coordinate frames differ (CoachAI+ is hitter-referenced, units of about 2.03 cm; yours is metres with the net at y = 0), so the adapter converts at the boundary and the tests check the round trip.

## Part H. Later (months 5-6): perception tools, each in its OWN venv
Do not install these now. They cannot share an environment with CoachAI+:
* **TrackNetV3** pins Python 3.8, PyTorch 1.10, NumPy 1.22. That PyTorch is old enough that it very likely has no kernels for the RTX 4050 (it is newer than that build supports), so expect CPU or a port. Verify before planning on it. Weights come from a Google Drive link; record their checksum.
* **MMPose** (Apache-2.0) needs matching `mmengine`, `mmcv`, `mmdet` builds for the PyTorch you choose.
* **3DShuttleTracking** has no licence and needs a Roboflow key and a Hugging Face token. Read it as a reference only; do not install or copy it.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No module named 'Utils'` | you ran upstream's command directly; use `scripts/setup/run_coachai_baseline.py` |
| `No module named 'torchsde'` (or `seaborn`, `matplotlib`) | `pip install -r requirements-coachai.txt` (they are in that file) |
| `Attempting to deserialize object on a CUDA device` | same wrapper (it loads on the CPU when no GPU is visible) |
| `FileNotFoundError: ./input_data/...` | the working directory is not the fork root (the wrapper does this for you) |
| `ModuleNotFoundError: badminton_rl` | venv not active, or run `pip install -e ".[dev]"` |
| `Permission denied` on a `.sh` file | prefix with `bash` |
| everything is slow | the repo is under `/mnt/c`; move it to `~/projects` |
| `pip install torch` runs out of disk | free about 10 GB in WSL, or `wsl --manage Ubuntu-22.04 --resize` |

## Done when
1. `pytest -q` passes in `badminton-tactical-rl`.
2. `python scripts/setup/run_coachai_baseline.py` simulates five rallies.
3. `python scripts/setup/check_setup.py` shows no FAIL.
4. `git log` shows your first commit on GitHub, and `baseline-pinned` exists in your fork.
