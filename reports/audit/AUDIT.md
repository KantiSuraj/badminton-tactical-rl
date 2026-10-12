# Audit of ShuttleSet and the CoachAI environment

Pinned upstream commit `45517f7` (2026-07-17). Reproduce with `scripts/audit/audit_shuttleset.py`
and `scripts/audit/compare_flight_models.py`; raw numbers are in `shuttleset_audit.json` and the CSVs.

## 1. What was and was not audited
Audited: ShuttleSet (all 44 matches), the `StrategicEnvironment` ball and player models (`core.py`) and its
data tables. **Not done:** running the CoachAI environment, ShuttleSet22, opponent-model checkpoints, RallyNet.
**"CoachAI+" itself was not found in this repository**; what exists is the AAAI-24 "CoachAI Badminton Environment".
Whether the supervisor has the CoachAI+ code decides what the B2 baseline is.

## 2. ShuttleSet: facts that change how the data must be used
| Finding | Detail |
|---|---|
| Counts vs README | 44 matches, 104 sets match. Rallies 3,683 (README 3,685), strokes 36,484 (README 36,492): small unexplained gap |
| Shot labels | 19, not 18: an undocumented `unknown` label covers 1,407 strokes (3.9%); `passive drop` is spelled differently in data and README |
| README schema is stale | data also has `server, hit_height, hit_area, hit_x, hit_y, win_reason, flaw, db` |
| Clock | `time` has 1 s resolution; use `frame_num`. **25 videos are 30 fps, 19 are 25 fps** (estimated per video by regression) |
| Clock quality | 240 within-rally strokes with non-increasing `frame_num` (35 decreasing, 205 equal); 2 videos whose `time` and frame clocks drift by 4.2 s and 18.1 s; 2 `ball_round` gaps; 6 same-player-twice |
| Coordinates | camera pixels. `court_units = H @ pixel` (README states the inverse in words). Court = doubles outline, 300 x 660 units = 6.10 x 13.40 m, so **1 unit = 2.033 cm (x) / 2.030 cm (y)**, net at y = 480 |
| Homography inconsistency | 24 of 44 matches map the court corners to x in [27.5, 327.5] instead of [25, 325] (5 cm shift, y exact). Corrected per match in the loader |
| Floor-plane only | the homography is valid for the floor. Player feet are fine; airborne shuttle positions (`hit_*`, mid-rally `landing_*`) carry an unknown parallax error |
| Missingness | `landing_height` 17.3%, `hit_x` 13.0% (first strokes), player/landing x about 3.2%; `lose/win_reason` 90% (rally end only) |

## 3. CoachAI environment: what the source actually does
* It **already contains** a ball flight model and a player speed model. The report's "verify directly" caution was warranted.
* Ball: closed-form drag, `g = 9.8`, `V_T = 6.86` m/s (the code comment beside it says 5.84-6.48), mass 5.19 g. The world default is
  `resistance_force_n = 1`, **linear drag**. For the same terminal speed the quadratic model used here gives (`compare_flight_models.py`):
  35 m/s clear at 35 degrees: 10.9 m / 2.05 s versus CoachAI 19.8 m / 3.08 s; 50 m/s clear: 13.1 m versus 31.4 m (court is 13.4 m).
  Its alternative `n = 2` branch is close (range within 2-16%) but continues the ascent solution past the apex, so times come out 2-11% short.
  **CoachAI's ball physics is therefore not ground truth for this project.** This model's constants are not yet validated either.
* Player: speed is a fitted polynomial of distance per role, applied instantly. No acceleration limit, no reaction delay, no net or
  boundary constraint beyond clamping targets. Speed cap `10/2*100` = 500 units/s, which is 10 m/s at 2 cm per unit.
* Units: hard-coded `pixel * 2 / 100` (2.00 cm; measured 2.03) and `frame / 30` (wrong for the 25 fps videos, 43% of matches).
* Flight-time table `ball_type_model.csv` (mean frames per shot type) is far closer to the interval between a player's two
  consecutive hits (two flights; median ratio 1.05, spread 0.18) than to one flight (median 2.05, spread 0.72). The generating
  script is not in the repo, so this is an inference, not a confirmed bug. CoachAI uses it as single-shot flight time.
* Fitted launch angles (`ball_initial_state_model.csv`, radians): clear 0.32 (19 deg), lob 0.64 (36), drop 0.10 (6), smash -0.04 (-2),
  net shot 0.67 (38). They disagree with this project's placeholders (clear 42, smash -24) and share the table's provenance question.
* Portability: Python 3.8.10, old `gym`, PySide6 GUI, Windows-style backslash paths in `read_csv`. Expect porting work for WSL.

## 4. Placeholder ledger
| Parameter | Placeholder | Evidence | Status |
|---|---|---|---|
| Court units, fps | assumed | 2.033/2.030 cm per unit; 25 or 30 fps per video | **measured**, in the loader |
| Hit-to-hit flight time per shot type | none | e.g. clear 1.17 s, lob 1.28, net 0.88, drop 0.80, smash 0.47 (`flight_time_by_type.csv`) | **measured**, usable as calibration targets |
| Player `v_max` | 4.5 m/s | average speed between own hits: p50 1.36, p95 2.62, p99 3.15 m/s, max 7.4; 0.14% above 4.5, none above 8 | data reject limits below about 3.5; 4.5 is consistent but not identified |
| Player `a_max`, `a_brake`, reaction | 7, 9, 0.20 s | **not identifiable**: the distance envelope saturates at court size. An envelope fit was tried and removed | still uncalibrated; needs tracked trajectories |
| Contact reach radius | 1.1 m | floor-projected reach: p50 0.82, p75 1.15, p90 1.55, p95 1.9 m (parallax-biased) | 1.1 m excludes about 25% of observed contacts; 1.5 m is a defensible provisional value |
| Recovery time budget (turnaround) | 0.7 s | hit-to-hit time over all shots: median 0.88 s, p10 0.50, p90 1.33 | within range; make it shot-dependent later |
| Shuttle terminal speed, mass | 6.8 m/s, 5.1 g | CoachAI uses 6.86 and 5.19 g with a contradicting comment; no measurement | **unvalidated** |
| Shot families | 10 invented types, nominal angles | data have 18 types (+unknown); provisional mapping in `FAMILY_OF`. Height-aware check (section 5): SMASH range too steep, CLEAR/LIFT hang 1.4-1.65x too long | **recalibrate** (next step) |

## 5. Placeholder check with the height-aware solver
`ShotExecutionModel.solve_launch(..., target_height=h)` now aims at the point where the opponent contacts the
shuttle: it passes over (x, y) descending through height h (h = 0 is the old floor landing). Verified against the full flight
model by the new `HEIGHT_CROSSING` event (`tests/test_shot_height.py`). Aiming higher at the same ground point needs a *higher*
speed, gives a *shorter* flight time and a floor landing further beyond; a smash that is a net collision when aimed at the floor
is feasible when aimed at a 1.0 m contact point.

Each observed stroke is solved with contact at the family's mid contact height, target = the observed contact point, and the
receiving height implied by the NEXT stroke's family (observed; e.g. low after a smash, since 79% of smashes are answered with a
`return net`). Time ratio = solver time-to-target / observed hit-to-hit time (1.0 = reproduces the data); 150 strokes per family.

| Family | feasible (floor) | time ratio (floor) | feasible (height-aware) | time ratio (height-aware) | mean receiving h (m) | n |
|---|---|---|---|---|---|---|
| BLOCK | 0.83 | 1.126 | 0.97 | 0.799 | 1.09 | 150 |
| CLEAR | 0.99 | 1.981 | 0.93 | 1.647 | 2.42 | 150 |
| CUT | 0.87 | 1.12 | 0.97 | 0.811 | 1.1 | 150 |
| DRIVE | 0.67 | 1.643 | 0.93 | 0.985 | 1.48 | 150 |
| DROP | 0.97 | 1.452 | 0.99 | 1.256 | 1.03 | 150 |
| HALF_SMASH | 0.97 | 1.406 | 0.87 | 1.038 | 1.24 | 150 |
| LIFT | 0.99 | 1.615 | 0.99 | 1.357 | 2.39 | 150 |
| NET_SHOT | 0.85 | 1.26 | 0.82 | 0.959 | 0.98 | 150 |
| PUSH | 0.96 | 0.993 | 0.67 | 0.842 | 2.15 | 150 |
| SMASH | 0.27 | 1.288 | 0.02 | 0.465* | 1.27 | 150 |

\* SMASH height-aware ratio rests on 3 feasible shots out of 150: not meaningful.

What it says, and what it does not:
* **SMASH is essentially infeasible (2%)**: every failure is `target_too_far_for_family_speed`. Observed smashes cover a median 9.6 m
  horizontally and are received low, so from about 2.65 m they must descend only about 1-2 m: flatter than the placeholder's -10 degree limit.
  CoachAI's fitted smash angle (-2 degrees) points the same way. The `-50..-10` range is probably wrong.
* **CLEAR (1.65x) and LIFT (1.36x) hang too long**; DROP is 1.26x. Their nominal elevations are probably too high. BLOCK, CUT and PUSH come out
  0.8x (slightly too fast); DRIVE, HALF_SMASH and NET_SHOT are within about 5%.
* **PUSH fails 48/100 with `target_height_not_reached`**: its receiving height averages 2.15 m (opponents often answer from the rear court), which the
  low, flat push family cannot reach. Either the push prior or the next-stroke height proxy is off; the data cannot say which.
* Receiving heights come from my placeholder contact ranges (not observed), the hitter's contact height is the family midpoint (not observed), and the
  airborne contact point is parallax-biased. These ratios guide where to calibrate first; they are not calibrated values.

## 6. Next steps, in order
1. Decide what the baseline is: obtain CoachAI+ code, or use the AAAI-24 environment (and port it to Linux).
2. ~~Make `solve_launch` height-aware~~ **Done** (this section 5).
3. Write `calibrate_families.py`: fit each family's elevation range and speed range so the predicted time-to-contact matches the observed table. Decide first how to
   treat the unobserved hitter contact height and the receiving height (fit them as nuisance parameters, or take them from tracked data).
4. Decide whether to keep 10 merged shot families or adopt ShuttleSet's real labels.
5. Let `ActionFeasibility` use height-aware contact targets once the families are calibrated (it still aims at floor cells, which is right for "where would it land").
6. Only then audit ShuttleSet22 and the opponent models.