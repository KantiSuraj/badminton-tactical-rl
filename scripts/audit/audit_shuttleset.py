"""Audit ShuttleSet and derive the numbers that replace placeholder physics parameters.

Usage (from the project root):
    export SHUTTLESET_ROOT=path/to/CoachAI-Projects/ShuttleSet/set
    python scripts/audit/audit_shuttleset.py --out reports/audit \
        --coachai-root path/to/CoachAI-Projects      # optional: compares against CoachAI tables

Writes  <out>/shuttleset_audit.json  and CSV tables, and prints a summary. Sections:
  A  integrity           counts vs README, labels, missingness, ordering, clock consistency
  B  flight time         seconds between consecutive strokes per shot type (per-video fps!)
  C  player kinematics   speed / distance-in-time envelope -> v_max, a_max priors
  D  contact reach       floor-projected distance between player and shuttle at contact
  E  CoachAI comparison  flight-time table and unit conversions used by CoachAI
  F  placeholder check   do the placeholder shot families reproduce the observed shots?

Interpretation limits are printed with each result; read them before using a number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from badminton_rl.data.load_shuttleset import DEFAULT_ROOT, load_matches, load_strokes

README_CLAIMS = {"matches": 44, "sets": 104, "rallies": 3685, "strokes": 36492, "shot_types": 18}
KEY = ["match_id", "set", "rally"]


def pct(a, qs=(5, 25, 50, 75, 90, 95, 99)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs} if len(a) else {}


# ---------------------------------------------------------------- A. integrity

def section_integrity(df: pd.DataFrame) -> dict:
    g = df.groupby(KEY, sort=False)
    ball_gap = g["ball_round"].diff().dropna()
    first = g["ball_round"].first()
    same_player = (df["player"] == g["player"].shift(1)) & g["player"].shift(1).notna()
    frame_back = g["frame_num"].diff() <= 0
    counts = {
        "matches": int(df.match_id.nunique()),
        "sets": int(df.groupby(["match_id", "set"]).ngroups),
        "rallies": int(g.ngroups),
        "strokes": int(len(df)),
        "shot_types": int(df.type.nunique()),
    }
    # clock consistency: how far frame_num strays from a straight line vs the 1-second `time`
    drift = {}
    for vid, d in df.groupby("video"):
        sec = pd.to_timedelta(d["time"]).dt.total_seconds().to_numpy()
        res = d["frame_num"].to_numpy() / d["fps"].iloc[0] - sec
        drift[vid] = float(np.abs(res - np.median(res)).max())
    bad_clock = {k: round(v, 1) for k, v in drift.items() if v > 2.0}
    miss = df[["time", "frame_num", "type", "player_location_x", "opponent_location_x", "hit_x",
               "landing_x", "hit_height", "landing_height", "lose_reason", "win_reason"]].isna().mean().round(4)
    return {
        "counts": counts,
        "readme_claims": README_CLAIMS,
        "count_differences": {k: counts[k] - README_CLAIMS[k] for k in README_CLAIMS},
        "unknown_shot_strokes": int((df.type_en == "unknown").sum()),
        "shot_type_counts": df.type_en.value_counts().to_dict(),
        "rallies_not_starting_at_1": int((first != 1).sum()),
        "ball_round_gaps": int((ball_gap != 1).sum()),
        "same_player_twice_in_a_row": int(same_player.sum()),
        "frame_not_increasing": int(frame_back.sum()),
        "frame_decreasing": int((g["frame_num"].diff() < 0).sum()),
        "frame_equal": int((g["frame_num"].diff() == 0).sum()),
        "fps_by_video": df.groupby("video").fps.first().round(2).to_dict(),
        "fps_counts": df.groupby("video").fps.first().round().value_counts().to_dict(),
        "videos_with_clock_drift_over_2s": bad_clock,
        "missing_fraction": miss.to_dict(),
        "outside_court_fraction": {
            k: float(((df[f"{k}_x_m"].abs() > 3.5) | (df[f"{k}_y_m"].abs() > 7.4)).sum() / df[f"{k}_x_m"].notna().sum())
            for k in ("player", "opponent", "landing")
        },
    }


# ---------------------------------------------------------------- B. flight time

def add_flight_time(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEY, sort=False)
    df = df.copy()
    df["next_frame"] = g["frame_num"].shift(-1)
    df["dt_frames"] = df["next_frame"] - df["frame_num"]
    df["dt_s"] = df["dt_frames"] / df["fps"]
    df["next_type_en"] = g["type_en"].shift(-1)
    return df


def section_flight_time(df: pd.DataFrame, out: Path) -> dict:
    ok = df[(df.dt_s > 0.15) & (df.dt_s < 6.0)]
    t = ok.groupby("type_en").agg(n=("dt_s", "size"), mean_s=("dt_s", "mean"), median_s=("dt_s", "median"),
                                  p10_s=("dt_s", lambda x: np.percentile(x, 10)),
                                  p90_s=("dt_s", lambda x: np.percentile(x, 90)),
                                  mean_frames=("dt_frames", "mean")).round(3)
    t.to_csv(out / "flight_time_by_type.csv")
    return {
        "usable_pairs": int(len(ok)),
        "excluded_pairs_outside_0.15_6s": int(df.dt_s.notna().sum() - len(ok)),
        "resolution_s_at_25fps": 0.04, "resolution_s_at_30fps": round(1 / 30, 4),
        "table": t.to_dict("index"),
        "note": "dt is the time between consecutive hits, i.e. flight time to the CONTACT point "
                "(includes the shuttle being hit above the floor), not time-to-floor.",
    }


# ---------------------------------------------------------------- C. player kinematics

def section_kinematics(df: pd.DataFrame, out: Path) -> dict:
    g = df.groupby(KEY, sort=False)
    d = df.copy()
    for c in ("frame_num", "player", "player_x_m", "player_y_m"):
        d[f"{c}_2"] = g[c].shift(-2)
    ok = (d["player"] == d["player_2"]) & d["player_x_m"].notna() & d["player_x_m_2"].notna()
    d = d[ok].copy()
    d["T"] = (d["frame_num_2"] - d["frame_num"]) / d["fps"]
    d["D"] = np.hypot(d["player_x_m_2"] - d["player_x_m"],
                      d["player_y_m_2"] - d["player_y_m"])
    d = d[(d["T"] > 0.4) & (d["T"] < 8.0)]
    d["v_avg"] = d["D"] / d["T"]
    jump = d["v_avg"] > 10.0                    # not physically plausible: tracking/homography jump
    clean = d[~jump]
    bins = [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 8.0]
    clean = clean.assign(bin=pd.cut(clean["T"], bins))
    env = clean.groupby("bin", observed=True).agg(n=("D", "size"), T_mean=("T", "mean"),
                                                   D_p50=("D", lambda x: np.percentile(x, 50)),
                                                   D_p90=("D", lambda x: np.percentile(x, 90)),
                                                   D_p95=("D", lambda x: np.percentile(x, 95)),
                                                   D_p99=("D", lambda x: np.percentile(x, 99))).round(3)
    env.to_csv(out / "kinematics_envelope.csv")
    # One-sided test of candidate speed limits: the data can REJECT a limit that is too low
    # (many observed average speeds above it) but cannot prove a limit that is high enough.
    exceed = {}
    for v in (3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0):
        exceed[f"v>{v}"] = {"all": round(float((clean["v_avg"] > v).mean()), 5),
                            "interval<1.2s": round(float((clean.loc[clean["T"] < 1.2, "v_avg"] > v).mean()), 5)}
    return {
        "pairs": int(len(d)), "dropped_as_jumps_over_10mps": int(jump.sum()),
        "avg_speed_mps": pct(clean["v_avg"]), "distance_m": pct(clean["D"]), "interval_s": pct(clean["T"]),
        "max_clean_avg_speed_mps": round(float(clean["v_avg"].max()), 2),
        "envelope_table": env.reset_index(drop=True).assign(bin=env.index.astype(str)).to_dict("records"),
        "avg_speed_exceedance": exceed,
        "avg_speed_p999_mps": round(float(clean["v_avg"].quantile(0.999)), 2),
        "note": "v_avg is the average speed between two hits of the same player, a LOWER bound on peak "
                "speed. The distance envelope saturates at court size (about 5 m) for intervals above ~2 s, "
                "so (v_max, a_max) are NOT identifiable from it (an earlier least-squares fit hit its grid "
                "boundary with 2.3 m RMSE and was removed). Use exceedance only to reject limits that are "
                "too LOW; peak speed needs tracked player trajectories (blueprint 7.4).",
    }


# ---------------------------------------------------------------- D. contact reach

def section_reach(df: pd.DataFrame, out: Path) -> dict:
    d = df[df.hit_x_m.notna() & df.player_x_m.notna() & ~df.type_en.isin(["short service", "long service", "unknown"])].copy()
    d["reach_m"] = np.hypot(d.hit_x_m - d.player_x_m, d.hit_y_m - d.player_y_m)
    t = d.groupby("type_en").reach_m.agg(n="size", median="median", p90=lambda x: np.percentile(x, 90),
                                         p99=lambda x: np.percentile(x, 99)).round(3)
    t.to_csv(out / "contact_reach_by_type.csv")
    return {"overall_m": pct(d.reach_m), "by_type": t.to_dict("index"),
            "note": "Floor-projected distance from the player's position to the shuttle's position at contact. "
                    "The shuttle is airborne, so its floor projection has an unknown parallax error: treat as an "
                    "indicative scale for ContactEnvelope.horizontal_radius, NOT a measurement."}


# ---------------------------------------------------------------- E. CoachAI comparison

def section_coachai(df: pd.DataFrame, coachai_root: Path) -> dict:
    p = coachai_root / "CoachAI Badminton Environment" / "StrategicEnvironment" / "Data" / "ball_type_model.csv"
    env = pd.read_csv(p, index_col=0)
    g = df.groupby(KEY, sort=False)
    f1 = (g["frame_num"].shift(-1) - df["frame_num"])
    f2 = (g["frame_num"].shift(-2) - df["frame_num"])
    one = df.assign(f=f1)[(f1 > 3) & (f1 < 200)].groupby("type_en").f.mean()   # stroke n -> n+1 (one flight)
    two = df.assign(f=f2)[(f2 > 3) & (f2 < 400)].groupby("type_en").f.mean()   # stroke n -> n+2 (two flights)
    j = pd.DataFrame({"coachai_mean_frames": env["mean"], "ours_n_to_n+1_frames": one, "ours_n_to_n+2_frames": two}).dropna()
    j["ratio_vs_n+1"] = j.coachai_mean_frames / j["ours_n_to_n+1_frames"]
    j["ratio_vs_n+2"] = j.coachai_mean_frames / j["ours_n_to_n+2_frames"]
    n25 = float((df.groupby("video").fps.first().round() == 25).mean())
    return {
        "table": j.round(2).to_dict("index"),
        "ratio_vs_one_flight": {"median": round(float(j["ratio_vs_n+1"].median()), 2), "std": round(float(j["ratio_vs_n+1"].std()), 2)},
        "ratio_vs_two_flights": {"median": round(float(j["ratio_vs_n+2"].median()), 2), "std": round(float(j["ratio_vs_n+2"].std()), 2)},
        "share_of_matches_at_25fps": round(n25, 3),
        "unit_scale_cm_per_unit": {"x": round(6.10 / 300 * 100, 3), "y": round(13.40 / 660 * 100, 3), "coachai_assumes": 2.0},
        "note": "CoachAI's per-type 'frame' table is far closer to the interval between a player's consecutive OWN "
                "hits (two flights) than to one flight. The script that generates ball_type_model.csv is not in "
                "the repository, so this is an inference. CoachAI uses these values as the flight time of a single "
                "shot (core.py: frame/30). Independently of that, frame/30 is wrong for the 25 fps videos "
                "(43% of matches).",
    }


# ---------------------------------------------------------------- F. placeholder check

FAMILY_OF = {  # PROVISIONAL mapping from ShuttleSet labels to the placeholder ShotType families
    "clear": "CLEAR", "drop": "DROP", "smash": "SMASH", "wrist smash": "HALF_SMASH",
    "drive": "DRIVE", "driven flight": "DRIVE", "back-court drive": "DRIVE", "defensive return drive": "DRIVE",
    "push": "PUSH", "rush": "PUSH", "lob": "LIFT", "defensive return lob": "LIFT",
    "net shot": "NET_SHOT", "cross-court net shot": "NET_SHOT", "return net": "BLOCK", "passive drop": "CUT",
}


def section_placeholders(df: pd.DataFrame, n_per_family: int, seed: int, z_recv=(1.0, 1.5, 2.0)) -> dict:
    """Do the placeholder shot families reproduce observed shots? Height-aware version.

    Each observed stroke becomes a solver query: contact at the hitter's floor position and the
    family's midpoint contact height, target = the observed (floor-projected) point where the
    opponent contacts the shuttle, at a receiving height h. The solver's flight time is the time
    to pass over that point at h, directly comparable with the observed hit-to-hit time. The
    receiving height is NOT observed in ShuttleSet, so a few values are tried (sensitivity).
    h = 0 is the old floor-target baseline, kept for comparison. "h=by_next_stroke" uses the
    receiving height implied by the (observed) next stroke type: the midpoint of that placeholder
    family's contact-height range, e.g. low after a smash (returns are mostly 'return net').
    """
    from badminton_rl.dynamics.shot_execution import DEFAULT_FAMILIES, ContactState, ShotExecutionModel, ShotType

    ex = ShotExecutionModel()
    rng = np.random.default_rng(seed)
    d = df[df.type_en.isin(FAMILY_OF) & df.landing_x_m.notna() & df.player_x_m.notna() & (df.dt_s > 0.15) & (df.dt_s < 6.0)].copy()
    d["family"] = d.type_en.map(FAMILY_OF)
    # keep shots that cross the net and land inside the singles court (what the solver can represent)
    d = d[(d.player_y_m * d.landing_y_m < 0) & (d.landing_x_m.abs() <= 2.59) & (d.landing_y_m.abs() <= 6.7)]
    heights = (0.0,) + tuple(z_recv)
    d["next_family"] = d.next_type_en.map(FAMILY_OF)
    res = {}
    for fam, grp in d.groupby("family"):
        grp = grp.sample(min(n_per_family, len(grp)), random_state=int(rng.integers(1 << 31)))
        lo, hi = DEFAULT_FAMILIES[ShotType[fam]].contact_height_range
        z0 = 0.5 * (lo + hi)               # contact height is NOT observed; use the family midpoint
        per_h = {}
        for h in heights:
            feas, ratios, reasons = 0, [], {}
            for r in grp.itertuples():
                sol = ex.solve_launch(ContactState((r.player_x_m, r.player_y_m, z0)), ShotType[fam],
                                      (r.landing_x_m, r.landing_y_m), verify=False, target_height=h)
                if sol.feasible:
                    feas += 1
                    ratios.append(sol.flight_time_s / r.dt_s)
                else:
                    reasons[sol.reason] = reasons.get(sol.reason, 0) + 1
            per_h[f"h={h}"] = {
                "feasible_rate": round(feas / len(grp), 3),
                "median_time_ratio": round(float(np.median(ratios)), 3) if ratios else None,
                "p10_p90_time_ratio": [round(float(np.percentile(ratios, 10)), 3), round(float(np.percentile(ratios, 90)), 3)] if ratios else None,
                "failure_reasons": reasons,
            }
        # receiving height implied by the NEXT stroke (observed): midpoint of that family's contact range
        nxt = grp[grp.next_family.notna()]
        feas, ratios, reasons, hs = 0, [], {}, []
        for r in nxt.itertuples():
            nlo, nhi = DEFAULT_FAMILIES[ShotType[r.next_family]].contact_height_range
            hh = 0.5 * (nlo + nhi)
            hs.append(hh)
            sol = ex.solve_launch(ContactState((r.player_x_m, r.player_y_m, z0)), ShotType[fam],
                                  (r.landing_x_m, r.landing_y_m), verify=False, target_height=hh)
            if sol.feasible:
                feas += 1
                ratios.append(sol.flight_time_s / r.dt_s)
            else:
                reasons[sol.reason] = reasons.get(sol.reason, 0) + 1
        per_h["h=by_next_stroke"] = {
            "n": int(len(nxt)), "mean_receiving_height_m": round(float(np.mean(hs)), 2) if hs else None,
            "feasible_rate": round(feas / len(nxt), 3) if len(nxt) else None,
            "median_time_ratio": round(float(np.median(ratios)), 3) if ratios else None,
            "p10_p90_time_ratio": [round(float(np.percentile(ratios, 10)), 3), round(float(np.percentile(ratios, 90)), 3)] if ratios else None,
            "failure_reasons": reasons,
        }
        res[fam] = {"n": int(len(grp)), "assumed_contact_height_m": z0, "by_target_height": per_h}
    return {"per_family": res, "receiving_heights_tested_m": list(z_recv),
            "note": "time_ratio = solver time-to-target / observed hit-to-hit time (1.0 = reproduces the data). "
                    "Still confounded by: unobserved hitter contact height (family midpoint used), unobserved "
                    "receiving height (sensitivity over h), parallax of the airborne contact point, and the "
                    "family priors themselves. Indicative, not a calibration."}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("reports/audit"))
    ap.add_argument("--coachai-root", type=Path, default=None)
    ap.add_argument("--placeholder-sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-placeholders", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = add_flight_time(load_strokes(a.root).sort_values(["match_id", "set", "rally", "ball_round"], kind="stable").reset_index(drop=True))
    report = {"integrity": section_integrity(df)}
    report["flight_time"] = section_flight_time(df, a.out)
    report["kinematics"] = section_kinematics(df, a.out)
    report["contact_reach"] = section_reach(df, a.out)
    if a.coachai_root:
        report["coachai_comparison"] = section_coachai(df, a.coachai_root)
    if not a.skip_placeholders:
        report["placeholder_check"] = section_placeholders(df, a.placeholder_sample, a.seed)
    (a.out / "shuttleset_audit.json").write_text(json.dumps(report, indent=2, default=float, ensure_ascii=False))
    print(f"wrote {a.out / 'shuttleset_audit.json'}")


if __name__ == "__main__":
    main()