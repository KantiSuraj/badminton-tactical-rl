#!/usr/bin/env python3
"""Run random masked-policy episodes against the REAL worker and record everything.

Purpose: produce (1) golden payloads for fixtures and (2) the statistics needed to size
the event history and to find rally shapes the serializer has not met yet (fallbacks,
unplayable transitions, A-ending rallies, rallies ending during reset, ...).

Nothing here changes behaviour: it only drives WorkerClient, optionally passes every
payload through the existing EpisodeRewardLedger, and writes files under reports/.

    COACHAI_ROOT="$PWD/../external/CoachAI-Plus" \
    B2_WORKER_PYTHON="$PWD/.venv-coachai/bin/python" \
    python scripts/capture_b2_payloads.py --episodes 500

Outputs (default reports/b2_capture/):
    payloads.jsonl   one line per reset/step payload.  The 1350-entry action_mask is stored as
                     `action_mask_true_indices` (the mask sha256 is kept) to keep the file small.
    summary.json     statistics + every failure with its full error details.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in payload.items() if key != "action_mask"}
    mask = payload.get("action_mask")
    if isinstance(mask, list):
        out["action_mask_true_indices"] = [i for i, flag in enumerate(mask) if flag]
    return out


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def pct(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "count": len(ordered), "min": ordered[0], "mean": round(statistics.fmean(ordered), 3),
        "median": statistics.median(ordered), "p90": pct(0.90), "p95": pct(0.95),
        "p99": pct(0.99), "max": ordered[-1],
    }


def run_capture(
    *, client: Any, episodes: int, base_seed: int, out_dir: Path,
    ledger_factory: Callable[[], Any] | None, error_types: tuple[type[BaseException], ...],
    max_reset_retries: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_per_rally: list[int] = []
    decisions_per_rally: list[int] = []
    counters: dict[str, Counter] = {
        name: Counter() for name in (
            "event_type", "controller", "actor", "native_shot_id", "shot_family_id",
            "last_event_shape", "winner_side", "reset_status", "decision_count_of_failed",
        )
    }
    anomalies: Counter = Counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    started = time.time()

    with (out_dir / "payloads.jsonl").open("w", encoding="utf-8") as sink:
        def write(episode: int, seed: int, kind: str, step: int, action: int | None, payload: dict) -> None:
            sink.write(json.dumps({
                "episode": episode, "seed": seed, "kind": kind, "step": step,
                "action_id": action, "payload": compact(payload),
            }, separators=(",", ":")) + "\n")

        for episode in range(episodes):
            seed = base_seed + episode
            rng = random.Random(seed)
            ledger = ledger_factory() if ledger_factory else None
            events: list[dict[str, Any]] = []
            payload_fallback_ids: list[str] = []
            decisions = 0
            payload: dict[str, Any] = {}
            try:
                payload = client.reset(seed, episode_index=episode, options=None)
                write(episode, seed, "reset", 0, None, payload)
                counters["reset_status"][payload.get("status")] += 1
                if ledger is not None:
                    ledger.begin_episode(payload)
                events += payload["rally_events_delta"]
                payload_fallback_ids += [e["fallback_event_id"] for e in payload["fallback_events"]]
                while payload["ready_for_policy"]:
                    valid = [i for i, flag in enumerate(payload["action_mask"]) if flag]
                    action = rng.choice(valid)
                    payload = client.step(action)
                    decisions += 1
                    write(episode, seed, "step", decisions, action, payload)
                    if ledger is not None:
                        ledger.record_step(action, payload)
                    events += payload["rally_events_delta"]
                    payload_fallback_ids += [e["fallback_event_id"] for e in payload["fallback_events"]]
            except error_types as exc:
                counters["decision_count_of_failed"][decisions] += 1
                failures.append({
                    "episode": episode, "seed": seed, "decisions_before_failure": decisions,
                    "error_type": type(exc).__name__, "code": getattr(exc, "code", None),
                    "message": str(exc)[:1500], "details": getattr(exc, "details", None),
                })
                continue

            completed += 1
            decisions_per_rally.append(decisions)
            events_per_rally.append(len(events))
            terminal = payload.get("terminal") or {}
            counters["winner_side"][terminal.get("winner_side")] += 1
            for event in events:
                counters["event_type"][event["event_type"]] += 1
                counters["controller"][event["controller"]] += 1
                counters["actor"][event["actor"]] += 1
                counters["native_shot_id"][event["native_shot_id"]] += 1
                counters["shot_family_id"][event["shot_family_id"]] += 1
            if events:
                last = events[-1]
                counters["last_event_shape"][f'{last["actor"]}/{last["controller"]}/{last["event_type"]}'] += 1
            if [e["event_index"] for e in events] != list(range(len(events))):
                anomalies["event_index_not_0_to_n_minus_1"] += 1
            if [e["ball_round"] for e in events] != list(range(1, len(events) + 1)):
                anomalies["ball_round_not_1_to_n"] += 1
            fallback_event_ids = [e["fallback_event_id"] for e in events if e["event_type"] == "FALLBACK_STROKE"]
            if sorted(payload_fallback_ids) != sorted(fallback_event_ids):
                anomalies["fallback_audit_ids_differ_from_fallback_events"] += 1
            if any(e["decision_index"] > decisions for e in events):
                anomalies["event_decision_index_above_decision_count"] += 1

    summary = {
        "episodes_requested": episodes, "episodes_completed": completed,
        "episodes_failed": len(failures), "seconds": round(time.time() - started, 1),
        "base_seed": base_seed, "max_reset_retries": max_reset_retries,
        "ledger_checked": ledger_factory is not None,
        "events_per_rally": describe(events_per_rally),
        "decisions_per_rally": describe(decisions_per_rally),
        "counts": {name: {str(k): v for k, v in counter.most_common()} for name, counter in counters.items()},
        "anomalies": dict(anomalies),
        "failures": failures[:25], "failure_count_total": len(failures),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=20260101)
    parser.add_argument("--max-reset-retries", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--coachai-root", default=os.environ.get("COACHAI_ROOT"))
    parser.add_argument("--worker-python", default=os.environ.get("B2_WORKER_PYTHON"))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "b2_capture")
    parser.add_argument("--no-ledger", action="store_true", help="do not pass payloads through EpisodeRewardLedger")
    args = parser.parse_args()
    if not args.coachai_root or not args.worker_python:
        parser.error("set COACHAI_ROOT and B2_WORKER_PYTHON (or pass --coachai-root/--worker-python)")

    from badminton_rl.adapters.protocol import ProtocolError, RemoteWorkerError
    from badminton_rl.adapters.worker_client import WorkerClient, WorkerClientError, WorkerConfig, WorkerTimeouts

    error_types: list[type[BaseException]] = [RemoteWorkerError, WorkerClientError, ProtocolError]
    ledger_factory = None
    if not args.no_ledger:
        try:
            from badminton_rl.env.reward import EpisodeRewardLedger, RewardConfig, RewardIntegrityError
        except ImportError as exc:
            print(f"[capture] reward ledger unavailable ({exc}); continuing without it", file=sys.stderr)
        else:
            error_types.append(RewardIntegrityError)
            config = RewardConfig(max_decisions=args.max_decisions, player_name="CHOU Tien Chen",
                                  opponent_name="Kento MOMOTA")
            ledger_factory = lambda: EpisodeRewardLedger(config)  # noqa: E731

    args.out_dir.mkdir(parents=True, exist_ok=True)
    worker_config = WorkerConfig.from_project(
        PROJECT_ROOT, coachai_root=args.coachai_root, worker_python=args.worker_python,
        max_decisions=args.max_decisions, max_reset_retries=args.max_reset_retries,
        fallback_events_path=args.out_dir / "fallback_events.jsonl",
        stderr_log_path=args.out_dir / "worker_stderr.log",
    )
    timeouts = WorkerTimeouts(health=180.0, reset=60.0, step=60.0, close=15.0)
    with WorkerClient(worker_config, timeouts=timeouts) as client:
        summary = run_capture(
            client=client, episodes=args.episodes, base_seed=args.base_seed, out_dir=args.out_dir,
            ledger_factory=ledger_factory, error_types=tuple(error_types),
            max_reset_retries=args.max_reset_retries,
        )
    print(json.dumps({k: summary[k] for k in (
        "episodes_requested", "episodes_completed", "episodes_failed", "events_per_rally",
        "decisions_per_rally", "anomalies")}, indent=2))
    print("last_event_shape:", summary["counts"]["last_event_shape"])
    print(f"[capture] wrote {args.out_dir}/summary.json and payloads.jsonl")
    return 0 if summary["episodes_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())