"""Telemetry endpoints: dashboard-friendly rollups of validator state for miner self-diagnosis."""

import os
import subprocess
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import STATE_DIR
from helpers.sanitize import _sanitize_floats, _safe_json_load
from state_store import read_state

router = APIRouter()


def _short_reason(r):
    if not r:
        return ""
    r = str(r)
    return r if len(r) <= 220 else r[:217] + "..."


def _apply_dq_annotation(entry, dq_map, uid_hotkey_map):
    if not entry or not dq_map:
        return entry
    hk_by_uid = {int(uid): hk for uid, hk in (uid_hotkey_map or {}).items()
                 if str(uid).lstrip("-").isdigit()}
    dq_blocked = []
    for r in entry.get("results") or []:
        if r.get("disqualified"):
            continue
        uid = r.get("uid")
        if uid is None or uid < 0:
            continue
        hk = hk_by_uid.get(int(uid))
        reason = None
        if hk:
            for k, v in dq_map.items():
                if k == hk or k.startswith(hk + ":"):
                    reason = v
                    break
        if not reason and str(uid) in dq_map:
            reason = dq_map[str(uid)]
        if reason:
            short = reason if len(reason) <= 120 else reason[:117] + "..."
            r["disqualified"] = True
            r["dq_reason"] = reason
            if r.get("vs_king") and "dethroned" in r["vs_king"]:
                dq_blocked.append({"uid": uid, "kl": r.get("kl"), "reason": short})
            r["vs_king"] = f"DQ — not crowned ({short})"
            r["dethrone_eligible"] = False
    if dq_blocked and not entry.get("king_retained_reason"):
        names = ", ".join(f"UID {d['uid']}" for d in dq_blocked)
        entry["king_retained_reason"] = (
            f"lower-KL challenger(s) {names} disqualified ({dq_blocked[0]['reason']})"
        )
        entry["dq_blocked_dethrone"] = dq_blocked
    return entry


def _dq_entry(key, reason, hk_to_uid):
    e = {"key": key, "reason": _short_reason(reason)}
    if ":" in key:
        hk, blk = key.split(":", 1)
        e["hotkey"] = hk
        e["block"] = int(blk) if blk.isdigit() else blk
        if hk in hk_to_uid:
            e["uid"] = int(hk_to_uid[hk])
    elif key in hk_to_uid:
        e["hotkey"] = key
        e["uid"] = int(hk_to_uid[key])
    return e


@router.get("/api/telemetry/overview", tags=["Telemetry"],
            summary="Full dashboard telemetry snapshot")
def telemetry_overview():
    """Recent DQs + private-pool commit + current round + recent events + king probe + composite axes."""
    now = time.time()

    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {}) or {}
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {}) or {}
    hk_to_uid = {v: k for k, v in uid_map.items()}

    recent_dqs = sorted(
        (_dq_entry(k, r, hk_to_uid) for k, r in dq.items()),
        key=lambda e: int(e.get("block") or 0),
        reverse=True,
    )[:40]

    current = read_state("current_round.json", {}) or {}
    commit = read_state("private_pool_commit.json", {}) or {}
    reveal = read_state("private_pool_reveal.json", {}) or {}
    last_eval = read_state("last_eval.json", {}) or {}
    h2h = read_state("h2h_latest.json", {}) or {}
    _apply_dq_annotation(h2h, dq, uid_map)

    validator_log = _safe_json_load(os.path.join(STATE_DIR, "validator_log.json"), [])
    if not isinstance(validator_log, list):
        validator_log = []
    recent_events = validator_log[-50:]

    king_probe = None
    king_uid = h2h.get("king_uid")
    king_model = h2h.get("king_model")
    students = last_eval.get("students", {})
    if king_model and king_model in students:
        ks = students[king_model] or {}
        king_probe = {
            "uid": king_uid,
            "model": king_model,
            "status": ks.get("status"),
            "kl": ks.get("kl_global_avg"),
            "capability": _compact_capability(ks.get("capability")),
            "length_axis": ks.get("length_axis"),
            "think_probe": _compact_think(ks.get("think_probe")),
            "adversarial": ks.get("adversarial"),
            "load_time": ks.get("load_time"),
        }

    round_detail = _compact_round(h2h, last_eval)

    return JSONResponse(
        content=_sanitize_floats({
            "server_time": now,
            "current_round": current,
            "private_pool": {
                "current": commit,
                "latest_reveal": reveal,
            },
            "king_probe": king_probe,
            "round_detail": round_detail,
            "recent_dqs": recent_dqs,
            "recent_events": recent_events,
        }),
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=15"},
    )


def _compact_capability(cap):
    if not cap:
        return None
    items = cap.get("items") or []
    compact_items = []
    for it in items[:30]:
        compact_items.append({
            "q": (it.get("q") or "")[:160],
            "expected": str(it.get("expected") or "")[:40],
            "pred": str(it.get("pred") or "")[:60],
            "ok": it.get("ok"),
        })
    return {
        "n": cap.get("n"),
        "correct": cap.get("correct"),
        "pass_frac": cap.get("pass_frac"),
        "teacher_pass_frac": cap.get("teacher_pass_frac"),
        "items": compact_items,
    }


def _compact_think(tp):
    if not tp:
        return None
    samples = []
    for s in (tp.get("samples") or [])[:6]:
        samples.append({
            "prompt": (s.get("prompt") or "")[:80],
            "gen_tokens": s.get("gen_tokens"),
            "terminated": s.get("terminated"),
            "gzip_ratio": s.get("gzip_ratio"),
            "distinct_4": s.get("distinct_4"),
            "top_6gram_rate": s.get("top_6gram_rate"),
            "tail": (s.get("tail") or "")[:220],
        })
    return {
        "pass": tp.get("pass"),
        "reason": tp.get("reason"),
        "prompts_tested": tp.get("prompts_tested"),
        "prompts_terminated": tp.get("prompts_terminated"),
        "prompts_degenerate": tp.get("prompts_degenerate"),
        "mean_gen_tokens": tp.get("mean_gen_tokens"),
        "self_bleu_across_prompts": tp.get("self_bleu_across_prompts"),
        "teacher_self_bleu": tp.get("teacher_self_bleu"),
        "samples": samples,
    }


def _compact_bench(payload):
    """Compact representation of a *_bench probe payload for the dashboard."""
    if not payload:
        return None
    items = payload.get("items") or []
    compact = []
    for it in items[:12]:
        entry = {
            "src": (it.get("src") or "")[:40],
            "ok": it.get("ok"),
            "reason": (it.get("reason") or "")[:80] if it.get("reason") else None,
            "pred": (str(it.get("pred") or "")[:80]) if it.get("pred") is not None else None,
            "gold": (str(it.get("gold") or "")[:40]) if it.get("gold") is not None else None,
            "task_id": it.get("task_id"),
        }
        if "tool_used" in it:
            entry["tool_used"] = bool(it.get("tool_used"))
            tr = it.get("tool_result")
            if tr is not None:
                entry["tool_result"] = str(tr)[:120]
        if "samples" in it and "vote_winner" in it:
            for k in ("samples", "vote_winner", "vote_count", "k"):
                entry[k] = it.get(k)
        compact.append(entry)
    out = {
        "n": payload.get("n"),
        "correct": payload.get("correct"),
        "pass_frac": payload.get("pass_frac"),
        "wall_s": payload.get("wall_s"),
        "error": payload.get("error"),
        "items": compact,
        "mean_gen_tokens": payload.get("mean_gen_tokens"),
        "mean_gen_tokens_correct": payload.get("mean_gen_tokens_correct"),
    }
    if payload.get("tool_used_count") is not None:
        out["tool_used_count"] = payload["tool_used_count"]
    if payload.get("k_samples") is not None:
        out["k_samples"] = payload["k_samples"]
        out["temperature"] = payload.get("temperature")
        out["top_p"] = payload.get("top_p")
    return out


_BENCH_AXES = (
    "math_bench", "code_bench", "reasoning_bench", "knowledge_bench", "ifeval_bench",
    "aime_bench", "mbpp_bench", "tool_use_bench", "self_consistency_bench",
    "arc_bench", "truthful_bench", "long_context_bench",
    "procedural_bench", "robustness_bench", "noise_resistance_bench",
)


def _compact_round(h2h, last_eval):
    if not h2h:
        return None
    results = h2h.get("results") or []
    students = (last_eval or {}).get("students", {})
    rows = []
    for r in results:
        model = r.get("model")
        s = students.get(model, {}) if model else {}
        cap = s.get("capability") or {}
        length = s.get("length_axis") or {}
        tp = s.get("think_probe") or {}
        adv = s.get("adversarial") or {}
        jp = s.get("judge_probe") or {}
        ctp = s.get("chat_turns_probe") or {}
        row = {
            "uid": r.get("uid"),
            "model": model,
            "kl": r.get("kl"),
            "is_king": r.get("is_king"),
            "vs_king": r.get("vs_king"),
            "disqualified": r.get("disqualified"),
            "dq_reason": r.get("dq_reason"),
            "dethrone_eligible": r.get("dethrone_eligible"),
            "early_stopped": r.get("early_stopped"),
            "prompts_scored": r.get("prompts_scored"),
            "prompts_total": r.get("prompts_total"),
            "paired_prompts": r.get("paired_prompts"),
            "composite": r.get("composite") or {},
            "capability_pass_frac": cap.get("pass_frac"),
            "capability_teacher": cap.get("teacher_pass_frac"),
            "length_ratio": length.get("ratio"),
            "length_penalty": length.get("penalty"),
            "think_pass": tp.get("pass"),
            "think_reason": tp.get("reason"),
            "adversarial_pass_frac": adv.get("pass_frac"),
            "adversarial_mean_tokens": adv.get("mean_gen_tokens"),
            "judge_mean_score": jp.get("mean_score"),
            "judge_normalized": jp.get("normalized"),
            "judge_n_valid": jp.get("n_valid"),
            "judge_n": jp.get("n"),
            "chat_turns_mean_score": ctp.get("mean_score"),
            "chat_turns_normalized": ctp.get("normalized"),
            "chat_turns_n_valid": ctp.get("n_valid"),
            "chat_turns_n": ctp.get("n"),
            "chat_turns_n_turns": ctp.get("n_turns"),
        }
        for axis in _BENCH_AXES:
            row[axis] = _compact_bench(s.get(axis))
        rows.append(row)
    return {
        "block": h2h.get("block"),
        "block_hash": (h2h.get("block_hash") or "")[:16],
        "timestamp": h2h.get("timestamp"),
        "king_uid": h2h.get("king_uid"),
        "prev_king_uid": h2h.get("prev_king_uid"),
        "king_changed": h2h.get("king_changed"),
        "new_king_uid": h2h.get("new_king_uid"),
        "king_retained_reason": h2h.get("king_retained_reason"),
        "dq_blocked_dethrone": h2h.get("dq_blocked_dethrone"),
        "n_prompts": h2h.get("n_prompts"),
        "n_students": h2h.get("n_students"),
        "elapsed_seconds": h2h.get("elapsed_seconds"),
        "epsilon": h2h.get("epsilon"),
        "paired_test_alpha": h2h.get("paired_test_alpha"),
        "results": rows,
    }


@router.get(
    "/api/telemetry/dqs", tags=["Telemetry"],
    summary="Recent disqualifications",
    description="Tail of `state/disqualified.json` with UID/hotkey annotations. Sorted newest-first. Cache 15s.",
)
def telemetry_dqs(limit: int = 80):
    limit = max(1, min(limit, 300))
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {}) or {}
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {}) or {}
    hk_to_uid = {v: k for k, v in uid_map.items()}
    out = sorted(
        (_dq_entry(k, r, hk_to_uid) for k, r in dq.items()),
        key=lambda e: (e.get("block") or 0),
        reverse=True,
    )
    return JSONResponse(
        content={"disqualified": out[:limit], "count": len(out)},
        headers={"Cache-Control": "public, max-age=15"},
    )


@router.get(
    "/api/telemetry/events", tags=["Telemetry"],
    summary="Validator event stream (info/warn/error)",
    description=(
        "Returns the tail of `state/validator_log.json` — structured events "
        "from the live validator (`scripts/validator/service.py`). Filter by "
        "`level=info|warn|error` to focus on a specific severity. Cache: "
        "`max-age=5` so the dashboard event feed stays roughly real-time "
        "without hammering disk."
    ),
)
def telemetry_events(limit: int = 200, level: str = None):
    limit = max(1, min(limit, 500))
    entries = _safe_json_load(os.path.join(STATE_DIR, "validator_log.json"), [])
    if not isinstance(entries, list):
        entries = []
    if level:
        entries = [e for e in entries if (e.get("level") or "").lower() == level.lower()]
    return JSONResponse(
        content={"entries": entries[-limit:], "count": len(entries)},
        headers={"Cache-Control": "public, max-age=5"},
    )


@router.get(
    "/api/telemetry/errors", tags=["Telemetry"],
    summary="Recent error / warning events",
    description=(
        "Convenience wrapper around `/api/telemetry/events` that filters to "
        "`warn`, `warning`, and `error` levels — the surface the dashboard's "
        "incidents widget reads. Cache: `max-age=5`."
    ),
)
def telemetry_errors(limit: int = 100):
    limit = max(1, min(limit, 300))
    entries = _safe_json_load(os.path.join(STATE_DIR, "validator_log.json"), [])
    if not isinstance(entries, list):
        entries = []
    filt = [e for e in entries
            if (e.get("level") or "").lower() in ("warn", "warning", "error")]
    return JSONResponse(
        content={"entries": filt[-limit:], "count": len(filt)},
        headers={"Cache-Control": "public, max-age=5"},
    )


@router.get(
    "/api/telemetry/pod-health", tags=["Telemetry"],
    summary="GPU / pod telemetry",
    description=(
        "Live `nvidia-smi` snapshot (per-GPU util, memory, temp, power) plus "
        "the validator's systemd state and the current eval phase. Used by "
        "the dashboard pod-health widget and by `sn97_healthcheck.py`. "
        "Cache: `max-age=5`."
    ),
)
def telemetry_pod_health():
    out = {"gpu": None, "pod": None, "validator_uptime_s": None}
    nvidia_csv_fmt = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        r = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={nvidia_csv_fmt}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            gpus = []
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "util_pct": float(parts[2] or 0),
                        "mem_used_mb": float(parts[3] or 0),
                        "mem_total_mb": float(parts[4] or 0),
                        "temp_c": float(parts[5] or 0),
                        "power_w": float(parts[6] or 0),
                        "power_limit_w": float(parts[7] or 0),
                    })
            out["gpu"] = gpus
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["systemctl", "show", "distil-validator",
             "--property=ActiveState,SubState,ExecMainStartTimestampMonotonic"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            props = {}
            for line in r.stdout.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    props[k] = v
            out["validator"] = {
                "active_state": props.get("ActiveState"),
                "sub_state": props.get("SubState"),
            }
    except Exception:
        pass

    progress = read_state("eval_progress.json", {}) or {}
    out["pod"] = {
        "eval_active": bool(progress.get("active")),
        "phase": progress.get("phase"),
        "started_at": progress.get("started_at"),
        "estimated_duration_s": progress.get("estimated_duration_s"),
    }
    return JSONResponse(
        content=_sanitize_floats(out),
        headers={"Cache-Control": "public, max-age=5"},
    )


@router.get(
    "/api/telemetry/king-diagnostic", tags=["Telemetry"],
    summary="Per-round king probe / status detail",
    description=(
        "Last `n` rounds of king state derived from `state/h2h_history.json`: "
        "block, timestamp, king UID/model, dethrone outcome, paired KL, and "
        "DQ-blocked-dethrone metadata. Newest round first. Cache: `max-age=10`."
    ),
)
def telemetry_king_diagnostic(n: int = 10):
    n = max(1, min(n, 50))
    history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    if not isinstance(history, list):
        history = []
    out = []
    for entry in history[-n:]:
        results = entry.get("results") or []
        king_uid = entry.get("king_uid")
        king_res = next((r for r in results if r.get("uid") == king_uid), None)
        out.append({
            "block": entry.get("block"),
            "timestamp": entry.get("timestamp"),
            "king_uid": king_uid,
            "king_model": entry.get("king_model"),
            "king_changed": entry.get("king_changed"),
            "new_king_uid": entry.get("new_king_uid"),
            "prev_king_uid": entry.get("prev_king_uid"),
            "king_retained_reason": entry.get("king_retained_reason"),
            "dq_blocked_dethrone": entry.get("dq_blocked_dethrone"),
            "king_kl": (king_res or {}).get("kl") if king_res else entry.get("king_kl"),
            "king_status": (king_res or {}).get("status") or (king_res or {}).get("vs_king"),
            "king_dq": (king_res or {}).get("disqualified"),
            "king_dq_reason": (king_res or {}).get("dq_reason"),
        })
    out.reverse()
    return JSONResponse(
        content=_sanitize_floats(out),
        headers={"Cache-Control": "public, max-age=10"},
    )
