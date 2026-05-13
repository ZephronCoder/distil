"""Miner-related endpoints: scores, commitments, model info, miner details, compare."""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import EPOCH_BLOCKS, MAX_BATCH_UIDS, MAX_COMPARE_UIDS, STATE_DIR
from external import get_commitments as fetch_commitments_data, get_model_info as fetch_model_info_data
from helpers.cache import _get_stale
from helpers.dq import _dq_reason_for_commitment
from helpers.h2h import compact_round, index_by_uid, load_history, rounds_for_uid, uid_stats
from helpers.sanitize import _safe_json_load, _sanitize_floats
from state_store import (
    disqualified,
    h2h_latest,
    h2h_tested_against_king,
    current_round,
    eval_progress,
    last_eval,
    read_state,
    scores as load_scores,
    top4_leaderboard,
    uid_hotkey_map,
)

router = APIRouter()


def _failure_matches_commitment(fail_entry: str, commitment: dict) -> bool:
    """Match failure_models entries (repo or "repo@revision") vs commitment."""
    repo = (commitment or {}).get("model")
    rev = (commitment or {}).get("revision")
    if not repo or not fail_entry:
        return False
    if "@" in fail_entry:
        f_repo, f_rev = fail_entry.split("@", 1)
        return f_repo == repo and (f_rev == rev or not rev)
    return fail_entry == repo


@router.get("/api/commitments", tags=["Miners"], summary="Miner model commitments",
         description="On-chain HuggingFace commitments per miner: `{model, revision, block}`. Cache 60s.")
def get_commitments():
    return fetch_commitments_data()


@router.get("/api/scores", tags=["Miners"], summary="Current KL scores and disqualifications",
         description="Latest KL scores + DQ status: `{scores, ema_scores, disqualified, last_eval, last_eval_time, tempo_seconds}`.")
def get_scores(fields: str = ""):
    s = load_scores()
    result = {
        "scores": s,
        "ema_scores": s,  # backward compat
        "disqualified": disqualified(),
        "last_eval": None,
        "last_eval_time": None,
        "tempo_seconds": 600,
    }
    last_eval_data = last_eval()
    if last_eval_data is not None:
        result["last_eval"] = last_eval_data
        try:
            result["last_eval_time"] = os.path.getmtime(os.path.join(STATE_DIR, "last_eval.json"))
        except OSError:
            result["last_eval_time"] = last_eval_data.get("timestamp")
        result["last_eval_block"] = last_eval_data.get("block")
        result["last_eval_type"] = last_eval_data.get("type")
    if fields:
        requested = {f.strip() for f in fields.split(",")}
        result = {k: v for k, v in result.items() if k in requested}
    return JSONResponse(
        content=_sanitize_floats(result),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/model-info/{model_path:path}", tags=["Miners"], summary="HuggingFace model info",
         description="HuggingFace card metadata: `{params_b, is_moe, num_experts, num_active_experts, tags, license, pipeline_tag, downloads, likes, base_model}`. Cache 1h.")
def get_model_info(model_path: str):
    return fetch_model_info_data(model_path)


@router.get("/api/miner/{uid}", tags=["Miners"], summary="Full miner details by UID",
         description="Everything known about a UID: `{hotkey, coldkey, commitment, kl_score, disqualified, h2h_history (last 10), in_top5, is_king, registered}`.")
def get_miner(uid: int):
    result = {"uid": uid, "registered": False}

    metagraph = _get_stale("metagraph") or {}
    neuron = next((n for n in metagraph.get("neurons", []) if n.get("uid") == uid), None)
    if neuron:
        result["registered"] = True
        for k in ("hotkey", "coldkey", "stake", "incentive", "emission"):
            result[k] = neuron.get(k)
        result["is_validator"] = neuron.get("is_validator", False)
    else:
        result["hotkey"] = None
        result["coldkey"] = None

    # Commitment
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    hotkey = result.get("hotkey")
    # Fall back to uid_hotkey_map.json (refreshed every epoch) if metagraph hotkey is stale.
    if not hotkey or hotkey not in commitments:
        mapped_hk = uid_hotkey_map().get(str(uid))
        if mapped_hk and mapped_hk in commitments:
            hotkey = mapped_hk
            result["hotkey"] = hotkey
    result["commitment"] = commitments.get(hotkey) if hotkey else None

    if isinstance(result.get("commitment"), dict):
        result["model"] = result["commitment"].get("model") or result["commitment"].get("repo")
    else:
        result["model"] = None

    # KL score
    scores = load_scores()
    uid_str = str(uid)
    result["kl_score"] = scores.get(uid_str)

    dq = disqualified()
    result["disqualified"] = _dq_reason_for_commitment(uid, hotkey, result.get("commitment"), dq)

    top4 = top4_leaderboard()
    king = top4.get("king") or {}
    contenders = top4.get("contenders") or []
    result["is_king"] = king.get("uid") == uid
    top5_uids = {c.get("uid") for c in [king, *contenders] if c.get("uid") is not None}
    result["in_top5"] = uid in top5_uids

    # Eval status: why (not) evaluated
    h2h_tracker = h2h_tested_against_king()
    latest = h2h_latest()
    current_king_uid = latest.get("king_uid")
    current_block = latest.get("block", 0)
    tracker_entry = h2h_tracker.get(uid_str, {})
    eval_status = {}
    failures_map = _safe_json_load(os.path.join(STATE_DIR, "failures.json"), default={})
    failure_models_map = _safe_json_load(os.path.join(STATE_DIR, "failure_models.json"), default={})
    fail_count = int(failures_map.get(uid_str, 0) or 0)
    fail_model = failure_models_map.get(uid_str)
    composite_scores = read_state("composite_scores.json", {})
    comp_record = composite_scores.get(uid_str) if isinstance(composite_scores, dict) else None
    evaluated_uids = set(read_state("evaluated_uids.json", []) or [])
    prog = eval_progress() or {}
    rnd = current_round() or {}
    active_slot = next(
        ({**entry, "position": idx}
         for idx, entry in enumerate(prog.get("eval_order") or [], start=1)
         if str(entry.get("uid")) == uid_str),
        None,
    )
    backlog = read_state("eval_backlog.json", {})
    backlog_row = next(
        (row for row in backlog.get("pending") or []
         if isinstance(row, dict) and str(row.get("uid")) == uid_str),
        None,
    )
    if result.get("disqualified"):
        eval_status["status"] = "disqualified"
        eval_status["reason"] = "Model is disqualified and won't be evaluated"
    elif result.get("is_king"):
        eval_status["status"] = "king"
        eval_status["reason"] = "Evaluated every round as the defending king"
    elif fail_count >= 3 and fail_model and _failure_matches_commitment(fail_model, result.get("commitment") or {}):
        eval_status["status"] = "skipped_stale"
        eval_status["reason"] = (
            f"Skipped for {fail_count} consecutive rounds due to eval errors on the same model+revision "
            f"({fail_model}). Push a new HuggingFace revision, or commit a new model_repo on-chain, to reset."
        )
        eval_status["failure_count"] = fail_count
    elif active_slot:
        current_model = prog.get("current_model")
        completed = {str(u) for u in (prog.get("completed") or [])}
        if uid_str in completed:
            slot_status = "done"
        elif active_slot.get("model") and active_slot.get("model") == current_model:
            slot_status = "running"
        else:
            slot_status = "queued_active_round"
        info = (rnd.get("models_to_eval") or {}).get(uid_str) or {}
        eval_status["status"] = slot_status
        eval_status["reason"] = (
            f"In the active eval round at slot {active_slot['position']}"
            + (f" ({prog.get('phase')})" if prog.get("phase") else "")
        )
        eval_status["position"] = active_slot["position"]
        eval_status["round_block"] = rnd.get("block") or prog.get("block")
        eval_status["commit_block"] = info.get("commit_block")
        eval_status["revision"] = info.get("revision")
    elif backlog_row and backlog_row.get("status") == "deferred":
        eval_status["status"] = "deferred"
        eval_status["reason"] = (
            "Deferred by the single-eval round cap; FIFO order by commit_block "
            "will put it into a later round."
        )
        eval_status["round_cap"] = backlog.get("round_cap")
        eval_status["commit_block"] = backlog_row.get("commit_block")
        eval_status["revision"] = backlog_row.get("revision")
    elif isinstance(comp_record, dict):
        eval_status["status"] = "scored"
        eval_status["reason"] = "Already evaluated; composite score is available."
        eval_status["composite_final"] = comp_record.get("final")
        eval_status["composite_version"] = comp_record.get("version")
        eval_status["scored_at"] = comp_record.get("ts")
    elif uid_str in evaluated_uids:
        eval_status["status"] = "evaluated_no_composite"
        eval_status["reason"] = "Marked evaluated, but no composite score is currently available."
    elif not result.get("kl_score"):
        eval_status["status"] = "queued"
        eval_status["reason"] = "Waiting for first evaluation - new submissions get priority"
    elif tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
        last_block = tracker_entry["block"]
        epochs_since = (current_block - last_block) // EPOCH_BLOCKS if current_block > last_block else 0
        # Single-eval policy: re-test only when commitment changes (new HF
        # revision OR new model_repo) or when the king changes. No time cooldown.
        eval_status["status"] = "tested"
        eval_status["reason"] = (
            f"Tested against current king at block {last_block} ({epochs_since} epoch(s) ago). "
            f"Re-tested only when on-chain commitment changes or king changes."
        )
        eval_status["last_test_block"] = last_block
        eval_status["epochs_since"] = epochs_since
    else:
        eval_status["status"] = "untested"
        eval_status["reason"] = "Not yet tested against the current king - will be scheduled"
    result["eval_status"] = eval_status

    h2h_index = index_by_uid(load_history())
    relevant = [
        {
            "block": item["round"].get("block"),
            "timestamp": item["round"].get("timestamp"),
            "kl": item["row"].get("kl"),
            "is_king": item["row"].get("is_king", False),
            "king_changed": item["round"].get("king_changed", False),
            "type": item["round"].get("type"),
        }
        for item in rounds_for_uid(h2h_index, uid, limit=10)
    ]
    result["h2h_history"] = relevant

    # Per-axis composite — prefer the latest H2H row, else last known from history
    # (keeps the per-miner endpoint stable for dashboards).
    composite_entry = None
    composite_block = latest.get("block")
    for r in (latest.get("results") or []):
        if r.get("uid") == uid and r.get("composite"):
            composite_entry = r["composite"]
            break
    if composite_entry is None:
        for item in rounds_for_uid(h2h_index, uid, limit=25):
            row = item.get("row") or {}
            comp = row.get("composite")
            if comp:
                composite_entry = comp
                composite_block = (item.get("round") or {}).get("block")
                break
    if composite_entry:
        result["composite"] = {
            "final": composite_entry.get("final"),
            "worst_3_mean": composite_entry.get("worst_3_mean"),
            "final_alpha": composite_entry.get("final_alpha"),
            "worst": composite_entry.get("worst"),
            "weighted": composite_entry.get("weighted"),
            "axes": composite_entry.get("axes", {}),
            "broken_axes": composite_entry.get("broken_axes", []),
            "present_count": composite_entry.get("present_count"),
            "version": composite_entry.get("version"),
            "baseline_penalty": composite_entry.get("baseline_penalty"),
            "axes_raw": composite_entry.get("axes_raw"),
            "round_block": composite_block,
            "is_latest_round": composite_block == latest.get("block"),
        }
        kh = composite_entry.get("king_health")
        if kh:
            streak_file = _safe_json_load(
                os.path.join(STATE_DIR, "king_regression_streak.json"), {}
            )
            result["king_health"] = {**kh, "streak": int(streak_file.get(str(uid), 0))}
    else:
        result["composite"] = None

    return JSONResponse(
        content=_sanitize_floats(result),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/evaluated_uids", tags=["Miners"], summary="All evaluated UIDs with scores",
         description="`{uids: [{uid, kl_score, model_id?}], count}` for every evaluated UID.")
def get_evaluated_uids():
    evaluated = _safe_json_load(os.path.join(STATE_DIR, "evaluated_uids.json"), [])
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    result = []
    for uid_str in evaluated:
        uid = int(uid_str) if isinstance(uid_str, str) else uid_str
        entry = {"uid": uid, "kl_score": scores.get(str(uid))}
        hotkey = uid_map.get(str(uid))
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            entry["model_id"] = c.get("model") or c.get("repo")
        result.append(entry)
    result.sort(key=lambda x: x.get("kl_score") or 999)
    return JSONResponse(
        content=_sanitize_floats({"uids": result, "count": len(result)}),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/composite-scores", tags=["Miners"],
         summary="All composite scores",
         description="Full state/composite_scores.json ranked by composite.final, with per-axis breakdown, baseline_penalty, axes_raw, broken_axes, and version. Cache 10s.")
def get_composite_scores():
    composite = _safe_json_load(os.path.join(STATE_DIR, "composite_scores.json"), {})
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {}) if isinstance(commitments_data, dict) else {}
    result = []
    for uid_str, record in (composite.items() if isinstance(composite, dict) else []):
        if not isinstance(record, dict):
            continue
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        hotkey = uid_map.get(uid_str)
        model = None
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            model = c.get("model") or c.get("repo")
        result.append({
            "uid": uid,
            "hotkey": hotkey,
            "model": model or record.get("model"),
            "revision": record.get("revision"),
            "block": record.get("block"),
            "ts": record.get("ts"),
            "version": record.get("version"),
            # v30.2 ranking key
            "final": record.get("final"),
            "worst_3_mean": record.get("worst_3_mean"),
            "final_alpha": record.get("final_alpha"),
            # Legacy fields
            "worst": record.get("worst"),
            "weighted": record.get("weighted"),
            # Axis breakdown
            "axes": record.get("axes", {}),
            "axes_raw": record.get("axes_raw"),
            "baseline_penalty": record.get("baseline_penalty"),
            "broken_axes": record.get("broken_axes", []),
            "n_axes": record.get("n_axes"),
            "axis_spread": record.get("axis_spread"),
            "bench_vs_rel_gap": record.get("bench_vs_rel_gap"),
        })
    # Sort: v30.2 records (with ``final``) first by final desc, then
    # legacy records by worst desc.
    def _sort_key(r):
        f = r.get("final")
        if f is not None:
            return (0, -float(f))
        w = r.get("worst")
        return (1, -(float(w) if w is not None else 0.0))
    result.sort(key=_sort_key)
    return JSONResponse(
        content=_sanitize_floats({
            "scores": result,
            "count": len(result),
            "v30_2_count": sum(1 for r in result if r.get("final") is not None),
            "schema_version": "v30.2",
            "ranking_key": "composite.final = 0.75 * worst_3_mean + 0.25 * weighted",
        }),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/dq_reasons", tags=["Miners"], summary="Disqualified UIDs with reasons",
         description="DQ entries normalized to a list. Keys may be UID, hotkey, or `hotkey:block`.")
def get_dq_reasons():
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
    # Build reverse map: hotkey -> uid
    hk_to_uid = {v: k for k, v in uid_map.items()}
    result = []
    for key, reason in dq.items():
        entry = {"key": key, "reason": reason}
        # Try to resolve UID
        if key.isdigit():
            entry["uid"] = int(key)
        elif ":" in key:
            hotkey = key.split(":")[0]
            if hotkey in hk_to_uid:
                entry["uid"] = int(hk_to_uid[hotkey])
            entry["hotkey"] = hotkey
            entry["block"] = key.split(":")[1]
        elif key in hk_to_uid:
            entry["uid"] = int(hk_to_uid[key])
            entry["hotkey"] = key
        result.append(entry)
    return JSONResponse(
        content={"disqualified": result, "count": len(result)},
        headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"},
    )


@router.get("/api/model_hashes", tags=["Miners"], summary="Model weight hashes for integrity",
         description="SHA256 of safetensor metadata per UID so anyone can verify a miner's weights haven't changed since evaluation.")
def get_model_hashes():
    hashes_raw = _safe_json_load(os.path.join(STATE_DIR, "model_hashes.json"), {})
    # Restructure: group by UID (skip _block and _hotkey auxiliary keys)
    result = {}
    for key, value in hashes_raw.items():
        if "_" in key:
            continue  # skip auxiliary keys like 174_block, 174_hotkey
        result[key] = {
            "hash": value,
            "block": hashes_raw.get(f"{key}_block"),
            "hotkey": hashes_raw.get(f"{key}_hotkey"),
        }
    return JSONResponse(
        content={"hashes": result, "count": len(result)},
        headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"},
    )


@router.get("/api/miner/{uid}/rounds", tags=["Miners"], summary="H2H rounds for a specific miner",
         description="All H2H rounds for a UID (newest first). `?limit=N` (default 50, max 200), `?page=N`. Each: `{block, timestamp, kl, is_king, king_changed, type, king_uid, n_prompts}`.")
def get_miner_rounds(uid: int, limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    try:
        idx = index_by_uid(load_history())
        relevant = [compact_round(it["round"], it["row"]) for it in idx.get(uid, [])]
        total = len(relevant)
        start = (page - 1) * limit
        end = start + limit
        page_data = relevant[start:end]

        return JSONResponse(
            content=_sanitize_floats({
                "uid": uid,
                "rounds": page_data,
                "total": total,
                "page": page,
                "limit": limit,
                "has_more": end < total,
            }),
            headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch miner rounds: {str(e)}"},
        )


@router.get("/api/commitment/{hotkey}", tags=["Miners"], summary="Lookup commitment by hotkey",
         description="Lookup commitment by ss58 hotkey: `{commitment, uid, registered}`. Useful for verifying the validator sees a fresh submission.")
def get_commitment_by_hotkey(hotkey: str):
    result = {"hotkey": hotkey, "registered": False, "uid": None, "commitment": None}

    # Find UID from metagraph
    metagraph = _get_stale("metagraph") or {}
    for n in metagraph.get("neurons", []):
        if n.get("hotkey") == hotkey:
            result["registered"] = True
            result["uid"] = n.get("uid")
            result["coldkey"] = n.get("coldkey")
            result["stake"] = n.get("stake")
            result["incentive"] = n.get("incentive")
            break

    # Commitment data
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    if hotkey in commitments:
        result["commitment"] = commitments[hotkey]

    return JSONResponse(
        content=result,
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/compare", tags=["Miners"], summary="Compare two or more miners",
         description="Side-by-side compare for UIDs (`?uids=2,34,36`). Per UID: KL score, model, H2H round count, best KL, win/loss vs king.")
def compare_miners(uids: str):
    uid_list = [int(u.strip()) for u in uids.split(",") if u.strip().isdigit()][:MAX_COMPARE_UIDS]
    if not uid_list:
        return JSONResponse(status_code=400, content={"error": "Provide ?uids=1,2,3"})

    scores = load_scores()
    uid_map = uid_hotkey_map()
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {}) if isinstance(commitments_data, dict) else {}
    latest = h2h_latest()
    dq = disqualified()
    idx = index_by_uid(load_history())

    result = []
    for uid in uid_list:
        entry = {"uid": uid, "kl_score": scores.get(str(uid))}
        hotkey = uid_map.get(str(uid))
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            entry["model"] = c.get("model") or c.get("repo")
        else:
            entry["model"] = None
        entry["is_king"] = latest.get("king_uid") == uid
        commitment = commitments.get(hotkey) if hotkey else None
        entry["disqualified"] = _dq_reason_for_commitment(uid, hotkey, commitment, dq) is not None
        entry.update(uid_stats(idx.get(uid, [])))
        result.append(entry)

    result.sort(key=lambda x: x.get("kl_score") or 999)
    return JSONResponse(
        content=_sanitize_floats({"miners": result}),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


# ── New compact endpoints (Phase 3) ───────────────────────────────────────────

@router.get("/api/miners/batch", tags=["Miners"], summary="Compact cards for a batch of UIDs",
         description=f"Compact cards per UID with `{{uid, model, kl_score, composite, is_king, disqualified}}`. Composite includes `final`, `worst_3_mean`, `worst`, `weighted`, per-axis. Limit {MAX_BATCH_UIDS} UIDs.")
def miners_batch(uids: str):
    uid_list = [int(u.strip()) for u in uids.split(",") if u.strip().isdigit()][:MAX_BATCH_UIDS]
    if not uid_list:
        return JSONResponse(status_code=400, content={"error": "Provide ?uids=1,2,3"})

    scores = load_scores()
    uid_map = uid_hotkey_map()
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {}) if isinstance(commitments_data, dict) else {}
    latest = h2h_latest()
    dq = disqualified()
    composite_scores_data = read_state("composite_scores.json", {})
    if not isinstance(composite_scores_data, dict):
        composite_scores_data = {}

    miners = []
    for uid in uid_list:
        hotkey = uid_map.get(str(uid))
        model = None
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            model = c.get("model") or c.get("repo")
        composite_summary = None
        comp_record = composite_scores_data.get(str(uid))
        if isinstance(comp_record, dict):
            composite_summary = {
                "final": comp_record.get("final"),
                "worst_3_mean": comp_record.get("worst_3_mean"),
                "worst": comp_record.get("worst"),
                "weighted": comp_record.get("weighted"),
                "version": comp_record.get("version"),
                "axes": comp_record.get("axes", {}),
            }
        miners.append({
            "uid": uid,
            "hotkey": hotkey,
            "model": model,
            "kl_score": scores.get(str(uid)),
            "composite": composite_summary,
            "is_king": latest.get("king_uid") == uid,
            "disqualified": _dq_reason_for_commitment(
                uid, hotkey, commitments.get(hotkey) if hotkey else None, dq
            ) is not None,
        })
    return JSONResponse(
        content=_sanitize_floats({"miners": miners}),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/cumulative-scores", tags=["Miners"], summary="Cumulative KL deltas per UID",
         description="Running cumulative-KL-delta tracker: `{miners: [{uid, cumulative_kl_diff, rounds, best_kl?}]}` sorted by largest delta first.")
def cumulative_scores():
    raw = read_state("cumulative_scores.json", {})
    uid_map = uid_hotkey_map()
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {}) if isinstance(commitments_data, dict) else {}
    miners = []
    for uid_str, info in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(info, dict):
            continue
        try:
            uid_int = int(uid_str)
        except (TypeError, ValueError):
            continue
        hotkey = uid_map.get(uid_str)
        model = None
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            model = c.get("model") or c.get("repo")
        miners.append({
            "uid": uid_int,
            "model": model,
            "cumulative_kl_diff": info.get("cumulative_kl_diff"),
            "rounds": info.get("rounds"),
            "best_kl": info.get("best_kl"),
        })
    miners.sort(key=lambda m: m.get("cumulative_kl_diff") or 0, reverse=True)
    return JSONResponse(
        content=_sanitize_floats({"miners": miners}),
        headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"},
    )
