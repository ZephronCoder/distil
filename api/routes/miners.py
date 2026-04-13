"""Miner-related endpoints: scores, commitments, model info, miner details, compare."""

import json
import os
import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import STATE_DIR, CACHE_TTL
from ..helpers.cache import _get_cached, _get_stale, _set_cached, _bg_refresh
from ..helpers.fetch import _fetch_commitments
from ..helpers.sanitize import _sanitize_floats, _safe_json_load

router = APIRouter()


@router.get("/api/commitments", tags=["Miners"], summary="Miner model commitments",
         description="""Returns all miner HuggingFace model commitments (on-chain).

Each commitment contains:
- `model`: HuggingFace repo (e.g. `aceini/q-dist`)
- `revision`: Git commit SHA of the submitted model
- `block`: Block number when the commitment was made

**Cached for 60s.**
""")
def get_commitments():
    cached = _get_cached("commitments", CACHE_TTL)
    if cached:
        return cached
    stale = _get_stale("commitments")
    if stale:
        _bg_refresh("commitments", _fetch_commitments)
        return stale
    try:
        result = _fetch_commitments()
        _set_cached("commitments", result)
        return result
    except Exception as e:
        return {"commitments": {}, "count": 0, "error": str(e)}


@router.get("/api/scores", tags=["Miners"], summary="Current KL scores and disqualifications",
         description="""Returns the latest KL-divergence scores for all evaluated miners, plus disqualification status.

Response includes:
- `scores`: Map of UID → KL score (lower is better)
- `ema_scores`: Same as scores (backward compat)
- `disqualified`: Map of UID → disqualification reason
- `last_eval`: Details of the most recent evaluation round
- `last_eval_time`: Unix timestamp of last eval
- `tempo_seconds`: Seconds between evaluation rounds (currently 600)
""")
def get_scores(fields: str = ""):
    result = {"scores": {}, "ema_scores": {}, "disqualified": {}, "last_eval": None, "last_eval_time": None, "tempo_seconds": 600}
    scores_path = os.path.join(STATE_DIR, "scores.json")
    s = _safe_json_load(scores_path, {})
    result["scores"] = s
    result["ema_scores"] = s  # backward compat
    dq_path = os.path.join(STATE_DIR, "disqualified.json")
    result["disqualified"] = _safe_json_load(dq_path, {})
    eval_path = os.path.join(STATE_DIR, "last_eval.json")
    last_eval = _safe_json_load(eval_path)
    if last_eval is not None:
        result["last_eval"] = last_eval
        try:
            result["last_eval_time"] = os.path.getmtime(eval_path)
        except OSError:
            result["last_eval_time"] = last_eval.get("timestamp")
        result["last_eval_block"] = last_eval.get("block")
        result["last_eval_type"] = last_eval.get("type")
    # Filter fields if requested
    if fields:
        requested = set(f.strip() for f in fields.split(","))
        result = {k: v for k, v in result.items() if k in requested}
    return JSONResponse(
        content=_sanitize_floats(result),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/model-info/{model_path:path}", tags=["Miners"], summary="HuggingFace model info",
         description="""Fetches model card metadata from HuggingFace for a given repo.

**Example**: `/api/model-info/aceini/q-dist`

Response includes:
- `params_b`: Total parameters in billions
- `is_moe`: Whether the model uses Mixture of Experts
- `num_experts` / `num_active_experts`: MoE configuration
- `tags`, `license`, `pipeline_tag`: HuggingFace metadata
- `downloads`, `likes`: Popularity metrics
- `base_model`: Parent model (if distilled/fine-tuned)

**Cached for 1 hour.**
""")
def get_model_info(model_path: str):
    cache_key = f"model_info:{model_path}"
    cached = _get_cached(cache_key, 3600)
    if cached:
        return cached
    try:
        import subprocess
        script = """
import json, os, sys
from huggingface_hub import model_info as hf_model_info, hf_hub_download

model_path = os.environ["MODEL_PATH"]

info = hf_model_info(model_path, files_metadata=True)

params_b = None
if info.safetensors and hasattr(info.safetensors, "total"):
    params_b = round(info.safetensors.total / 1e9, 2)

active_params_b = None
is_moe = False
num_experts = None
num_active_experts = None
try:
    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    ne = config.get("num_local_experts", config.get("num_experts", 1))
    is_moe = ne > 1
    if is_moe:
        hidden = config.get("hidden_size", 0)
        num_experts = ne
        num_active_experts = config.get("num_experts_per_tok", config.get("num_active_experts", ne))
except Exception:
    pass

card = info.card_data
result = {
    "model": model_path,
    "author": info.author or model_path.split("/")[0],
    "tags": list(info.tags) if info.tags else [],
    "downloads": info.downloads,
    "likes": info.likes,
    "created_at": info.created_at.isoformat() if info.created_at else None,
    "last_modified": info.last_modified.isoformat() if info.last_modified else None,
    "params_b": params_b,
    "active_params_b": active_params_b,
    "is_moe": is_moe,
    "num_experts": num_experts,
    "num_active_experts": num_active_experts,
    "license": getattr(card, "license", None) if card else None,
    "pipeline_tag": info.pipeline_tag,
    "base_model": getattr(card, "base_model", None) if card else None,
}
print(json.dumps(result))
"""
        env = os.environ.copy()
        env["MODEL_PATH"] = model_path
        result_proc = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        if result_proc.returncode != 0:
            raise RuntimeError(result_proc.stderr[-300:])
        result = json.loads(result_proc.stdout)
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        return {"error": str(e), "model": model_path}


@router.get("/api/miner/{uid}", tags=["Miners"], summary="Full miner details by UID",
         description="""Returns everything known about a specific miner UID.

Response includes:
- `hotkey` / `coldkey`: On-chain keys
- `commitment`: Model repo, revision, and commitment block
- `kl_score`: Current KL-divergence score (lower = better)
- `disqualified`: Disqualification status and reason (if any)
- `h2h_history`: Last 10 head-to-head rounds involving this UID
- `in_top5`: Whether this UID is in the top 5 (king or contender)
- `is_king`: Whether this UID is the current king
- `registered`: Whether this UID is registered in the metagraph
""")
def get_miner(uid: int):
    result = {"uid": uid, "registered": False}

    # Metagraph data
    metagraph = _get_stale("metagraph") or {}
    neurons = metagraph.get("neurons", [])
    neuron = None
    for n in neurons:
        if n.get("uid") == uid:
            neuron = n
            break
    if neuron:
        result["registered"] = True
        result["hotkey"] = neuron.get("hotkey")
        result["coldkey"] = neuron.get("coldkey")
        result["stake"] = neuron.get("stake")
        result["incentive"] = neuron.get("incentive")
        result["emission"] = neuron.get("emission")
        result["is_validator"] = neuron.get("is_validator", False)
    else:
        result["hotkey"] = None
        result["coldkey"] = None

    # Commitment
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    hotkey = result.get("hotkey")
    # Fallback: if metagraph hotkey is stale/missing, try uid_hotkey_map.json
    # (maintained by the validator every epoch - always current)
    if not hotkey or hotkey not in commitments:
        uid_hk_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
        mapped_hk = uid_hk_map.get(str(uid))
        if mapped_hk and mapped_hk in commitments:
            hotkey = mapped_hk
            result["hotkey"] = hotkey  # update result with fresh hotkey
    if hotkey and hotkey in commitments:
        result["commitment"] = commitments[hotkey]
    else:
        result["commitment"] = None

    # KL score
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    uid_str = str(uid)
    result["kl_score"] = scores.get(uid_str)

    # Disqualification - check per-commit key first, fall back to legacy keys
    # only if no commit_block is known (same logic as eval/scoring.py is_disqualified)
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
    commit_block = result.get("commitment", {}).get("block") if result.get("commitment") else None
    dq_reason = None
    if commit_block is not None and hotkey:
        dq_reason = dq.get(f"{hotkey}:{commit_block}")
    if dq_reason is None and commit_block is None:
        # Only use legacy bare keys when we don't know the commit block
        dq_reason = dq.get(uid_str) or dq.get(hotkey) if hotkey else dq.get(uid_str)
    result["disqualified"] = dq_reason

    # Top 5 / king status
    top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {})
    king = top4.get("king") or {}
    contenders = top4.get("contenders") or []
    result["is_king"] = king.get("uid") == uid
    top5_uids = set()
    if king.get("uid") is not None:
        top5_uids.add(king["uid"])
    for c in contenders:
        if c.get("uid") is not None:
            top5_uids.add(c["uid"])
    result["in_top5"] = uid in top5_uids

    # Eval status: why (not) evaluated
    h2h_tracker = _safe_json_load(os.path.join(STATE_DIR, "h2h_tested_against_king.json"), {})
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    current_king_uid = h2h_latest.get("king_uid")
    current_block = h2h_latest.get("block", 0)
    tracker_entry = h2h_tracker.get(uid_str, {})
    eval_status = {}
    if result.get("disqualified"):
        eval_status["status"] = "disqualified"
        eval_status["reason"] = "Model is disqualified and won't be evaluated"
    elif result.get("is_king"):
        eval_status["status"] = "king"
        eval_status["reason"] = "Evaluated every round as the defending king"
    elif not result.get("kl_score"):
        eval_status["status"] = "queued"
        eval_status["reason"] = "Waiting for first evaluation - new submissions get priority"
    elif tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
        last_block = tracker_entry["block"]
        epochs_since = (current_block - last_block) // 360 if current_block > last_block else 0
        stale_threshold = 50
        if epochs_since < stale_threshold:
            eval_status["status"] = "tested"
            eval_status["reason"] = f"Already tested against current king ({epochs_since} epochs ago, re-test after {stale_threshold})"
            eval_status["last_test_block"] = last_block
            eval_status["epochs_since"] = epochs_since
            eval_status["stale_after"] = stale_threshold
        else:
            eval_status["status"] = "stale"
            eval_status["reason"] = f"Due for re-test ({epochs_since} epochs since last H2H, threshold is {stale_threshold})"
            eval_status["last_test_block"] = last_block
            eval_status["epochs_since"] = epochs_since
    else:
        eval_status["status"] = "untested"
        eval_status["reason"] = "Not yet tested against the current king - will be scheduled"
    result["eval_status"] = eval_status

    # H2H history (last 10 rounds involving this UID)
    h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    relevant = []
    for rnd in reversed(h2h_history):
        for r in rnd.get("results", []):
            if r.get("uid") == uid:
                relevant.append({
                    "block": rnd.get("block"),
                    "timestamp": rnd.get("timestamp"),
                    "kl": r.get("kl"),
                    "is_king": r.get("is_king", False),
                    "king_changed": rnd.get("king_changed", False),
                    "type": rnd.get("type"),
                })
                break
        if len(relevant) >= 10:
            break
    result["h2h_history"] = relevant

    return JSONResponse(
        content=result,
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/evaluated_uids", tags=["Miners"], summary="All evaluated UIDs with scores",
         description="""Returns all UIDs that have been evaluated, with their latest KL scores.

Response: `{uids: [{uid, kl_score, model_id?}], count: int}`
""")
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


@router.get("/api/dq_reasons", tags=["Miners"], summary="Disqualified UIDs with reasons",
         description="""Returns all disqualified entries with reasons.

Entries may be keyed by UID, hotkey, or hotkey:block. Response normalizes to a list.
""")
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
         description="""Returns model weight hashes (SHA256 of safetensor metadata) for all tracked UIDs.

Used for transparency - anyone can verify a miner's model hasn't changed since evaluation.
""")
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
         description="""Returns all head-to-head rounds where a specific UID participated.

Supports `?limit=N` (default 50, max 200) and `?page=N` (1-indexed). Newest rounds first.

Each round entry includes:
- `block`: Block number of the round
- `timestamp`: Unix timestamp
- `kl`: This miner's KL score in that round
- `is_king`: Whether the miner was king during this round
- `king_changed`: Whether the king was dethroned
- `type`: Round type (h2h or full_eval)
- `king_uid`: Who was king that round
- `n_prompts`: Number of prompts used
""")
def get_miner_rounds(uid: int, limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    try:
        h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
        if not isinstance(h2h_history, list):
            h2h_history = []

        # Filter rounds where this UID participated
        relevant = []
        for rnd in reversed(h2h_history):
            for r in rnd.get("results", []):
                if r.get("uid") == uid:
                    relevant.append({
                        "block": rnd.get("block"),
                        "timestamp": rnd.get("timestamp"),
                        "kl": r.get("kl"),
                        "model": r.get("model"),
                        "is_king": r.get("is_king", False),
                        "vs_king": r.get("vs_king"),
                        "king_changed": rnd.get("king_changed", False),
                        "king_uid": rnd.get("king_uid"),
                        "new_king_uid": rnd.get("new_king_uid"),
                        "type": rnd.get("type"),
                        "n_prompts": rnd.get("n_prompts"),
                        "p_value": rnd.get("p_value"),
                    })
                    break

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
         description="""Lookup a miner's on-chain model commitment by their hotkey (ss58 address).

Useful for miners to verify the validator sees their commitment after submitting.

Response includes:
- `commitment`: Model repo, revision, and commitment block (if found)
- `uid`: Registered UID (if registered in metagraph)
- `registered`: Whether this hotkey is registered
""")
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
         description="""Compare KL scores and H2H history for multiple UIDs side by side.

Usage: `/api/compare?uids=2,34,36,218`

Returns for each UID:
- Current KL score
- Model name
- Number of H2H rounds participated
- Best KL ever achieved
- Win/loss record vs king
""")
def compare_miners(uids: str):
    uid_list = [int(u.strip()) for u in uids.split(",") if u.strip().isdigit()][:10]
    if not uid_list:
        return JSONResponse(status_code=400, content={"error": "Provide ?uids=1,2,3"})

    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})

    result = []
    for uid in uid_list:
        entry = {"uid": uid, "kl_score": scores.get(str(uid))}

        # Model name
        hotkey = uid_map.get(str(uid))
        if hotkey and hotkey in commitments:
            c = commitments[hotkey]
            entry["model"] = c.get("model") or c.get("repo")
        else:
            entry["model"] = None

        # Is king?
        entry["is_king"] = h2h_latest.get("king_uid") == uid

        # DQ status
        entry["disqualified"] = str(uid) in dq or (hotkey and hotkey in dq)

        # H2H stats
        rounds_participated = 0
        best_kl = None
        wins_vs_king = 0
        losses_vs_king = 0
        for rnd in h2h_history:
            for r in rnd.get("results", []):
                if r.get("uid") == uid:
                    rounds_participated += 1
                    kl = r.get("kl")
                    if kl is not None and (best_kl is None or kl < best_kl):
                        best_kl = kl
                    if r.get("is_king"):
                        if rnd.get("king_changed") and rnd.get("new_king_uid") != uid:
                            losses_vs_king += 1
                        else:
                            wins_vs_king += 1
                    else:
                        if rnd.get("king_changed") and rnd.get("new_king_uid") == uid:
                            wins_vs_king += 1
                    break

        entry["rounds_participated"] = rounds_participated
        entry["best_kl"] = best_kl
        entry["wins"] = wins_vs_king
        entry["losses"] = losses_vs_king
        result.append(entry)

    result.sort(key=lambda x: x.get("kl_score") or 999)
    return JSONResponse(
        content=_sanitize_floats({"miners": result}),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )
