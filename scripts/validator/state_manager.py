"""
State file management: H2H state, scores, commitments, leaderboard updates.
"""
import json
import logging
import os
import time

from eval.state import ValidatorState, atomic_json_write
from eval.scoring import (
    is_disqualified, get_dq_reason, disqualify,
    append_score_history,
)
from scripts.validator.config import (
    MAX_KL_THRESHOLD, EPSILON, PAIRED_TEST_ALPHA, TOP_N_ALWAYS_INCLUDE,
)

logger = logging.getLogger("distillation.remote_validator")


# Minimum prompts the king must complete for a round's results to be
# trusted for persistent state updates (top-4 leaderboard, state.scores,
# score_history, h2h_tested_against_king). Partial rounds are written to
# h2h_history with ``_invalid`` but cannot overwrite the leaderboard or
# scores. Stricter than the dethronement gate in results.py because the
# full-round leaderboard needs a higher bar than single-round
# dethronement.
MIN_PROMPTS_FOR_LEADERBOARD = 150


def migrate_dq_entries(state: ValidatorState, commitments: dict):
    """Migrate bare-hotkey and stale bare-UID DQ entries to per-commit format."""
    hotkey_to_block = {
        com["hotkey"]: com["block"]
        for com in commitments.values()
        if "hotkey" in com and "block" in com
    }

    # Migrate bare hotkey → hotkey:block
    migrated = 0
    for key in list(state.dq_reasons.keys()):
        if key.startswith("flag:") or key.isdigit() or ":" in key:
            continue
        if key in hotkey_to_block:
            new_key = f"{key}:{hotkey_to_block[key]}"
            state.dq_reasons[new_key] = state.dq_reasons.pop(key)
            migrated += 1
    if migrated:
        logger.info(f"Migrated {migrated} DQ entries to per-commit format")

    # Scrub stale bare-UID entries
    scrubbed = 0
    for key in list(state.dq_reasons.keys()):
        if not key.isdigit():
            continue
        uid = int(key)
        if uid not in commitments:
            continue
        com = commitments[uid]
        hk = com.get("hotkey", "")
        blk = com.get("block")
        if blk and f"{hk}:{blk}" in state.dq_reasons:
            del state.dq_reasons[key]
            scrubbed += 1
            continue
        current_model = com.get("model", "")
        dq_reason = state.dq_reasons[key]
        if current_model and current_model not in dq_reason:
            logger.info(f"Removing stale bare-UID DQ: UID {uid}")
            del state.dq_reasons[key]
            scrubbed += 1
    if scrubbed:
        logger.info(f"Scrubbed {scrubbed} stale bare-UID DQ entries")

    # Scrub stale hotkey:block entries where the model was re-committed
    recommit_scrubbed = 0
    for key in list(state.dq_reasons.keys()):
        if ":" not in key or key.startswith("flag:"):
            continue
        parts = key.split(":", 1)
        if len(parts) != 2:
            continue
        hk, blk_str = parts
        try:
            dq_block = int(blk_str)
        except ValueError:
            continue
        current_block = hotkey_to_block.get(hk)
        if current_block is not None and current_block != dq_block:
            logger.info(f"Removing stale DQ for re-committed hotkey {hk[:16]}... "
                        f"(DQ block {dq_block} → current block {current_block})")
            del state.dq_reasons[key]
            recommit_scrubbed += 1
    if recommit_scrubbed:
        logger.info(f"Scrubbed {recommit_scrubbed} stale hotkey:block DQ entries (model re-committed)")


def update_h2h_state(state: ValidatorState, h2h_results, king_uid, winner_uid,
                     king_h2h_kl, king_kl, king_per_prompt, current_block,
                     n_prompts, is_full_eval, uid_to_model, valid_models,
                     challengers, epoch_count, disqualified, block_hash=None,
                     epoch_start_time=None):
    """Update H2H state files: latest, history, tested-against-king."""

    n_challenger_results = sum(1 for r in h2h_results if not r.get("is_king"))
    # Cold-start crowning (king_uid=None) is also a king change so the
    # post-round path announces the first crown of a new era.
    if king_uid is None:
        king_changed = winner_uid is not None
    else:
        king_changed = winner_uid != king_uid

    if n_challenger_results == 0 and not king_changed:
        logger.info("All challengers failed and king unchanged — skipping H2H round save")
        return
    effective_king_uid = winner_uid if winner_uid is not None else king_uid
    effective_king_kl = king_h2h_kl
    effective_king_model = uid_to_model.get(effective_king_uid, valid_models.get(effective_king_uid, {}).get("model", ""))
    if king_changed and winner_uid is not None:
        # Try to get KL from h2h_results first, then state.scores
        found_kl = False
        for r in h2h_results:
            if r["uid"] == winner_uid:
                effective_king_kl = r.get("kl", king_h2h_kl)
                found_kl = True
                break
        if not found_kl:
            # Winner not in h2h_results (e.g. king-failed promotion).
            winner_kl_from_scores = state.scores.get(str(winner_uid))
            if winner_kl_from_scores and winner_kl_from_scores > 0:
                effective_king_kl = winner_kl_from_scores
                logger.info(f"Using global score {effective_king_kl:.6f} for new king UID {winner_uid} (not in h2h_results)")

    _king_h2h_kl = round(effective_king_kl, 6) if effective_king_kl else None

    shard_idx: int | None = None
    try:
        from eval.dataset import CLIMBMIX_NUM_SHARDS, _compute_hash_hex
        _hex = _compute_hash_hex(current_block, block_hash)
        shard_idx = int(_hex[:8], 16) % CLIMBMIX_NUM_SHARDS
    except Exception:
        shard_idx = None

    dq_blocked = []
    if not king_changed:
        for r in h2h_results:
            if r.get("is_king") or r.get("is_reference"):
                continue
            tt = r.get("t_test") or {}
            beat_king = (tt.get("mean_delta", 0) > 0 and tt.get("p", 1.0) < PAIRED_TEST_ALPHA)
            if beat_king and r.get("disqualified"):
                dq_blocked.append({"uid": r.get("uid"), "model": r.get("model"),
                                   "dq_reason": r.get("dq_reason"), "kl": r.get("kl"),
                                   "p": tt.get("p"), "mean_delta": tt.get("mean_delta")})
    king_retained_reason = None
    if not king_changed and dq_blocked:
        king_retained_reason = (
            f"{len(dq_blocked)} lower-KL challenger(s) would have dethroned but were DQ'd "
            f"(e.g. UID {dq_blocked[0]['uid']}: {(dq_blocked[0]['dq_reason'] or '')[:80]})"
        )
        for entry in dq_blocked:
            logger.info(
                f"King retained: UID {entry['uid']} had KL={entry['kl']:.6f} "
                f"(p={entry['p']:.4f}, mean_delta={entry['mean_delta']:.6f}) "
                f"but DQ'd — {entry['dq_reason']}"
            )

    # Round canonicality: king must complete enough paired prompts.
    king_prompts_completed = len(king_per_prompt) if king_per_prompt else 0
    # SINGLE_EVAL_MODE: the previous king is intentionally absent (the
    # cross-round composite selector picks next king); skip the paired
    # threshold or every single-eval round would be flagged PARTIAL.
    try:
        from scripts.validator.single_eval import is_single_eval_mode as _is_single
        single_eval_active = bool(_is_single())
    except Exception:
        single_eval_active = bool(int(os.environ.get("SINGLE_EVAL_MODE", "0") or 0))
    # single-eval round with no king prompts -> canonical if any challenger
    # results came back.
    single_eval_kingless = single_eval_active and king_prompts_completed == 0
    if single_eval_kingless:
        round_is_canonical = bool(h2h_results)
    else:
        round_is_canonical = (
            king_prompts_completed >= MIN_PROMPTS_FOR_LEADERBOARD
            and not (king_uid is not None and king_h2h_kl is None)
        )
    if not round_is_canonical:
        logger.warning(
            f"🚧 Round at block {current_block} is PARTIAL "
            f"(king completed {king_prompts_completed}/{n_prompts} prompts, "
            f"threshold={MIN_PROMPTS_FOR_LEADERBOARD}). h2h_history entry will "
            f"be marked `_invalid_for_leaderboard=True`; skipping "
            f"top4_leaderboard/scores/h2h_tested_against_king writes."
        )
    elif single_eval_kingless:
        logger.info(
            f"single-eval round at block {current_block}: king deliberately "
            f"absent (cross-round composite selection); treating round as "
            f"canonical with {len(h2h_results)} challenger results."
        )

    h2h_round = {
        "block": current_block, "block_hash": block_hash, "timestamp": time.time(),
        "shard_idx": shard_idx,
        "king_uid": effective_king_uid, "king_model": effective_king_model,
        "prev_king_uid": king_uid,
        "king_kl": _king_h2h_kl,  # canonical field for API consumers
        "king_h2h_kl": _king_h2h_kl,
        "king_global_kl": round(king_kl, 6),
        "epsilon": EPSILON,
        "epsilon_threshold": round(king_h2h_kl * (1.0 - EPSILON), 6) if king_h2h_kl else None,
        "paired_test_alpha": PAIRED_TEST_ALPHA,
        "dethrone_method": "paired_t_test" if king_per_prompt else "legacy_epsilon",
        "n_prompts": n_prompts, "results": h2h_results,
        "king_changed": king_changed,
        "new_king_uid": winner_uid if king_changed else None,
        "king_retained_reason": king_retained_reason,
        "dq_blocked_dethrone": dq_blocked or None,
        "type": "full_eval" if is_full_eval else "h2h",
        "elapsed_seconds": round(time.time() - epoch_start_time, 1) if epoch_start_time else None,
        "n_students": len(h2h_results),
        "king_prompts_completed": king_prompts_completed,
        "_invalid_for_leaderboard": not round_is_canonical,
        "_min_prompts_for_leaderboard": MIN_PROMPTS_FOR_LEADERBOARD,
    }

    # Overwrite canonical h2h_latest only on trustworthy rounds; partial
    # rounds still get appended to h2h_history with the invalid flag.
    if round_is_canonical:
        state.h2h_latest = h2h_round
    state.h2h_history = [h for h in state.h2h_history if not (h.get("block") == current_block and h.get("_preliminary"))]
    state.h2h_history.append(h2h_round)
    state.h2h_history = state.h2h_history[-50:]
    state.save_h2h()

    # Tested-against-king tracker only on canonical rounds.
    if round_is_canonical and king_uid is not None:
        for uid in challengers:
            uid_str = str(uid)
            if uid_str in state.scores and state.scores[uid_str] > 0:
                state.h2h_tested_against_king[uid_str] = {
                    "king_uid": king_uid, "epoch": epoch_count,
                    "block": current_block, "kl": round(state.scores[uid_str], 6),
                    "model": challengers[uid].get("model", ""), "timestamp": time.time(),
                }
        atomic_json_write(state._path("h2h_tested_against_king.json"),
                          state.h2h_tested_against_king, indent=2)

    # King regression streak: count consecutive "at risk" rounds (king.worst
    # below floor or below base.worst). Telemetry-only until KING_REGRESSION_
    # GATE=1; canonical rounds only.
    if round_is_canonical and effective_king_uid is not None:
        try:
            king_row = next(
                (r for r in h2h_results if r.get("is_king")), None
            )
            health = ((king_row or {}).get("composite") or {}).get("king_health")
            if health:
                king_key = str(effective_king_uid)
                at_risk = bool(health.get("at_risk"))
                prev_streak = int(state.king_regression_streak.get(king_key, 0))
                # Drop stale streaks for previous kings (keep only current
                # king + the immediate predecessor for debugging).
                state.king_regression_streak = {
                    k: v for k, v in state.king_regression_streak.items()
                    if k in (king_key, str(king_uid) if king_uid is not None else None)
                }
                if king_changed:
                    # When crowning a king that is ALREADY at-risk (cold-start
                    # cohort score=0 case), seed the streak at MIN_STREAK-1 so
                    # the next at_risk increment immediately reaches the floor
                    # waiver. Healthy fresh kings still start at 0.
                    if at_risk:
                        try:
                            from scripts.validator.composite import (
                                KING_REGRESSION_MIN_STREAK as _MIN_STREAK,
                            )
                            prev_streak = max(0, int(_MIN_STREAK) - 1)
                        except Exception:
                            prev_streak = 2
                    else:
                        prev_streak = 0
                new_streak = prev_streak + 1 if at_risk else 0
                state.king_regression_streak[king_key] = new_streak
                if at_risk:
                    logger.warning(
                        f"🚨 King UID {effective_king_uid} at risk: "
                        f"worst={health.get('king_worst'):.3f} "
                        f"axis={health.get('king_worst_axis')} "
                        f"(below_floor={health.get('below_floor')}, "
                        f"worse_than_base={health.get('worse_than_base')}) — "
                        f"streak={new_streak}/{health.get('min_streak')}"
                    )
                else:
                    logger.info(
                        f"King UID {effective_king_uid} healthy: "
                        f"worst={health.get('king_worst'):.3f} "
                        f"(floor={health.get('floor')}) — streak reset."
                    )
                # Persist the streak (save_h2h ran above; this second write is cheap).
                atomic_json_write(
                    state._path("king_regression_streak.json"),
                    state.king_regression_streak,
                    indent=2,
                )
        except Exception as exc:
            logger.warning(f"king_regression_streak update failed (non-fatal): {exc}")

    # King canary streak: held-out evalscope regression tracker (sibling of
    # king_regression_streak above). Either streak >= its min waives the
    # composite-floor veto when both gates are active.
    if round_is_canonical and effective_king_uid is not None:
        try:
            from scripts.validator.composite import (
                _compute_king_canary_regression,
                KING_CANARY_GATE,
                KING_CANARY_MIN_STREAK,
            )
            if KING_CANARY_GATE:
                canary = _compute_king_canary_regression(
                    effective_king_uid, state.state_dir,
                )
                if canary is not None:
                    if not hasattr(state, "king_canary_streak") or not isinstance(getattr(state, "king_canary_streak", None), dict):
                        state.king_canary_streak = {}
                    king_key = str(effective_king_uid)
                    at_risk = bool(canary.get("at_risk"))
                    prev_streak = int(state.king_canary_streak.get(king_key, 0))
                    state.king_canary_streak = {
                        k: v for k, v in state.king_canary_streak.items()
                        if k in (king_key, str(king_uid) if king_uid is not None else None)
                    }
                    if king_changed:
                        prev_streak = 0  # fresh king, fresh slate
                    new_streak = prev_streak + 1 if at_risk else 0
                    state.king_canary_streak[king_key] = new_streak
                    if at_risk:
                        logger.warning(
                            f"🚨 King UID {effective_king_uid} CANARY at risk: "
                            f"held-out mean={canary.get('king_canary_mean')} "
                            f"vs base={canary.get('base_canary_mean')} "
                            f"(gap={canary.get('gap_pp')} > margin={canary.get('margin')}) — "
                            f"axes={canary.get('axes_compared')} "
                            f"streak={new_streak}/{KING_CANARY_MIN_STREAK}"
                        )
                    else:
                        logger.info(
                            f"King UID {effective_king_uid} canary healthy: "
                            f"held-out mean={canary.get('king_canary_mean')} "
                            f"vs base={canary.get('base_canary_mean')} — streak reset."
                        )
                    atomic_json_write(
                        state._path("king_canary_streak.json"),
                        state.king_canary_streak,
                        indent=2,
                    )
        except Exception as exc:
            logger.warning(f"king_canary_streak update failed (non-fatal): {exc}")


_COPY_LIKE_DQ_PATTERNS = (
    "activation-space duplicate",
    "identical weights",
    "copy: activation",
    "copy: identical",
)


def _is_copy_like_dq(reason: str) -> bool:
    """True iff ``reason`` reads like a copy-of-another-model DQ (same weights
    or near-identical activations), rather than a quality/integrity failure."""
    if not isinstance(reason, str):
        return False
    lowered = reason.lower()
    return any(p in lowered for p in _COPY_LIKE_DQ_PATTERNS)


def _get_dq_reason_for_uid(uid, info: dict, dq_reasons: dict) -> str:
    """Resolve the DQ reason (if any) for a UID + commit, tolerating missing
    hotkey / commit_block.
    """
    hotkey = (info or {}).get("hotkey", "") or ""
    cb = (info or {}).get("commit_block")
    try:
        return get_dq_reason(uid, hotkey, dq_reasons, commit_block=cb)
    except Exception:
        return ""


def update_model_tracking(state: ValidatorState, models_to_eval, current_block,
                          king_kl, disqualified):
    """Update persistent model score history and permanently bad models."""
    for uid, info in models_to_eval.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.scores and state.scores[uid_str] > 0:
            kl = state.scores[uid_str]
            prev = state.model_score_history.get(model_name, {})
            # Collapse explicit-None best_kl/worst_kl to inf/0 sentinels so
            # ``kl < None`` comparisons stay total (a past rollback persisted
            # nulls).
            if kl <= MAX_KL_THRESHOLD:
                prev_best_raw = prev.get("best_kl")
                prev_best = float("inf") if prev_best_raw is None else prev_best_raw
                if kl < prev_best:
                    state.model_score_history[model_name] = {
                        **prev, "best_kl": round(kl, 6), "uid": uid,
                        "block": current_block, "timestamp": time.time(),
                    }
            else:
                prev_worst_raw = prev.get("worst_kl")
                prev_worst = 0 if prev_worst_raw is None else prev_worst_raw
                if kl > prev_worst:
                    state.model_score_history[model_name] = {
                        **prev, "worst_kl": round(kl, 6), "uid": uid,
                        "block": current_block, "timestamp": time.time(),
                    }
                existing = state.model_score_history.get(model_name, {})
                if existing.get("best_kl") is None:
                    state.model_score_history.setdefault(model_name, {})["best_kl"] = round(kl, 6)

    if king_kl > 0 and king_kl < float("inf"):
        perm_bad_threshold = king_kl * 10.0
        newly_banned = []
        for uid, info in models_to_eval.items():
            uid_str = str(uid)
            if uid_str in state.scores and state.scores[uid_str] > perm_bad_threshold:
                model_name = info["model"]
                if model_name in state.permanently_bad_models:
                    continue
                dq_reason = _get_dq_reason_for_uid(uid, info, state.dq_reasons)
                if dq_reason and _is_copy_like_dq(dq_reason):
                    # Copy-like DQs set score=3.0 as a penalty but the model
                    # itself may be valid; don't perm-ban it.
                    logger.info(
                        f"  skipping perm-ban of {model_name} (UID {uid}) — "
                        f"DQ is copy-like: '{dq_reason[:80]}...'"
                    )
                    continue
                state.permanently_bad_models.add(model_name)
                newly_banned.append(f"{model_name} (UID {uid}, KL={state.scores[uid_str]:.4f})")
        if newly_banned:
            logger.info(f"🚫 Added {len(newly_banned)} models to permanently_bad_models")

    state.save_model_tracking()


def update_top4_leaderboard(state: ValidatorState, winner_uid, king_uid, king_kl,
                            h2h_results, uid_to_model, valid_models, current_block,
                            epoch_count, disqualified):
    """Update the top-4 leaderboard (initial eval -> maintenance transition).

    Skips the overwrite when ``state.h2h_latest._invalid_for_leaderboard`` is
    set (partial-prompt KLs would silently clobber the leaderboard scale)."""
    latest = getattr(state, "h2h_latest", None) or {}
    if latest.get("_invalid_for_leaderboard"):
        logger.warning(
            f"🚧 Skipping top4_leaderboard update for block {current_block} "
            f"— round flagged _invalid_for_leaderboard "
            f"(king_prompts_completed={latest.get('king_prompts_completed')})."
        )
        return

    try:
        if state.top4_leaderboard.get("phase") == "initial_eval":
            # Auto-promote to maintenance as soon as a canonical round
            # produces a real winner; the legacy "wait for 4 tested + zero
            # untested" gate never fires post-teacher-cutover.
            if winner_uid is not None:
                winner_kl_for_lb = next(
                    (r.get("kl") for r in h2h_results if r.get("uid") == winner_uid),
                    state.scores.get(str(winner_uid)),
                )
                winner_model = uid_to_model.get(
                    winner_uid,
                    valid_models.get(winner_uid, {}).get("model", "unknown"),
                )
                state.top4_leaderboard["king"] = {
                    "uid": int(winner_uid), "model": winner_model,
                    "h2h_kl": round(winner_kl_for_lb, 6) if isinstance(winner_kl_for_lb, float) else winner_kl_for_lb,
                    "block": current_block,
                }
                contender_cap_cs = max(1, TOP_N_ALWAYS_INCLUDE - 1)
                cs_contenders = []
                for r in sorted(h2h_results, key=lambda r: r.get("kl", float("inf"))):
                    if r.get("uid") == winner_uid:
                        continue
                    if int(r.get("uid", 0)) < 0:
                        continue
                    if int(r.get("uid", 0)) in disqualified:
                        continue
                    cs_contenders.append({
                        "uid": r.get("uid"), "model": r.get("model"),
                        "h2h_kl": round(r["kl"], 6) if isinstance(r.get("kl"), float) else r.get("kl"),
                        "block": current_block,
                    })
                    if len(cs_contenders) >= contender_cap_cs:
                        break
                state.top4_leaderboard["contenders"] = cs_contenders
                state.top4_leaderboard["phase"] = "maintenance"
                state.top4_leaderboard["initial_eval_complete"] = True
                state.top4_leaderboard["completed_at"] = time.time()
                state.top4_leaderboard["completed_block"] = current_block
                state.save_top4()
                logger.info(
                    f"👑 TOP-4 PROMOTED to maintenance via cold-start "
                    f"crowning: UID {winner_uid} ({winner_model})"
                )
                return
            untested_count = 0
            tested_results = []
            for uid_str, score in state.scores.items():
                if score <= 0 or score > MAX_KL_THRESHOLD:
                    continue
                if int(uid_str) in disqualified:
                    continue
                record = state.h2h_tested_against_king.get(uid_str, {})
                if record.get("king_uid") == king_uid and record.get("kl"):
                    tested_results.append((uid_str, record["kl"], record.get("model", "")))
                else:
                    untested_count += 1

            if untested_count == 0 and len(tested_results) >= 4:
                tested_results.sort(key=lambda x: x[1])
                state.top4_leaderboard["king"] = {
                    "uid": int(tested_results[0][0]), "model": tested_results[0][2],
                    "h2h_kl": round(tested_results[0][1], 6), "block": current_block,
                }
                # Up to TOP_N_ALWAYS_INCLUDE-1 contenders (slot 0 is the king).
                contender_cap = max(1, TOP_N_ALWAYS_INCLUDE - 1)
                state.top4_leaderboard["contenders"] = [
                    {"uid": int(tested_results[i][0]), "model": tested_results[i][2],
                     "h2h_kl": round(tested_results[i][1], 6), "block": current_block}
                    for i in range(1, min(contender_cap + 1, len(tested_results)))
                ]
                state.top4_leaderboard["phase"] = "maintenance"
                state.top4_leaderboard["initial_eval_complete"] = True
                state.top4_leaderboard["completed_at"] = time.time()
                state.top4_leaderboard["completed_block"] = current_block
                logger.info(f"👑 TOP-4 INITIAL EVAL COMPLETE")
            else:
                logger.info(f"📊 Initial eval: {len(tested_results)} tested, {untested_count} remaining")

        elif state.top4_leaderboard.get("phase") == "maintenance":
            actual_king = winner_uid if winner_uid is not None else king_uid
            king_model = uid_to_model.get(actual_king, valid_models.get(actual_king, {}).get("model", "unknown"))
            king_kl_lb = next((r["kl"] for r in h2h_results if r["uid"] == actual_king), state.scores.get(str(actual_king), 999))

            state.top4_leaderboard["king"] = {
                "uid": actual_king, "model": king_model,
                "h2h_kl": round(king_kl_lb, 6) if isinstance(king_kl_lb, float) else king_kl_lb,
                "block": current_block,
            }
            contender_cap = max(1, TOP_N_ALWAYS_INCLUDE - 1)
            contenders = []
            for r in sorted(h2h_results, key=lambda r: r.get("kl", float("inf"))):
                if r["uid"] == actual_king:
                    continue
                if int(r["uid"]) < 0:
                    continue  # skip reference model
                if int(r["uid"]) in disqualified:
                    continue
                contenders.append({
                    "uid": r["uid"], "model": r["model"],
                    "h2h_kl": round(r["kl"], 6), "block": current_block,
                })
                if len(contenders) >= contender_cap:
                    break
            state.top4_leaderboard["contenders"] = contenders

        state.save_top4()
        top4_str = ", ".join(
            f"#{i+1} UID {e['uid']} (KL={e['h2h_kl']})"
            for i, e in enumerate([state.top4_leaderboard.get('king', {})] + (state.top4_leaderboard.get('contenders') or []))
            if e and e.get('uid') is not None
        )
        if top4_str:
            logger.info(f"📊 TOP-4: {top4_str}")
    except Exception as e:
        logger.warning(f"Top-4 leaderboard error (non-fatal): {e}")
