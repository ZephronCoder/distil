import logging
import json
import os
import time

from eval.scoring import disqualify
from eval.state import ValidatorState
from scripts.eval_policy import policy_env
from scripts.validator.config import MAX_KL_THRESHOLD, TOP_N_ALWAYS_INCLUDE
from scripts.validator import single_eval as single_eval_mod
from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
from scripts.validator.single_eval import (
    bootstrap_composite_from_h2h,
    evict_stale_evaluated_uids,
    is_single_eval_mode,
)

logger = logging.getLogger("distillation.remote_validator")


def _write_eval_backlog(state: ValidatorState, *, cap: int, pending: dict, kept: dict, deferred: list[int]) -> None:
    try:
        state_dir = getattr(state, "state_dir", None)
        if state_dir is None:
            return
        kept_set = {int(uid) for uid in kept.keys()}
        pending_rows = []
        for uid, info in sorted(
            pending.items(),
            key=lambda kv: (int((kv[1] or {}).get("commit_block") or 0), int(kv[0])),
        ):
            pending_rows.append({
                "uid": int(uid),
                "model": (info or {}).get("model"),
                "revision": (info or {}).get("revision"),
                "commit_block": (info or {}).get("commit_block"),
                "status": "queued" if int(uid) in kept_set else "deferred",
            })
        payload = {
            "updated_at": time.time(),
            "round_cap": cap,
            "pending_total": len(pending),
            "queued_uids": [int(uid) for uid in kept.keys()],
            "deferred_uids": [int(uid) for uid in deferred],
            "pending": pending_rows,
        }
        path = state_dir / "eval_backlog.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("single-eval: failed to persist eval backlog: %s", exc)


# Rotate N dormant miners per round whose global KL beats the king's,
# so dormant high-scorers can re-challenge without re-uploading.
DORMANT_ROTATION_N = int(policy_env("DORMANT_ROTATION_N", "2"))

# Maintenance rounds cap challenger count; top H2H contenders stay sticky.
MAINTENANCE_CHALLENGER_CAP = int(policy_env("MAINTENANCE_CHALLENGER_CAP", "12"))
PROTECTED_H2H_CONTENDERS = int(
    policy_env("PROTECTED_H2H_CONTENDERS", str(min(4, max(1, TOP_N_ALWAYS_INCLUDE - 1))))
)


# Evict H2H leaderboard contenders after N consecutive precheck failures
# (e.g. miner privates a public repo). Counter resets on first pass.
LB_PRECHECK_EVICTION_STREAK = int(policy_env("LB_PRECHECK_EVICTION_STREAK", "3"))


def select_challengers(valid_models, state: ValidatorState, king_uid, king_kl,
                       epoch_count: int, trust_king_kl: bool = True):
    """Pick challengers for the round.

    ``trust_king_kl`` = False disables the ``best_ever > king_kl*2`` prune
    when ``king_kl`` came from a stale cached score.

    ``SINGLE_EVAL_MODE=1`` returns only commitments not yet scored (or
    with a changed on-chain commit); re-evaluation is disallowed.
    """
    if is_single_eval_mode():
        evict_stale_evaluated_uids(state, valid_models)
        challengers = {}
        # Force-eligible: current king (paired re-eval on shared prompts,
        # fixes cross-sample variance) + schema-bump fallback.
        force_eligible: set[str] = set()
        if king_uid is not None:
            king_record = (state.composite_scores or {}).get(str(king_uid))
            force_eligible.add(str(king_uid))
            if isinstance(king_record, dict):
                try:
                    king_version = int(king_record.get("version") or 0)
                except (TypeError, ValueError):
                    king_version = 0
                if king_version < int(COMPOSITE_SHADOW_VERSION):
                    logger.info(
                        f"single-eval: forcing king UID {king_uid} re-eval "
                        f"(stored composite version {king_version} < "
                        f"current schema {COMPOSITE_SHADOW_VERSION}); ensures "
                        f"like-for-like comparison against challengers."
                    )
                else:
                    logger.info(
                        f"single-eval: king UID {king_uid} included in "
                        f"this round (paired re-eval on shared prompts)."
                    )
        for uid, info in valid_models.items():
            uid_str = str(uid)
            model_name = info["model"]
            if info.get("is_reference"):
                continue
            if model_name in state.permanently_bad_models:
                state.evaluated_uids.add(uid_str)
                continue
            if uid_str in state.composite_scores and uid_str not in force_eligible:
                continue
            # Strict no-re-eval: any UID already in evaluated_uids has had
            # its one shot (commitment-changed entries are evicted above).
            if uid_str in state.evaluated_uids and uid_str not in force_eligible:
                continue
            challengers[uid] = info
        # FIFO cap (oldest commit first) keeps each round inside the
        # 60-75 min target. Read live from single_eval for runtime overrides.
        cap = int(single_eval_mod.SINGLE_EVAL_MAX_PER_ROUND)
        pending_before_cap = dict(challengers)
        deferred: list[int] = []
        if challengers and cap > 0 and len(challengers) > cap:
            ordered = sorted(
                challengers.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("commit_block") or 0),
                    kv[0],
                ),
            )
            kept = dict(ordered[:cap])
            deferred = [uid for uid, _ in ordered[cap:]]
            logger.info(
                f"single-eval: capping round at {cap} of {len(challengers)} "
                f"pending new commitments (FIFO by commit_block); deferred "
                f"to next round: {deferred}"
            )
            challengers = kept
        _write_eval_backlog(
            state,
            cap=cap,
            pending=pending_before_cap,
            kept=challengers,
            deferred=deferred,
        )
        if challengers:
            n_king = 1 if (king_uid is not None and str(king_uid) in {str(u) for u in challengers}) else 0
            n_others = len(challengers) - n_king
            logger.info(
                f"single-eval: {n_others} new commitment(s) to evaluate"
                + (" + king (paired re-eval)" if n_king else "")
                + " (no top-N rotation, no dormant rotation)"
            )
        else:
            logger.info(
                "single-eval: no new commitments this round — round will be a no-op "
                "(king retains crown, weights stay)"
            )
        return challengers
    challengers = {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.evaluated_uids and uid_str in state.scores:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        best_ever = state.model_score_history.get(model_name, {}).get("best_kl")
        if trust_king_kl and best_ever is not None and king_kl < float("inf"):
            skip_threshold = max(king_kl * 2.0, king_kl + 0.05)
            if best_ever > skip_threshold:
                state.evaluated_uids.add(uid_str)
                continue
        challengers[uid] = info
    if king_uid is None:
        return challengers
    p1_new = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info["model"] in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        if state.scores.get(uid_str) is not None:
            continue
        if uid_str in state.evaluated_uids:
            continue
        p1_new.append(uid)
    for uid in p1_new:
        challengers[uid] = valid_models[uid]
    if p1_new:
        logger.info(f"🎯 SMART CHALLENGER: {len(p1_new)} new submission(s) — Priority 1: never evaluated")
    if state.top4_leaderboard.get("phase") == "initial_eval":
        full_eval_kl_cutoff = 0.12
        p1b = []
        for uid, info in valid_models.items():
            if uid == king_uid or uid in challengers:
                continue
            if info["model"] in state.permanently_bad_models:
                continue
            uid_str = str(uid)
            global_kl = state.scores.get(uid_str)
            if global_kl is None or global_kl <= 0 or global_kl > full_eval_kl_cutoff:
                continue
            h2h_record = state.h2h_tested_against_king.get(uid_str, {})
            if h2h_record.get("king_uid") == king_uid:
                continue
            p1b.append((uid, global_kl))
        if p1b:
            p1b.sort(key=lambda x: x[1])
            for uid, _ in p1b:
                challengers[uid] = valid_models[uid]
            logger.info(f"🏆 FULL EVAL: {len(p1b)} scored models added (untested vs new king, KL<=0.12)")
    return challengers


def add_top5_contenders(challengers, valid_models, state: ValidatorState, king_uid):
    """Always include top H2H contenders in every eval round.

    Prefers ``top4_leaderboard.contenders`` (same-prompt-set ranking);
    falls back to ``state.scores`` only when no H2H leaderboard exists.

    No-op in single-eval mode (one-eval-per-commitment).
    """
    if is_single_eval_mode():
        return
    if king_uid is None:
        return
    contenders_added = 0

    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if lb_contenders:
        for entry in lb_contenders:
            uid = entry.get("uid")
            if uid is None or uid == king_uid or uid in challengers:
                continue
            if uid in valid_models:
                challengers[uid] = valid_models[uid]
                contenders_added += 1
        if contenders_added:
            logger.info(
                f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
                f"to eval (from H2H leaderboard)"
            )
        return

    scored = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is not None and 0 < kl < float("inf"):
            scored.append((uid, kl))
    scored.sort(key=lambda x: x[1])
    for uid, kl in scored[:TOP_N_ALWAYS_INCLUDE - 1]:
        challengers[uid] = valid_models[uid]
        contenders_added += 1
    if contenders_added:
        logger.info(
            f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
            f"to eval (from global scores — fallback)"
        )


def add_dormant_rotation(challengers, valid_models, state: ValidatorState,
                         king_uid, king_kl):
    """Rotate in ``DORMANT_ROTATION_N`` dormant miners with KL < king's.

    Picks the N best dormant scorers below ``king_kl`` so they can re-test.
    Skips king, current challengers, and permanently_bad_models.
    No-op when ``DORMANT_ROTATION_N=0`` or in single-eval mode.
    """
    if is_single_eval_mode():
        return
    if king_uid is None or DORMANT_ROTATION_N <= 0:
        return
    if king_kl is None or king_kl == float("inf"):
        return
    candidates = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info.get("model") in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is None or kl <= 0 or kl >= float("inf"):
            continue
        if kl >= king_kl:
            continue
        candidates.append((uid, kl))
    candidates.sort(key=lambda x: x[1])
    added = []
    for uid, kl in candidates[:DORMANT_ROTATION_N]:
        challengers[uid] = valid_models[uid]
        added.append((uid, kl))
    if added:
        roster = ", ".join(f"UID {u}(kl={k:.4f})" for u, k in added)
        logger.info(
            f"♻️  Dormant rotation: added {len(added)} of {len(candidates)} "
            f"candidates better than king_kl={king_kl:.4f}: {roster}"
        )


def cap_challengers(challengers, state: ValidatorState, king_uid):
    # Single-eval mode: registration burn is the spam control, no cap needed.
    if is_single_eval_mode():
        return
    phase = state.top4_leaderboard.get("phase", "maintenance")
    max_cap = 80 if phase == "initial_eval" else MAINTENANCE_CHALLENGER_CAP
    if len(challengers) <= max_cap:
        return
    logger.warning(f"{len(challengers)} challengers exceeds cap of {max_cap} (phase={phase}). Truncating.")
    king_entry = challengers.pop(king_uid, None)
    # Pin only the strongest H2H contenders; the rest still compete below.
    lb_entries = [
        entry for entry in (state.top4_leaderboard.get("contenders") or [])
        if entry.get("uid") is not None and entry.get("uid") != king_uid
    ]
    lb_rank = {entry.get("uid"): i for i, entry in enumerate(lb_entries)}
    protected_uids = {
        entry.get("uid") for entry in lb_entries[:max(0, PROTECTED_H2H_CONTENDERS)]
    }
    protected = {uid: info for uid, info in challengers.items() if uid in protected_uids}
    remaining = {uid: info for uid, info in challengers.items() if uid not in protected_uids}

    def priority(item):
        uid, info = item
        uid_str = str(uid)
        score = state.scores.get(uid_str)
        is_new = score is None and uid_str not in state.evaluated_uids
        is_lb = uid in lb_rank
        commit_block = int((info or {}).get("commit_block") or 0)
        # Sort order: new/unevaluated -> scored dormant -> H2H rank -> rest.
        if is_new:
            return (0, -commit_block, uid)
        if score is not None and 0 < score < float("inf"):
            return (1, float(score), -commit_block, uid)
        if is_lb:
            return (2, lb_rank[uid], -commit_block, uid)
        return (3, -commit_block, uid)

    sorted_remaining = sorted(remaining.items(), key=priority)
    slots_for_remaining = max(0, max_cap - len(protected) - (1 if king_entry else 0))
    challengers.clear()
    challengers.update(protected)
    challengers.update(dict(sorted_remaining[:slots_for_remaining]))
    if king_entry:
        challengers[king_uid] = king_entry
    if protected:
        logger.info(
            f"cap_challengers: protected {len(protected)} top-contender(s) "
            f"from truncation: {sorted(protected)}; cap={max_cap}"
        )


def assert_top_contenders_present(challengers, valid_models, state: ValidatorState, king_uid):
    """Regression guard: WARN if any H2H contender is absent from the round.

    Also auto-evicts ghost contenders after ``LB_PRECHECK_EVICTION_STREAK``
    precheck failures. No-op in single-eval mode.
    """
    if is_single_eval_mode():
        return
    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if not lb_contenders:
        return
    missing = []
    forced = []
    evicted = []
    kept = []
    for entry in lb_contenders:
        uid = entry.get("uid")
        if uid is None or uid == king_uid:
            kept.append(entry)
            continue
        in_valid = uid in valid_models
        model = (valid_models.get(uid) or {}).get("model") if in_valid else entry.get("model")
        if uid in challengers or in_valid:
            if entry.get("precheck_fail_streak"):
                entry["precheck_fail_streak"] = 0
            if uid in challengers:
                kept.append(entry)
                continue
            # Force a lost-but-valid H2H contender back instead of warning.
            if in_valid:
                challengers[uid] = valid_models[uid]
                forced.append({"uid": uid, "model": model, "h2h_kl": entry.get("h2h_kl") or entry.get("kl")})
                kept.append(entry)
                continue
        if not in_valid:
            entry["precheck_fail_streak"] = int(entry.get("precheck_fail_streak", 0)) + 1
            if entry["precheck_fail_streak"] >= LB_PRECHECK_EVICTION_STREAK:
                evicted.append({"uid": uid, "model": model,
                                "streak": entry["precheck_fail_streak"]})
                continue
        missing.append({
            "uid": uid,
            "model": model,
            "in_valid_models": in_valid,
            "in_bad_list": model in state.permanently_bad_models if model else None,
            "h2h_kl": entry.get("h2h_kl") or entry.get("kl"),
            "precheck_fail_streak": entry.get("precheck_fail_streak", 0),
        })
        kept.append(entry)
    if forced:
        roster = ", ".join(f"UID {e['uid']} ({e['model']})" for e in forced)
        logger.warning(
            f"🛡️  Forced {len(forced)} valid H2H leaderboard contender(s) "
            f"back into the eval round after cap/planning: {roster}"
        )
    if evicted:
        state.top4_leaderboard["contenders"] = kept
        try:
            state.save_top4()
        except Exception as exc:
            logger.warning(f"failed to persist leaderboard after eviction: {exc}")
        roster = ", ".join(f"UID {e['uid']} ({e['model']}, streak={e['streak']})" for e in evicted)
        logger.warning(
            f"🪦 Evicted {len(evicted)} ghost contender(s) from H2H leaderboard "
            f"after {LB_PRECHECK_EVICTION_STREAK}+ consecutive precheck failures: {roster}"
        )
    if missing:
        logger.warning(
            f"⚠️  TOP-CONTENDER REGRESSION CHECK: {len(missing)} H2H leaderboard "
            f"contender(s) NOT in this round: {missing}"
        )
    else:
        logger.info(
            f"✅ top-contender check: all {len(lb_contenders) - len(evicted)} H2H "
            f"leaderboard contender(s) present in round"
        )


def check_models_exist(models_to_eval, uid_to_hotkey, state: ValidatorState, commitments: dict):
    removed = []
    for uid in list(models_to_eval.keys()):
        model_repo = models_to_eval[uid]["model"]
        try:
            import urllib.request

            req = urllib.request.Request(f"https://huggingface.co/api/models/{model_repo}", method="HEAD")
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.warning(f"UID {uid} ({model_repo}): deleted from HF — DQ")
                hotkey = models_to_eval[uid].get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                commit_block = models_to_eval[uid].get("commit_block")
                disqualify(hotkey, f"Model {model_repo} no longer exists on HuggingFace (404)", state.dq_reasons, commit_block=commit_block)
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                state.evaluated_uids.add(str(uid))
                removed.append(uid)
    for uid in removed:
        models_to_eval.pop(uid, None)
    return removed
