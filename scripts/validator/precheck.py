import json
import logging
import math
import time

from eval.hf_upload_meta import get_first_upload_epoch
from eval.model_checker import (
    check_duplicate_content_hash,
    check_duplicate_hash,
    check_model_architecture,
    compute_content_hash,
    compute_model_hash,
    is_allowed_student_arch,
    register_content_hash,
    register_model_hash,
    verify_model_integrity,
)
from eval.runtime import STUDENT_ARCH_ALLOWLIST
from eval.scoring import (
    disqualify,
    get_dq_reason,
    is_disqualified,
    is_flagged,
    is_stale,
    record_failure,
    reset_failures,
)
from eval.state import ValidatorState, atomic_json_write
from scripts.validator.config import ACTIVATION_COPY_THRESHOLD, MAX_KL_THRESHOLD

logger = logging.getLogger("distillation.remote_validator")


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _hf_upload_griefing_swap(
    *,
    state_dir,
    this_uid,
    this_repo,
    this_revision,
    this_block,
    this_hotkey,
    orig_uid,
    orig_repo,
    orig_revision,
    orig_block,
    orig_hotkey,
    detection_label: str,
):
    """Tiebreak duplicate-detection by HuggingFace upload time.

    chain commit_block is not authoritative -- a miner can pre-commit a
    UID then upload stolen weights to the repo days later. HF records
    when safetensors actually became available, which is what copy
    detection should compare against.
    Returns the swap payload or None if HF agrees with chain / HF is
    unreachable.
    """
    try:
        this_ts = get_first_upload_epoch(this_repo, this_revision, state_dir=state_dir)
        orig_ts = get_first_upload_epoch(orig_repo, orig_revision, state_dir=state_dir)
    except Exception as exc:
        logger.info(f"hf_upload_meta lookup failed ({detection_label}): {exc}")
        return None
    if this_ts is None or orig_ts is None:
        return None
    if orig_ts <= this_ts:
        return None
    logger.warning(
        f"{detection_label} HF_UPLOAD_GRIEFING: chain says UID {orig_uid} ({orig_repo}) "
        f"committed earlier (block {orig_block}) than UID {this_uid} ({this_repo}) "
        f"(block {this_block}), but HF says {orig_repo} was uploaded "
        f"{int(orig_ts - this_ts)}s AFTER {this_repo}. Reversing DQ direction — "
        f"UID {orig_uid} flagged as the copy, UID {this_uid} protected."
    )
    return {
        "copy_uid": orig_uid,
        "copy_repo": orig_repo,
        "copy_block": orig_block,
        "copy_hotkey": orig_hotkey,
        "original_uid": this_uid,
        "original_repo": this_repo,
        "original_block": this_block,
        "original_hotkey": this_hotkey,
        "this_ts": this_ts,
        "orig_ts": orig_ts,
    }


def check_activation_fingerprint(model_name: str, uid: int, fingerprint: dict, state_dir,
                                 commit_block=None, uid_to_commit_block=None,
                                 uid_to_coldkey=None,
                                 evaluated_uids=None, composite_scores=None,
                                 revision: str = "main",
                                 uid_to_revision: dict = None):
    """Compare an incoming fingerprint to stored ones.

    Returns (is_copy, copy_uid, copy_model, original_uid, original_model, sim).

    Same-coldkey matches are reported but NOT DQ'd (miner iterating on their
    own model). Anti-griefing guard (requires ``evaluated_uids`` +
    ``composite_scores``): if the "later" side is evaluated and the "earlier"
    side never was, reverse the DQ direction -- the unevaluated, earlier UID
    is the pre-commit-then-copy attacker.
    """
    from pathlib import Path

    fp_file = Path(state_dir) / "activation_fingerprints.json"
    stored = {}
    if fp_file.exists():
        try:
            stored = json.loads(fp_file.read_text())
        except Exception:
            stored = {}
    layer_fps = fingerprint.get("layer_fingerprints", {})
    if not layer_fps:
        return False, None, None, None, None, 0.0
    max_sim = 0.0
    max_sim_uid = None
    max_sim_model = None
    max_sim_stored_block = None
    for other_uid_str, other_data in stored.items():
        try:
            other_uid = int(other_uid_str)
        except (TypeError, ValueError):
            continue
        if other_uid == uid:
            continue
        other_fps = other_data.get("layer_fingerprints", {})
        if not other_fps:
            continue
        sims = []
        for layer_key in layer_fps:
            if layer_key in other_fps:
                a = layer_fps[layer_key]
                b = other_fps[layer_key]
                if len(a) == len(b) and len(a) > 0:
                    sims.append(_cosine_sim(a, b))
        if sims:
            avg_sim = sum(sims) / len(sims)
            if avg_sim > max_sim:
                max_sim = avg_sim
                max_sim_uid = other_uid
                max_sim_model = other_data.get("model", "unknown")
                max_sim_stored_block = other_data.get("commit_block")

    is_copy = max_sim >= ACTIVATION_COPY_THRESHOLD

    copy_uid = uid
    copy_model = model_name
    original_uid = max_sim_uid
    original_model = max_sim_model

    if is_copy and uid_to_coldkey is not None and max_sim_uid is not None:
        my_ck = uid_to_coldkey.get(uid)
        other_ck = uid_to_coldkey.get(max_sim_uid)
        if my_ck and other_ck and my_ck == other_ck:
            logger.info(
                f"UID {uid} ({model_name}) activation-matches UID {max_sim_uid} "
                f"({max_sim_model}) at sim={max_sim:.6f} BUT they share coldkey "
                f"{my_ck[:12]}… — self-copy carve-out, skipping DQ for both sides."
            )
            is_copy = False

    if is_copy:
        # Resolve commit_block on both sides with layered fallback so we
        # don't flip-flop when the matched UID isn't in the current round.
        my_block = commit_block
        if my_block is None and uid_to_commit_block is not None:
            my_block = uid_to_commit_block.get(uid)
        other_block = None
        if uid_to_commit_block is not None and max_sim_uid is not None:
            other_block = uid_to_commit_block.get(max_sim_uid)
        if other_block is None:
            other_block = max_sim_stored_block
        try:
            my_b = float(my_block) if my_block is not None else None
        except (TypeError, ValueError):
            my_b = None
        try:
            other_b = float(other_block) if other_block is not None else None
        except (TypeError, ValueError):
            other_b = None
        # Safety: if commit order is unknown, do not DQ (the copy will be
        # caught next round once blocks are resolvable).
        if my_b is None or other_b is None:
            logger.warning(
                f"UID {uid} ({model_name}) activation-matched UID {max_sim_uid} "
                f"({max_sim_model}) at sim={max_sim:.6f} but commit_block unresolved "
                f"(my={my_block}, other={other_block}) — skipping DQ to avoid false positives. "
                f"Will re-evaluate once on-chain blocks are known."
            )
            is_copy = False
        else:
            if other_b > my_b:
                copy_uid = max_sim_uid
                copy_model = max_sim_model
                original_uid = uid
                original_model = model_name
            elif other_b == my_b and max_sim_uid is not None and max_sim_uid < uid:
                copy_uid = max_sim_uid
                copy_model = max_sim_model
                original_uid = uid
                original_model = model_name

        # Anti-griefing: reverse the DQ when a stale-but-evaluated UID
        # would lose to a never-evaluated pre-commit-then-copy attacker.
        if is_copy and evaluated_uids is not None and copy_uid is not None and original_uid is not None:
            def _is_real_eval(u):
                if u is None:
                    return False
                key = str(u)
                in_eval = key in evaluated_uids
                in_comp = composite_scores is None or key in composite_scores
                return in_eval and in_comp
            copy_eval = _is_real_eval(copy_uid)
            orig_eval = _is_real_eval(original_uid)
            if copy_eval and not orig_eval:
                logger.warning(
                    f"ACTIVATION GRIEFING: UID {original_uid} ({original_model}) committed earlier "
                    f"(block {other_b if original_uid == max_sim_uid else my_b}) but never evaluated; "
                    f"UID {copy_uid} ({copy_model}) has composite scores. Reversing DQ direction "
                    f"— UID {original_uid} flagged as the copy, UID {copy_uid} protected."
                )
                copy_uid, original_uid = original_uid, copy_uid
                copy_model, original_model = original_model, copy_model

        # HF-upload-time guard: chain commit_block can be pre-reserved;
        # HF records when safetensors actually became available.
        if is_copy and copy_uid is not None and original_uid is not None and copy_model and original_model:
            def _resolve_revision(u):
                if uid_to_revision and u in uid_to_revision:
                    return uid_to_revision[u]
                return revision if u == uid else "main"
            try:
                copy_ts = get_first_upload_epoch(copy_model, _resolve_revision(copy_uid), state_dir=state_dir)
                orig_ts = get_first_upload_epoch(original_model, _resolve_revision(original_uid), state_dir=state_dir)
            except Exception:
                copy_ts = orig_ts = None
            if copy_ts is not None and orig_ts is not None and orig_ts > copy_ts:
                logger.warning(
                    f"ACTIVATION HF_UPLOAD_GRIEFING: chain says UID {original_uid} ({original_model}) "
                    f"committed earlier than UID {copy_uid} ({copy_model}), but HF says "
                    f"{original_model} was uploaded {int(orig_ts - copy_ts)}s AFTER {copy_model}. "
                    f"Reversing DQ direction — UID {original_uid} flagged as the copy, "
                    f"UID {copy_uid} protected."
                )
                copy_uid, original_uid = original_uid, copy_uid
                copy_model, original_model = original_model, copy_model

    if not is_copy or copy_uid != uid:
        stored[str(uid)] = {
            "model": model_name,
            "layer_fingerprints": layer_fps,
            "n_layers": fingerprint.get("n_layers"),
            "hidden_size": fingerprint.get("hidden_size"),
            "commit_block": commit_block,
            "updated": time.time(),
        }
        try:
            atomic_json_write(fp_file, stored, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save fingerprints: {exc}")
    else:
        logger.info(
            f"UID {uid} ({model_name}) flagged as later-committed copy of UID {original_uid} "
            f"({original_model}) — NOT persisting fingerprint to keep the original canonical"
        )

    return is_copy, copy_uid, copy_model, original_uid, original_model, max_sim


def _one_eval_per_registration_enabled() -> bool:
    """ONE_EVAL_PER_REGISTRATION policy flag (default on)."""
    import os
    return bool(int(os.environ.get("ONE_EVAL_PER_REGISTRATION", "1") or 1))


def _check_registration_already_used(
    hotkey: str, model_repo: str, revision: str,
    state: ValidatorState,
    commit_block: int | None = None,
) -> tuple[bool, str | None]:
    """Has this hotkey already spent its one-eval slot?

    Returns (already_used, reason). Per-hotkey only: fresh hotkey eligible,
    same (model, revision) replay eligible, different (model, revision)
    rejected as recommit spam. Cross-hotkey enforcement is intentionally
    NOT applied -- coldkey clustering is dashboard-only.
    """
    if not hotkey or not _one_eval_per_registration_enabled():
        return False, None
    rec = (state.evaluated_hotkeys or {}).get(hotkey)
    if not rec:
        return False, None
    # Skip backfilled entries that predate this commit's block (recycled
    # UID with a stale record from the previous occupant).
    if rec.get("backfilled") and commit_block:
        eval_block = rec.get("evaluated_at_block", 0)
        if eval_block and eval_block < commit_block:
            logger.warning(
                f"one_eval_per_reg: ignoring stale backfilled entry for "
                f"{hotkey[:12]}… (eval block {eval_block} < commit block "
                f"{commit_block}), clearing it"
            )
            (state.evaluated_hotkeys or {}).pop(hotkey, None)
            return False, None
    prev_model = rec.get("model")
    prev_revision = rec.get("revision", "main")
    if prev_model == model_repo and prev_revision == revision:
        return False, None
    return True, (
        f"one_eval_per_registration: hotkey {hotkey[:12]}… already "
        f"evaluated {prev_model}@{prev_revision[:8]} at block "
        f"{rec.get('evaluated_at_block')}; new commit "
        f"{model_repo}@{revision[:8]} rejected. To get another eval, "
        f"register a new hotkey on chain."
    )


def _clear_uid_hash_and_stale_dqs(
    *,
    uid: int,
    hotkey: str,
    stored_hotkey,
    stored_commit_block,
    state: ValidatorState,
    log_reason: str,
    new_block,
    reset_failures_fn=None,
) -> None:
    """Drop a UID's stored hash + stale DQ entries so the next round runs a
    fresh integrity check. Clears both ``{hk}:{block}`` and legacy ``{hk}``
    DQ keys, discards ``evaluated_uids`` membership, and (optionally) resets
    failures."""
    logger.info(
        f"UID {uid}: {log_reason} at block {new_block} "
        f"(was {stored_commit_block}), resetting hash"
    )
    state.model_hashes.pop(str(uid), None)
    state.model_hashes.pop(f"{uid}_block", None)
    state.model_hashes.pop(f"{uid}_hotkey", None)
    for dq_hk in [hotkey, stored_hotkey] if stored_hotkey else [hotkey]:
        for dq_key in [f"{dq_hk}:{stored_commit_block}", dq_hk]:
            if dq_key and dq_key in state.dq_reasons:
                logger.info(f"UID {uid}: Clearing stale DQ: {dq_key}")
                del state.dq_reasons[dq_key]
    state.evaluated_uids.discard(str(uid))
    state.scores.pop(str(uid), None)
    if reset_failures_fn is not None:
        reset_failures_fn(uid, state.failures)


def _resolve_weight_duplicate(
    *,
    uid: int,
    model_repo: str,
    revision: str,
    hotkey: str,
    this_commit_block,
    this_block,
    duplicate_uid: int,
    hash_value: str,
    commit: dict,
    commitments: dict,
    uid_to_hotkey: dict,
    state: ValidatorState,
    valid_models: dict,
    disqualified: set,
    register_fn,
    detection_label: str,
    duplicate_log_label: str,
    griefing_log_label: str,
    weight_phrase: str,
    weight_suffix: str = "",
) -> bool:
    """Resolve a weight-duplicate detection (SHA or shard-invariant content hash).

    Pipeline:
      1. HF-upload-time tiebreak: swap DQ direction if the chain "original"
         was uploaded later than the "copy".
      2. Otherwise the later chain commit is the copy, unless the griefing
         guard fires (later UID evaluated, earlier UID never was -- earlier
         pre-reserved + uploaded king weights).
      3. Register the canonical hash under the keeper so future detections
         match against the right UID.

    The two callers differ only in (a) which ``register_fn`` /
    ``detection_label`` they pass and (b) the log/DQ wording.
    Centralising the resolver keeps fix-ups in lockstep — a bug fixed
    in one block was historically not fixed in the other (drift bug class).

    ``weight_suffix`` is appended after the model name in DQ reasons
    (content-hash uses " (re-sharded)"; SHA-hash leaves it empty).

    Returns ``True`` if ``uid`` itself was DQ'd and the caller should
    ``continue`` to the next UID; ``False`` if a different UID was DQ'd
    (or no DQ at all) and the caller should keep processing ``uid``.
    """
    orig_block = commitments.get(duplicate_uid, {}).get("block", float("inf"))
    orig_model = commitments.get(duplicate_uid, {}).get("model", "?")
    orig_revision = commitments.get(duplicate_uid, {}).get("revision", "main")
    orig_hotkey = uid_to_hotkey.get(duplicate_uid, str(duplicate_uid))
    orig_commit_block = commitments.get(duplicate_uid, {}).get("block")

    hf_swap = _hf_upload_griefing_swap(
        state_dir=state.state_dir,
        this_uid=uid, this_repo=model_repo, this_revision=revision,
        this_block=this_block, this_hotkey=hotkey,
        orig_uid=duplicate_uid, orig_repo=orig_model, orig_revision=orig_revision,
        orig_block=orig_block, orig_hotkey=orig_hotkey,
        detection_label=detection_label,
    )
    if hf_swap is not None:
        griefer_uid = hf_swap["copy_uid"]
        griefer_hotkey = hf_swap["copy_hotkey"]
        griefer_block = hf_swap["copy_block"]
        keeper_uid = hf_swap["original_uid"]
        keeper_repo = hf_swap["original_repo"]
        state.scores[str(griefer_uid)] = MAX_KL_THRESHOLD + 1
        disqualify(
            griefer_hotkey,
            f"copy: {weight_phrase} UID {keeper_uid} ({keeper_repo}){weight_suffix}, "
            f"HF upload griefing (chain block {griefer_block} earlier than victim, "
            f"but HF upload was {int(hf_swap['orig_ts'] - hf_swap['this_ts'])}s "
            f"after victim's upload)",
            state.dq_reasons,
            commit_block=griefer_block,
        )
        valid_models.pop(griefer_uid, None)
        disqualified.add(griefer_uid)
        register_fn(hash_value, keeper_uid, state.state_dir)
        return griefer_uid == uid

    this_evaluated = (
        str(uid) in state.evaluated_uids
        and str(uid) in state.composite_scores
    )
    orig_evaluated = (
        str(duplicate_uid) in state.evaluated_uids
        and str(duplicate_uid) in state.composite_scores
    )
    if this_block >= orig_block and not (this_evaluated and not orig_evaluated):
        logger.info(
            f"UID {uid} ({model_repo}): {duplicate_log_label} of UID {duplicate_uid}"
        )
        state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
        disqualify(
            hotkey,
            f"copy: {weight_phrase} UID {duplicate_uid} ({orig_model}){weight_suffix}, "
            f"committed later at block {this_block} vs {orig_block}",
            state.dq_reasons,
            commit_block=this_commit_block,
        )
        disqualified.add(uid)
        return True
    if this_evaluated and not orig_evaluated and this_block >= orig_block:
        logger.warning(
            f"UID {duplicate_uid} ({orig_model}): {griefing_log_label} — committed earlier "
            f"(block {orig_block}) but never evaluated; UID {uid} has composite scores. "
            f"DQ'ing the unevaluated model."
        )
        state.scores[str(duplicate_uid)] = MAX_KL_THRESHOLD + 1
        disqualify(
            orig_hotkey,
            f"copy: {weight_phrase} UID {uid} ({model_repo}){weight_suffix}, "
            f"griefing attack (committed earlier but never evaluated)",
            state.dq_reasons,
            commit_block=orig_commit_block,
        )
        valid_models.pop(duplicate_uid, None)
        disqualified.add(duplicate_uid)
        register_fn(hash_value, uid, state.state_dir)
    else:
        logger.info(
            f"UID {duplicate_uid} is {duplicate_log_label.lower()} of UID {uid} (committed earlier)"
        )
        state.scores[str(duplicate_uid)] = MAX_KL_THRESHOLD + 1
        disqualify(
            orig_hotkey,
            f"copy: {weight_phrase} UID {uid} ({model_repo}){weight_suffix}, committed later",
            state.dq_reasons,
            commit_block=orig_commit_block,
        )
        valid_models.pop(duplicate_uid, None)
        disqualified.add(duplicate_uid)
        register_fn(hash_value, uid, state.state_dir)
    return False


def precheck_all_models(commitments, uid_to_hotkey, uid_to_coldkey, state: ValidatorState, max_params_b: float):
    valid_models = {}
    disqualified = set()
    for uid, commit in commitments.items():
        model_repo = commit["model"]
        revision = commit.get("revision", "main")
        hotkey = commit.get("hotkey", uid_to_hotkey.get(uid, ""))
        this_commit_block = commit.get("block")
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=this_commit_block):
            reason = get_dq_reason(uid, hotkey, state.dq_reasons, commit_block=this_commit_block)
            logger.info(f"UID {uid} ({model_repo}): DISQUALIFIED — {reason}")
            disqualified.add(uid)
            continue
        # One-eval-per-registration: reject any commit from a hotkey
        # that already used its slot on a different (model, revision).
        coldkey = uid_to_coldkey.get(uid) if uid_to_coldkey else None
        already_used, reason = _check_registration_already_used(
            hotkey, model_repo, revision, state,
            commit_block=this_commit_block,
        )
        if already_used:
            logger.info(f"UID {uid} ({model_repo}): DISQUALIFIED — {reason}")
            disqualify(
                hotkey, reason, state.dq_reasons,
                commit_block=this_commit_block,
            )
            disqualified.add(uid)
            continue
        if state.scores.get(str(uid), 0) > MAX_KL_THRESHOLD:
            disqualified.add(uid)
            continue
        if is_stale(uid, state.failures):
            last_failed_model = state.failure_models.get(str(uid))
            current_model_key = f"{model_repo}@{revision}"
            if not last_failed_model:
                logger.info(f"UID {uid}: stale failure counter with no tracked model — resetting to retry {current_model_key}")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            elif last_failed_model != current_model_key and last_failed_model != model_repo:
                logger.info(f"UID {uid}: model changed from {last_failed_model} to {current_model_key}, resetting failure counter")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            elif last_failed_model == model_repo and last_failed_model != current_model_key:
                logger.info(f"UID {uid}: revision changed on {model_repo} (legacy pre-@-tracking entry), resetting failure counter")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            else:
                logger.info(f"UID {uid} ({current_model_key}): SKIPPED — stale ({state.failures.get(str(uid), 0)} failures on same model@revision). "
                            f"Push a new HuggingFace revision or commit a new model_repo on-chain to reset.")
                disqualified.add(uid)
                continue
        uid_str = str(uid)
        _needs_full_check = False
        if uid_str in state.evaluated_uids and uid_str in state.scores and state.scores[uid_str] <= MAX_KL_THRESHOLD:
            try:
                from huggingface_hub import hf_hub_download
                import json as _json

                cfg_path = hf_hub_download(model_repo, "config.json", revision=revision)
                with open(cfg_path) as handle:
                    cfg = _json.load(handle)
                archs = cfg.get("architectures", [])
                mtype = cfg.get("model_type", "")
                allowed, label = is_allowed_student_arch(mtype, archs)
                if not allowed:
                    allowed_pairs = ", ".join(
                        f"{e.get('model_type','?')}/{e.get('architecture','?')}"
                        for e in STUDENT_ARCH_ALLOWLIST if isinstance(e, dict)
                    ) or "(empty)"
                    logger.info(
                        f"UID {uid} ({model_repo}): FAIL — arch not in allowlist "
                        f"({mtype}/{','.join(archs)}); allowed: {allowed_pairs}"
                    )
                    record_failure(uid, state.failures, state.failure_models, f"{model_repo}@{revision}")
                    disqualify(
                        hotkey,
                        (
                            f"arch: Not in allowlist (found {','.join(archs)}, "
                            f"model_type={mtype}; allowed pairs: {allowed_pairs}). "
                            f"Fix: edit config.json on HuggingFace."
                        ),
                        state.dq_reasons,
                        commit_block=this_commit_block,
                    )
                    disqualified.add(uid)
                    state.scores.pop(uid_str, None)
                    state.evaluated_uids.discard(uid_str)
                    continue
            except Exception:
                pass
            expected_hash = state.model_hashes.get(str(uid))
            stored_hotkey_quick = state.model_hashes.get(f"{uid}_hotkey")
            stored_block_quick = state.model_hashes.get(f"{uid}_block")
            hotkey_changed_quick = stored_hotkey_quick is not None and stored_hotkey_quick != hotkey
            block_changed_quick = this_commit_block and stored_block_quick and this_commit_block != stored_block_quick
            if hotkey_changed_quick or block_changed_quick:
                reason = "hotkey changed (UID recycled)" if hotkey_changed_quick else "new commitment"
                _clear_uid_hash_and_stale_dqs(
                    uid=uid, hotkey=hotkey,
                    stored_hotkey=stored_hotkey_quick,
                    stored_commit_block=stored_block_quick,
                    state=state,
                    log_reason=f"quick re-check: {reason}",
                    new_block=this_commit_block,
                )
                expected_hash = None
                _needs_full_check = True
            if not _needs_full_check:
                integrity = verify_model_integrity(model_repo, revision, expected_hash)
                if integrity.get("transient"):
                    pass
                elif not integrity["pass"]:
                    logger.info(f"UID {uid} ({model_repo}): INTEGRITY FAIL — {integrity['reason']}")
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons, commit_block=this_commit_block)
                    disqualified.add(uid)
                    state.evaluated_uids.discard(uid_str)
                    continue
                valid_models[uid] = {
                    "model": model_repo,
                    "revision": revision,
                    "params_b": None,
                    "hotkey": hotkey,
                    "coldkey": coldkey,
                    "commit_block": this_commit_block if this_commit_block is not None else float("inf"),
                }
                continue
        if not _needs_full_check and uid_str in state.evaluated_uids:
            continue
        logger.info(f"Checking {model_repo}...")
        hf_user = model_repo.split("/")[0] if "/" in model_repo else None
        flag_reason = is_flagged(coldkey=coldkey, hf_username=hf_user, dq=state.dq_reasons)
        if flag_reason:
            logger.warning(f"UID {uid} FLAGGED: {flag_reason}")
        check = check_model_architecture(model_repo, revision, max_params_b)
        if check.get("transient"):
            logger.info(f"UID {uid} ({model_repo}): TRANSIENT ERROR — {check['reason']}, will retry next epoch")
            continue
        if not check["pass"]:
            logger.info(f"UID {uid} ({model_repo}): FAIL — {check['reason']}")
            record_failure(uid, state.failures, state.failure_models, f"{model_repo}@{revision}")
            disqualify(hotkey, f"arch: {check['reason']}", state.dq_reasons, coldkey=coldkey, hf_username=hf_user, commit_block=this_commit_block)
            disqualified.add(uid)
            continue
        model_hash = compute_model_hash(model_repo, revision)
        if model_hash:
            original_uid = check_duplicate_hash(model_hash, uid, state.state_dir)
            if original_uid is not None:
                this_block = commit.get("block", float("inf"))
                if _resolve_weight_duplicate(
                    uid=uid, model_repo=model_repo, revision=revision, hotkey=hotkey,
                    this_commit_block=this_commit_block, this_block=this_block,
                    duplicate_uid=original_uid, hash_value=model_hash,
                    commit=commit, commitments=commitments, uid_to_hotkey=uid_to_hotkey,
                    state=state, valid_models=valid_models, disqualified=disqualified,
                    register_fn=register_model_hash,
                    detection_label="SHA256-hash copy:",
                    duplicate_log_label="DUPLICATE",
                    griefing_log_label="GRIEFING COPY",
                    weight_phrase="identical weights to",
                ):
                    continue
            else:
                register_model_hash(model_hash, uid, state.state_dir)
        # Shard-invariant content hash catches re-sharded copies that slip
        # past compute_model_hash.
        content_hash = compute_content_hash(model_repo, revision)
        if content_hash:
            dup_uid = check_duplicate_content_hash(content_hash, uid, state.state_dir)
            if dup_uid is not None:
                this_block = commit.get("block", float("inf"))
                if _resolve_weight_duplicate(
                    uid=uid, model_repo=model_repo, revision=revision, hotkey=hotkey,
                    this_commit_block=this_commit_block, this_block=this_block,
                    duplicate_uid=dup_uid, hash_value=content_hash,
                    commit=commit, commitments=commitments, uid_to_hotkey=uid_to_hotkey,
                    state=state, valid_models=valid_models, disqualified=disqualified,
                    register_fn=register_content_hash,
                    detection_label="content-hash copy:",
                    duplicate_log_label="CONTENT-DUPLICATE",
                    griefing_log_label="GRIEFING COPY (content hash)",
                    weight_phrase="identical tensor content as",
                    weight_suffix=" (re-sharded)",
                ):
                    continue
            else:
                register_content_hash(content_hash, uid, state.state_dir)
        expected_hash = state.model_hashes.get(str(uid))
        stored_commit_block = state.model_hashes.get(f"{uid}_block")
        stored_hotkey = state.model_hashes.get(f"{uid}_hotkey")
        hotkey_changed = stored_hotkey is not None and stored_hotkey != hotkey
        block_changed = this_commit_block and stored_commit_block and this_commit_block != stored_commit_block
        legacy_no_block = expected_hash is not None and stored_commit_block is None and this_commit_block
        if hotkey_changed or block_changed or legacy_no_block:
            reason = "hotkey changed (UID recycled)" if hotkey_changed else "new commitment" if block_changed else "legacy hash (no block stored)"
            _clear_uid_hash_and_stale_dqs(
                uid=uid, hotkey=hotkey,
                stored_hotkey=stored_hotkey,
                stored_commit_block=stored_commit_block,
                state=state,
                log_reason=reason,
                new_block=this_commit_block,
                reset_failures_fn=reset_failures,
            )
            expected_hash = None
        integrity = verify_model_integrity(model_repo, revision, expected_hash)
        if integrity.get("transient"):
            logger.info(f"UID {uid} integrity: TRANSIENT ERROR — {integrity['reason']}, will retry")
            continue
        if not integrity["pass"]:
            logger.info(f"UID {uid} DISQUALIFIED: {integrity['reason']}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons, commit_block=this_commit_block)
            disqualified.add(uid)
            continue
        if integrity["current_hash"]:
            state.model_hashes[str(uid)] = integrity["current_hash"]
            if this_commit_block:
                state.model_hashes[f"{uid}_block"] = this_commit_block
            state.model_hashes[f"{uid}_hotkey"] = hotkey
            state.save_model_hashes()
        valid_models[uid] = {
            "model": model_repo,
            "revision": revision,
            "params_b": check.get("params_b", 0),
            "commit_block": commit.get("block", float("inf")),
            "hotkey": hotkey,
            "coldkey": coldkey,
            "vllm_compatible": check.get("vllm_compatible"),
            "vllm_reason": check.get("vllm_reason"),
        }
        logger.info(f"UID {uid}: {model_repo} ({check.get('params_b', 0):.2f}B) ✓")
    return valid_models, disqualified
