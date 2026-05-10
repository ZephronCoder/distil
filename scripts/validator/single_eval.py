"""Single-eval mode: one registration -> one commitment -> one eval.

When SINGLE_EVAL_MODE=1 (production default), rounds contain only new
submissions + the reference baseline; the king is picked cross-round
from state.composite_scores by composite.final. The legacy re-eval
path is restored by flipping the flag back off."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

from scripts.eval_policy import policy_env
from scripts.validator.composite import (
    COMPOSITE_SHADOW_VERSION,
    _blended_final_score,
)

logger = logging.getLogger("distillation.remote_validator")


def is_single_eval_mode() -> bool:
    """True when SINGLE_EVAL_MODE=1 (read each call so tests can flip it)."""
    return policy_env("SINGLE_EVAL_MODE", "0") == "1"


# Composite.final margin a challenger must clear to dethrone the king
# (live default 0.05; variance-reduction sweep raised the floor above 3%).
SINGLE_EVAL_DETHRONE_MARGIN = float(
    policy_env("SINGLE_EVAL_DETHRONE_MARGIN", "0.05")
)


# Per-round cap on never-evaluated commitments (FIFO by commit_block).
SINGLE_EVAL_MAX_PER_ROUND = int(policy_env("SINGLE_EVAL_MAX_PER_ROUND", "10"))


# When worst <= this epsilon, treat the UID as floor-saturated and break
# ties on ``weighted`` (otherwise a floor-saturated incumbent is unbeatable).
SINGLE_EVAL_WORST_FLOOR_EPSILON = float(
    policy_env("SINGLE_EVAL_WORST_FLOOR_EPSILON", "0.005")
)


def _composite_record_from_payload(
    comp: dict,
    *,
    model: str,
    revision: str,
    block: int | None,
    extras: dict | None = None,
) -> dict:
    """Project an h2h ``composite`` payload into the canonical on-disk record.
    ``extras`` is merged in last so callers can stamp path-specific fields."""
    def _maybe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    record: dict = {
        "worst": float(comp.get("worst")),
        # v30.2 — final ranking key + worst_3_mean.
        "final": _maybe_float(comp.get("final")),
        "worst_3_mean": _maybe_float(comp.get("worst_3_mean")),
        "final_alpha": comp.get("final_alpha"),
        "weighted": _maybe_float(comp.get("weighted")),
        "axes": dict(comp.get("axes") or {}),
        "n_axes": int(comp.get("present_count") or 0),
        "model": model,
        "revision": revision,
        "block": block,
        "ts": time.time(),
        "axis_spread": comp.get("axis_spread"),
        "bench_vs_rel_gap": comp.get("bench_vs_rel_gap"),
    }
    if extras:
        record.update(extras)
    return record


def _commit_signature(info: dict | None) -> tuple:
    """Unique-commitment tuple (model, revision, commit_block)."""
    if not info:
        return ("", "", None)
    return (
        str(info.get("model") or ""),
        str(info.get("revision") or "main"),
        info.get("commit_block"),
    )


def commitment_changed(
    composite_record: dict | None, current_info: dict | None
) -> bool:
    """True iff the stored composite record describes a different commitment.
    Bootstrap records (no commit_block/revision) compare on model name only."""
    if not composite_record:
        return True
    cur = _commit_signature(current_info)
    stored_model = str(composite_record.get("model") or "")
    if stored_model != cur[0]:
        return True
    if composite_record.get("_bootstrapped"):
        return False
    stored_rev = str(composite_record.get("revision") or "main")
    if stored_rev != cur[1]:
        return True
    return composite_record.get("block") != cur[2]


def evict_stale_evaluated_uids(state, valid_models: dict) -> list[str]:
    """Drop ``evaluated_uids`` + ``composite_scores`` rows whose on-chain
    commitment has moved since the last eval. Returns evicted UID strs."""
    evicted: list[str] = []
    composite_scores = state.composite_scores or {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        in_eu = uid_str in state.evaluated_uids
        in_cs = uid_str in composite_scores
        if not (in_eu or in_cs):
            continue
        stored = composite_scores.get(uid_str)
        if stored is None and in_eu:
            # precheck-DQ'd UIDs have no composite row; re-commits are still
            # picked up via the DQ row's commit_block.
            continue
        if stored is not None and commitment_changed(stored, info):
            state.evaluated_uids.discard(uid_str)
            state.scores.pop(uid_str, None)
            state.composite_scores.pop(uid_str, None)
            evicted.append(uid_str)
    if evicted:
        logger.info(
            f"single-eval: evicted {len(evicted)} stale evaluated UIDs "
            f"(re-committed since last eval): {evicted}"
        )
        _safe_persist_composite_scores(state, context="eviction")
    return evicted


def merge_composite_scores(
    state,
    h2h_results: list[dict],
    models_to_eval: dict,
    current_block: int | None,
) -> int:
    """Persist absolute composite scores for every UID scored this round.

    Always-on so we accumulate the ranking table whether SINGLE_EVAL_MODE is
    flipped or not — when the flag eventually flips, the table is already
    populated and the king-by-composite selector has data to work with.

    Returns the number of records updated. DQ rows and reference rows are
    skipped. Rows with no composite payload are skipped (e.g. probes errored).
    """
    if not isinstance(getattr(state, "composite_scores", None), dict):
        state.composite_scores = {}
    n_updated = 0
    for row in h2h_results or []:
        if row.get("disqualified") or row.get("is_reference"):
            continue
        comp = row.get("composite") or {}
        worst = comp.get("worst")
        if worst is None:
            continue
        uid = row.get("uid")
        if uid is None:
            continue
        uid_str = str(uid)
        info = models_to_eval.get(uid, {}) or {}
        record = _composite_record_from_payload(
            comp,
            model=info.get("model") or row.get("model") or "",
            revision=info.get("revision") or "main",
            block=info.get("commit_block") or current_block,
            extras={
                "present_count": int(comp.get("present_count") or 0),
                "broken_axes": list(comp.get("broken_axes") or []),
                "version": comp.get("version"),
            },
        )
        state.composite_scores[uid_str] = record
        n_updated += 1
        # One-eval-per-registration tracker (precheck rejects further
        # commits from the same hotkey). Persist coldkey for the Sybil
        # guard on cross-hotkey re-eval attempts.
        hotkey = info.get("hotkey") or ""
        if hotkey:
            if not isinstance(getattr(state, "evaluated_hotkeys", None), dict):
                state.evaluated_hotkeys = {}
            state.evaluated_hotkeys[hotkey] = {
                "uid": int(uid),
                "model": record["model"],
                "revision": record["revision"],
                "coldkey": info.get("coldkey"),
                "evaluated_at_block": record["block"],
                "evaluated_at_ts": record["ts"],
                "composite_final": record["final"],
                "composite_worst": record["worst"],
            }
    if n_updated:
        _safe_persist_composite_scores(state, context="merge")
        # Persist evaluated_hotkeys too so the policy survives validator
        # restarts. Failure is non-fatal (in-memory state remains correct).
        _safe_persist_evaluated_hotkeys(state)
    return n_updated


def _is_eligible_uid(
    uid: int,
    valid_models: dict,
    dq_reasons: dict,
    uid_to_hotkey: dict | None,
    commitments: dict | None,
) -> bool:
    """True iff UID is eligible to hold weights (valid_models, not DQ'd at its
    commit_block, not the reference row)."""
    from eval.scoring import is_disqualified

    if uid not in valid_models:
        return False
    info = valid_models.get(uid) or {}
    if info.get("is_reference"):
        return False
    hotkey = (uid_to_hotkey or {}).get(uid, info.get("hotkey", ""))
    commit_block = (
        (commitments or {}).get(uid, {}).get("block") or info.get("commit_block")
    )
    return not is_disqualified(uid, hotkey, dq_reasons, commit_block=commit_block)


def _is_kingship_eligible(
    state,
    uid: int,
    dq_reasons: dict,
    uid_to_hotkey: dict | None,
    commitments: dict | None,
) -> bool:
    """True iff this UID may hold the crown (weaker than _is_eligible_uid).

    Requires: on the current metagraph, hotkey matches the row we evaluated,
    not currently DQ'd, not the reference baseline.
    """
    from eval.scoring import is_disqualified

    if commitments is None or uid not in commitments:
        return False
    commit = commitments.get(uid) or {}
    if commit.get("is_reference"):
        return False
    chain_hotkey = (uid_to_hotkey or {}).get(uid) or commit.get("hotkey", "")
    if not chain_hotkey:
        return False
    # Drop the stored composite if a different miner now holds the slot
    # (UID was deregistered + re-registered under a new hotkey/model).
    composite_scores = getattr(state, "composite_scores", {}) or {}
    rec = composite_scores.get(str(uid)) or {}
    rec_model = rec.get("model")
    if rec_model and commit.get("model") and rec_model != commit.get("model"):
        return False
    commit_block = commit.get("block") or rec.get("block")
    if is_disqualified(
        uid, chain_hotkey, dq_reasons, commit_block=commit_block,
    ):
        return False
    return True


# Legacy bootstrap records (n=3-4) skip the harder benches and so
# carry inflated worst; 12 is the safe floor under the current schema.
# Records below the floor are tracked but not kingship-eligible.
_KING_SELECTION_MIN_AXES = 12

# Schema-version gate. ``select_king_by_composite`` prefers records
# graded by the current scoring code and falls back to legacy records
# only when no v_current candidate exists.
_KING_SELECTION_MIN_VERSION = COMPOSITE_SHADOW_VERSION


def select_king_by_composite(
    state,
    valid_models: dict,
    uid_to_hotkey: dict | None = None,
    commitments: dict | None = None,
) -> tuple[int | None, dict | None]:
    """Pick the king from stored composite scores.

    Returns (uid, record) or (None, None). Prior-king is a stability
    tiebreaker, not a hard lock; a measurably-better challenger must
    clear SINGLE_EVAL_DETHRONE_MARGIN to take the crown. Three-tier
    candidate fallback: schema-current v_current first, then any
    version with enough axes, then any record with a worst score."""
    composite_scores = getattr(state, "composite_scores", {}) or {}
    prior_king_uid = None
    h2h_latest = getattr(state, "h2h_latest", None) or {}
    try:
        prior_king_uid = (
            int(h2h_latest.get("king_uid"))
            if h2h_latest.get("king_uid") is not None
            else None
        )
    except (TypeError, ValueError):
        prior_king_uid = None

    def _build_candidates(
        min_axes: int,
        min_version: int | None = None,
    ) -> list[tuple[float, int, float, int]]:
        out: list[tuple[float, int, float, int]] = []
        for uid_str, rec in composite_scores.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            # Kingship pool is round-participants-only (apples-to-apples).
            if not _is_eligible_uid(
                uid, valid_models, state.dq_reasons,
                uid_to_hotkey, commitments,
            ):
                continue
            if min_version is not None:
                rec_version = rec.get("version")
                try:
                    rec_version_i = int(rec_version) if rec_version is not None else -1
                except (TypeError, ValueError):
                    rec_version_i = -1
                if rec_version_i < min_version:
                    continue
            n_axes = rec.get("n_axes")
            try:
                n_axes_i = int(n_axes) if n_axes is not None else 0
            except (TypeError, ValueError):
                n_axes_i = 0
            if n_axes_i < min_axes:
                continue
            # Primary sort key is ``final``; synthesize from worst+weighted
            # for legacy records using the current alpha.
            final_v = rec.get("final")
            if final_v is None:
                worst_v = rec.get("worst")
                weighted_v = rec.get("weighted")
                if worst_v is None and weighted_v is None:
                    continue
                try:
                    w_f = float(worst_v) if worst_v is not None else 0.0
                    wt_f = float(weighted_v) if weighted_v is not None else 0.0
                except (TypeError, ValueError):
                    continue
                # ``worst`` proxies worst_k_mean (legacy single-axis); blend
                # with weighted using the current shared bottom-weight alpha.
                final_f = _blended_final_score(w_f, wt_f) or 0.0
            else:
                try:
                    final_f = float(final_v)
                except (TypeError, ValueError):
                    continue
            if math.isnan(final_f) or math.isinf(final_f):
                continue
            weighted = rec.get("weighted")
            try:
                weighted_f = float(weighted) if weighted is not None else 0.0
            except (TypeError, ValueError):
                weighted_f = 0.0
            prior_bonus = 1 if uid == prior_king_uid else 0
            # Sort tuple: (final desc, weighted desc, prior_bonus desc, uid desc).
            # ``final`` is the canonical ranker; ``weighted`` is the
            # tiebreaker; ``prior_bonus`` only matters on exact ties to
            # avoid coin-flip churn.
            out.append((final_f, weighted_f, prior_bonus, uid))
        return out

    # Tier 1 — schema-current AND grader-current records. These were graded
    # under the latest scoring code (e.g. the long_context_bench confuser-
    # rejection grader). Strongly preferred so a stale-grader record can't
    # inherit kingship from inflated bench scores.
    candidates = _build_candidates(
        _KING_SELECTION_MIN_AXES, min_version=_KING_SELECTION_MIN_VERSION,
    )
    if not candidates:
        # Tier 2 — schema-current shape, any grader version. Used during the
        # transition window after a schema bump while v_current records are
        # still being collected, so we don't go kingless.
        candidates = _build_candidates(_KING_SELECTION_MIN_AXES)
    if not candidates:
        # Tier 3 — any record with a worst score. Bootstrap fallback.
        candidates = _build_candidates(0)
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, _, _, top_uid = candidates[0]
    top_record = composite_scores.get(str(top_uid))

    # Dethrone gate: a challenger must beat the prior king by
    # SINGLE_EVAL_DETHRONE_MARGIN on either ``worst`` or ``weighted``.
    # Applied even when the prior king is below the version filter so
    # the schema-bump transition can't strip the crown unmeasured.
    if prior_king_uid is not None and top_uid != prior_king_uid:
        prior_record = composite_scores.get(str(prior_king_uid))
        prior_eligible = (
            prior_record is not None
            and prior_record.get("worst") is not None
            and _is_eligible_uid(
                prior_king_uid, valid_models, state.dq_reasons,
                uid_to_hotkey, commitments,
            )
        )
        if prior_eligible and not resolve_dethrone(
            prior_king_uid, prior_record, top_uid, top_record,
        ):
            logger.info(
                f"single-eval: top candidate UID {top_uid} (worst={top_record.get('worst')}, "
                f"weighted={top_record.get('weighted')}, version={top_record.get('version')}) "
                f"did not clear dethrone margin against prior king UID {prior_king_uid} "
                f"(worst={prior_record.get('worst')}, weighted={prior_record.get('weighted')}, "
                f"version={prior_record.get('version')}); preserving prior king."
            )
            return prior_king_uid, prior_record
    return top_uid, top_record


def _dethrone_compare(
    incumbent_uid: int | None,
    incumbent_record: dict | None,
    challenger_uid: int,
    challenger_record: dict,
    margin: float,
    primary_key: str,
) -> bool:
    """Generic dethrone comparison on a primary key with a weighted fallback.

    Decision rule, applied in order:
      1. Clear win:        ``ch_primary > inc_primary × (1 + margin)``.
      2. Clear regression: ``ch_primary < inc_primary × (1 − margin)``.
      3. Tied region (within ±margin or both at the saturated floor):
         fall back to ``weighted`` × (1 + margin).

    ``primary_key`` is "final" for the v30.2+ blended rank and "worst"
    for legacy v28 records that predate the schema bump.
    """
    ch_p = (challenger_record or {}).get(primary_key)
    if ch_p is None:
        return False
    if incumbent_uid is None or not incumbent_record:
        return float(ch_p) > 0.0
    if challenger_uid == incumbent_uid:
        return True
    inc_p = incumbent_record.get(primary_key)
    if inc_p is None:
        return float(ch_p) > 0.0

    inc_pf = float(inc_p)
    ch_pf = float(ch_p)
    rel_margin = max(0.0, float(margin))

    both_saturated = (
        inc_pf <= SINGLE_EVAL_WORST_FLOOR_EPSILON
        and ch_pf <= SINGLE_EVAL_WORST_FLOOR_EPSILON
    )
    if not both_saturated:
        if ch_pf > inc_pf * (1.0 + rel_margin):
            return True
        if ch_pf < inc_pf * (1.0 - rel_margin):
            return False

    ch_w = (challenger_record or {}).get("weighted")
    inc_w = incumbent_record.get("weighted")
    if ch_w is None or inc_w is None:
        return False
    try:
        ch_wf = float(ch_w)
        inc_wf = float(inc_w)
    except (TypeError, ValueError):
        return False
    if inc_wf <= 0.0:
        return ch_wf > 0.0
    return ch_wf > inc_wf * (1.0 + rel_margin)


def resolve_dethrone(
    incumbent_uid: int | None,
    incumbent_record: dict | None,
    challenger_uid: int,
    challenger_record: dict,
    margin: float = SINGLE_EVAL_DETHRONE_MARGIN,
) -> bool:
    """Return True iff challenger should take the crown from incumbent.

    v30.2 (2026-04-29): the canonical dethrone key is ``final``
    (0.7·worst_3_mean + 0.3·weighted blend) rather than ``worst``
    (single-axis min). The blend smooths the single-axis-min noise
    pathology while preserving anti-Goodhart pressure (~70 % of the
    score is still bottom-3-axis driven).

    Backward compat: legacy v28 records without ``final`` fall back to
    the v28 ``worst``-based decision rule.
    """
    primary = "final" if (challenger_record or {}).get("final") is not None else "worst"
    return _dethrone_compare(
        incumbent_uid, incumbent_record,
        challenger_uid, challenger_record,
        margin, primary_key=primary,
    )


def _seed_one_h2h_round(state, latest: dict) -> int:
    """Seed composite_scores from a single H2H round payload.

    Skips any UID already present in ``state.composite_scores`` (older rounds
    must not overwrite newer data). Returns count of newly seeded records.
    """
    rows = latest.get("results") or []
    block = latest.get("block")
    seeded = 0
    for row in rows:
        if row.get("disqualified") or row.get("is_reference"):
            continue
        comp = row.get("composite") or {}
        worst = comp.get("worst")
        if worst is None:
            continue
        uid = row.get("uid")
        if uid is None:
            continue
        uid_str = str(uid)
        if uid_str in state.composite_scores:
            continue
        state.composite_scores[uid_str] = _composite_record_from_payload(
            comp,
            model=row.get("model") or "",
            revision=row.get("revision") or "main",
            block=block,
            extras={"_bootstrapped": True},
        )
        seeded += 1
    return seeded


def bootstrap_composite_from_h2h(state) -> int:
    """Seed ``state.composite_scores`` from every canonical H2H round we have.

    Originally only seeded from ``state.h2h_latest`` (one round = ~8 UIDs).
    But h2h_history contains every previous round, often with 60+ unique UIDs
    that were already scored before single-eval mode existed. Without those,
    a re-committed validator would re-evaluate 70+ historically-scored UIDs
    on the first round after restart — exactly the opposite of "one
    eval per commitment" (see Discord 2026-04-25, sebastian + leeroyjkin).

    Iteration order is newest → oldest so the most recent score for any UID
    wins (older rounds cannot overwrite a UID that's already been seeded
    from a more recent round). Persists immediately so a second restart
    can read from disk instead of re-walking history.

    Returns the total number of seeded records across all sources.
    """
    if not isinstance(getattr(state, "composite_scores", None), dict):
        state.composite_scores = {}
    latest = getattr(state, "h2h_latest", None) or {}
    seeded_latest = _seed_one_h2h_round(state, latest) if latest else 0
    history = getattr(state, "h2h_history", None) or []

    def _round_block(entry: dict) -> int:
        try:
            return int(entry.get("block") or entry.get("round_block") or 0)
        except (TypeError, ValueError):
            return 0

    seeded_history = 0
    for entry in sorted(history, key=_round_block, reverse=True):
        seeded_history += _seed_one_h2h_round(state, entry or {})
    seeded = seeded_latest + seeded_history
    if seeded:
        logger.info(
            f"single-eval bootstrap: seeded {seeded} composite_scores records "
            f"({seeded_latest} from latest H2H block={latest.get('block')}, "
            f"{seeded_history} from {len(history)} historical rounds)"
        )
        _safe_persist_composite_scores(state, context="bootstrap")
    return seeded


def persist_composite_scores(state) -> None:
    """Write ``state.composite_scores`` to disk immediately.

    Called eagerly after bootstrap and after every ``merge_composite_scores``
    so a validator restart never loses the canonical ranking table. The
    full ``state.save()`` path is also fine, but it only runs at end of
    round; if a round crashes mid-flight the bootstrap+merge work otherwise
    evaporates.
    """
    save_fn = getattr(state, "save_composite_scores", None)
    if callable(save_fn):
        save_fn()
        return
    save_state = getattr(state, "save", None)
    if callable(save_state):
        save_state()


def _safe_persist_composite_scores(state, *, context: str) -> None:
    """Persist composite_scores with a non-fatal warning on failure.

    Wraps the three call sites (eviction / merge / bootstrap) so the boilerplate
    try/except + warning lives in exactly one place.
    """
    try:
        persist_composite_scores(state)
    except Exception as exc:
        logger.warning(
            f"single-eval: failed to persist composite_scores after {context} "
            f"(non-fatal): {exc}"
        )


def _safe_persist_evaluated_hotkeys(state) -> None:
    """Persist evaluated_hotkeys table with a non-fatal warning on failure."""
    try:
        from eval.state import EVALUATED_HOTKEYS_FILE, atomic_json_write

        atomic_json_write(
            state._path(EVALUATED_HOTKEYS_FILE),
            state.evaluated_hotkeys,
            indent=2,
        )
    except Exception as exc:
        logger.warning(
            f"single-eval: failed to persist evaluated_hotkeys "
            f"(non-fatal): {exc}"
        )
