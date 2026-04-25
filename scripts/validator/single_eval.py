"""Single-eval mode helpers (one registration → one commitment → one eval).

Background (2026-04-25, distil-97): the previous round model re-evaluated the
king every round, plus the top-N H2H contenders, plus a dormant rotation, on
top of new submissions. Round sizes drifted to 12+ models and 90–120 minutes
of compute. The user policy update is "every commitment is evaluated exactly
once" — miners pay for the eval via the on-chain registration burn, and the
validator keeps a canonical per-UID composite that survives across rounds.

This module is the seam where the new policy is enforced. When
``SINGLE_EVAL_MODE=1`` is set in the validator environment:

* ``select_challengers`` returns only UIDs whose current commitment hasn't
  been composite-scored yet (driven by ``state.composite_scores``).
* ``add_top5_contenders`` / ``add_dormant_rotation`` / ``cap_challengers`` /
  ``assert_top_contenders_present`` are no-ops — there is no "re-eval slot"
  to fill.
* ``plan_round`` does NOT seat the king. Rounds contain only new
  submissions plus the always-in reference baseline.
* The crown is selected cross-round from ``state.composite_scores`` by the
  worst-axis composite, not from the round's paired t-test.

Anything outside the flag stays on the existing behavior, so the flip is
reversible.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger("distillation.remote_validator")


def is_single_eval_mode() -> bool:
    """Return True when the env flag enables single-eval policy.

    Read each call rather than at import time so we can unit-test the on/off
    paths via ``monkeypatch.setenv`` without re-importing the module.
    """
    return os.environ.get("SINGLE_EVAL_MODE", "0") == "1"


# Composite-worst margin a challenger must clear to dethrone the current king
# in single-eval mode. Mirrors ``EPSILON`` in the legacy KL-paired path so
# defaults stay symmetric: 3% margin = clearly better, not noise.
SINGLE_EVAL_DETHRONE_MARGIN = float(
    os.environ.get("SINGLE_EVAL_DETHRONE_MARGIN", "0.03")
)


# Hard ceiling on how many never-evaluated commitments enter a single round.
# When a backlog of new commits accumulates faster than rounds can consume
# them (e.g. after a 12 h restart loop), the planner would otherwise queue
# 20+ models per round and each round bloats to 8+ hours of pod compute.
# That hurts everyone: miners can't see results, base-model regression
# checks slow to a crawl, and a single bad node in the queue takes the
# whole round down with it. The cap forces FIFO rotation by commit_block
# (oldest commitment first), so every miner is evaluated within a few
# rounds without a single round being a multi-hour wallclock fire.
#
# Default 10 keeps round target around 60–75 min on H200 with shadow axes
# off. Override per-deployment with ``SINGLE_EVAL_MAX_PER_ROUND`` env.
SINGLE_EVAL_MAX_PER_ROUND = int(
    os.environ.get("SINGLE_EVAL_MAX_PER_ROUND", "10")
)


def _commit_signature(info: dict | None) -> tuple:
    """Return a tuple that uniquely identifies a commitment.

    Two commitments at the same UID are considered "different" if any of
    (model, revision, commit_block) changes. Used to decide whether a stored
    composite score is still valid for the current on-chain commitment.
    """
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

    Missing record → "changed" (we have nothing on file).

    Records produced by ``merge_composite_scores`` carry the UID's actual
    on-chain ``commit_block`` and ``revision`` — strict tuple match.

    Records produced by ``bootstrap_composite_from_h2h`` come from
    ``state.h2h_latest`` which only records the H2H round block (not the
    miner's commit_block) and frequently has ``revision=None`` for
    commitments that were stored without an explicit pin. We can only
    trust the **model name** for bootstrap records — comparing on block
    or revision evicts every seeded UID on first restart, which is what
    happened during round 8045570 (2026-04-25): all 8 prior-king-era
    UIDs were re-queued for evaluation and the round ballooned to 30+
    students. A model swap is still detected and re-eval'd.
    """
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
    """Remove stale composite + evaluated_uids entries when on-chain
    commitments have moved since the last eval.

    Iterates **every** UID in ``valid_models`` that has either an
    ``evaluated_uids`` flag or a ``composite_scores`` row, and drops the
    bookkeeping if the stored commitment no longer matches. Without
    this, a UID whose only stored row is a bootstrapped composite (no
    evaluated_uids entry) silently stays "filtered" forever even after
    the miner re-uploads — exactly the bug surfaced 2026-04-25 round
    8046286, where UID 12 re-committed natrium43/p1 → natrium43/t2 but
    the stale bootstrap record kept it out of the queue.

    Returns the list of evicted UID strings (for logging).
    """
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
            # No composite to compare against — leave evaluated_uids alone.
            # Edge case: precheck-DQ'd UIDs that never got a composite row.
            # Re-commits will still be picked up because the DQ row carries
            # the new commit_block; the planner ignores DQ'd UIDs anyway.
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
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after eviction (non-fatal): {exc}")
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
        record = {
            "worst": float(worst),
            "weighted": (
                float(comp["weighted"]) if comp.get("weighted") is not None else None
            ),
            "axes": dict(comp.get("axes") or {}),
            "n_axes": int(comp.get("present_count") or 0),
            "model": info.get("model") or row.get("model") or "",
            "revision": info.get("revision") or "main",
            "block": info.get("commit_block") or current_block,
            "ts": time.time(),
            "axis_spread": comp.get("axis_spread"),
            "bench_vs_rel_gap": comp.get("bench_vs_rel_gap"),
        }
        state.composite_scores[uid_str] = record
        n_updated += 1
    if n_updated:
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after merge (non-fatal): {exc}")
    return n_updated


def _is_eligible_uid(
    state,
    uid: int,
    valid_models: dict,
    dq_reasons: dict,
    uid_to_hotkey: dict | None,
    commitments: dict | None,
) -> bool:
    """Return True iff this UID is currently eligible to hold weights.

    Filters: must be in valid_models, not disqualified at its current
    commit_block, and not flagged as the always-in reference row.
    """
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


# Records produced before Arena v3 promotion (n_axes=3: on_policy_rkl, kl,
# capability) carry artificially high ``composite.worst`` because they
# never had to pass AIME/MBPP/tool-use/etc — those axes simply weren't
# scored. Mixing them into king selection lets a long-dormant 3-axis UID
# leapfrog a current 20-axis incumbent whose worst sits at 0 because of
# one hard axis. We require modern records to carry at least this many
# axes to participate in king selection.
_KING_SELECTION_MIN_AXES = 10


def select_king_by_composite(
    state,
    valid_models: dict,
    uid_to_hotkey: dict | None = None,
    commitments: dict | None = None,
) -> tuple[int | None, dict | None]:
    """Pick the UID with the highest composite-worst from stored scores.

    Returns (uid, record) or (None, None) when no eligible composite-scored
    UID remains. ``record`` is the entry from ``state.composite_scores``.

    Selection order (designed to keep the crown stable across the bootstrap
    path described in Discord 2026-04-25 — the broader history bootstrap
    was reseeding old 3-axis records that out-rank current 20-axis ones):

    1. **Trust the prior round's king** — if ``h2h_latest.king_uid`` exists,
       is eligible, and has a stored composite record, return them. The
       paired-KL test already chose this UID; we shouldn't dethrone via
       composite comparison until a real future round produces a clearly
       better challenger on the *current* schema.
    2. Otherwise, pick the UID with highest ``worst`` from records that
       were scored on the current Arena v3 schema (``n_axes >= 10``).
       Legacy 3-axis bootstrap records are skipped to avoid the apples-
       to-oranges comparison.
    3. Tie-breakers: prior-king bonus → ``weighted`` → UID.
    4. If no schema-current records are available (e.g. very early state),
       fall back to **all** records using the same tiebreakers — better
       to crown a stale legacy king than no king at all.
    """
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

    # Fast path: prior round's king is eligible and has composite data.
    if prior_king_uid is not None:
        prior_record = composite_scores.get(str(prior_king_uid))
        if prior_record and _is_eligible_uid(
            state, prior_king_uid, valid_models, state.dq_reasons,
            uid_to_hotkey, commitments,
        ):
            logger.info(
                f"single-eval: preserving prior king UID {prior_king_uid} from "
                f"h2h_latest (worst={prior_record.get('worst')}, "
                f"n_axes={prior_record.get('n_axes')})"
            )
            return prior_king_uid, prior_record

    def _build_candidates(min_axes: int) -> list[tuple[float, int, float, int]]:
        out: list[tuple[float, int, float, int]] = []
        for uid_str, rec in composite_scores.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            if not _is_eligible_uid(
                state, uid, valid_models, state.dq_reasons,
                uid_to_hotkey, commitments,
            ):
                continue
            n_axes = rec.get("n_axes")
            try:
                n_axes_i = int(n_axes) if n_axes is not None else 0
            except (TypeError, ValueError):
                n_axes_i = 0
            if n_axes_i < min_axes:
                continue
            worst = rec.get("worst")
            if worst is None:
                continue
            try:
                worst_f = float(worst)
            except (TypeError, ValueError):
                continue
            if math.isnan(worst_f) or math.isinf(worst_f):
                continue
            weighted = rec.get("weighted")
            try:
                weighted_f = float(weighted) if weighted is not None else 0.0
            except (TypeError, ValueError):
                weighted_f = 0.0
            prior_bonus = 1 if uid == prior_king_uid else 0
            out.append((worst_f, prior_bonus, weighted_f, uid))
        return out

    candidates = _build_candidates(_KING_SELECTION_MIN_AXES)
    if not candidates:
        # No schema-current records — fall back to legacy behaviour so
        # we still produce a king on a freshly-bootstrapped state.
        candidates = _build_candidates(0)
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, _, _, top_uid = candidates[0]
    return top_uid, composite_scores.get(str(top_uid))


def resolve_dethrone(
    incumbent_uid: int | None,
    incumbent_record: dict | None,
    challenger_uid: int,
    challenger_record: dict,
    margin: float = SINGLE_EVAL_DETHRONE_MARGIN,
) -> bool:
    """Return True iff challenger should take the crown from incumbent.

    Rule: challenger.worst > incumbent.worst * (1 + margin). If there is no
    incumbent record, any challenger with a positive composite_worst is
    eligible (covers the bootstrap case after a king regression / DQ).
    """
    ch_worst = (challenger_record or {}).get("worst")
    if ch_worst is None:
        return False
    if incumbent_uid is None or not incumbent_record:
        return float(ch_worst) > 0.0
    if challenger_uid == incumbent_uid:
        return True
    inc_worst = incumbent_record.get("worst")
    if inc_worst is None:
        return float(ch_worst) > 0.0
    threshold = float(inc_worst) * (1.0 + max(0.0, float(margin)))
    return float(ch_worst) > threshold


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
        state.composite_scores[uid_str] = {
            "worst": float(worst),
            "weighted": (
                float(comp["weighted"]) if comp.get("weighted") is not None else None
            ),
            "axes": dict(comp.get("axes") or {}),
            "n_axes": int(comp.get("present_count") or 0),
            "model": row.get("model") or "",
            "revision": row.get("revision") or "main",
            "block": block,
            "ts": time.time(),
            "axis_spread": comp.get("axis_spread"),
            "bench_vs_rel_gap": comp.get("bench_vs_rel_gap"),
            "_bootstrapped": True,
        }
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
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after bootstrap (non-fatal): {exc}")
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
