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

    Missing record → "changed" (we have nothing on file). Records older than
    the new commitment → changed. Same model/revision/block → unchanged.
    """
    if not composite_record:
        return True
    stored = (
        str(composite_record.get("model") or ""),
        str(composite_record.get("revision") or "main"),
        composite_record.get("block"),
    )
    return stored != _commit_signature(current_info)


def evict_stale_evaluated_uids(state, valid_models: dict) -> list[str]:
    """Remove ``evaluated_uids`` entries whose stored commitment no longer
    matches the current on-chain commitment.

    Returns the list of evicted UID strings (for logging). The single-eval
    planner relies on ``evaluated_uids`` as the "seen this commitment" set,
    so this prevents a re-committed miner from being silently skipped.
    """
    evicted: list[str] = []
    for uid, info in valid_models.items():
        uid_str = str(uid)
        if uid_str not in state.evaluated_uids:
            continue
        stored = (state.composite_scores or {}).get(uid_str)
        if stored is None:
            continue
        if commitment_changed(stored, info):
            state.evaluated_uids.discard(uid_str)
            state.scores.pop(uid_str, None)
            state.composite_scores.pop(uid_str, None)
            evicted.append(uid_str)
    if evicted:
        logger.info(
            f"single-eval: evicted {len(evicted)} stale evaluated UIDs "
            f"(re-committed since last eval): {evicted}"
        )
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


def select_king_by_composite(
    state,
    valid_models: dict,
    uid_to_hotkey: dict | None = None,
    commitments: dict | None = None,
) -> tuple[int | None, dict | None]:
    """Pick the UID with the highest composite-worst from stored scores.

    Returns (uid, record) or (None, None) when no eligible composite-scored
    UID remains. ``record`` is the entry from ``state.composite_scores``.
    """
    candidates: list[tuple[float, float, int]] = []
    composite_scores = getattr(state, "composite_scores", {}) or {}
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
        candidates.append((worst_f, weighted_f, uid))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, _, top_uid = candidates[0]
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


def bootstrap_composite_from_h2h(state) -> int:
    """Seed ``state.composite_scores`` from the latest canonical H2H round.

    Used once on first switch to single-eval mode so the king-by-composite
    selector has data to compare against. Reads ``state.h2h_latest.results``
    (which contains composite annotations for every scored UID in that
    round) and writes one record per non-DQ, non-reference row that isn't
    already populated. Returns the number of seeded records.
    """
    if not isinstance(getattr(state, "composite_scores", None), dict):
        state.composite_scores = {}
    latest = getattr(state, "h2h_latest", None) or {}
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
    if seeded:
        logger.info(
            f"single-eval bootstrap: seeded {seeded} composite_scores "
            f"records from latest H2H (block={block})"
        )
    return seeded
