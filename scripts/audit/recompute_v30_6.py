"""v30.6 dry-run: recompute composite for every UID in h2h_history.

Loads the existing per-round student rows from ``state/h2h_history.json``
(the canonical record of what each row's raw axis values were) and
re-runs ``compute_composite`` with the v30.6 weights + canary axes.

Outputs a side-by-side table: legacy v30.2 final vs v30.6 final per UID,
plus the canary axis values that now anchor the king's composite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.validator.composite import (  # noqa: E402
    AXIS_WEIGHTS,
    BENCH_GROUP_AXIS_WEIGHTS,
    BENCH_AXIS_WEIGHTS,
    ARENA_V3_AXIS_WEIGHTS,
    CANARY_AXIS_WEIGHTS,
    JUDGE_AXIS_WEIGHT,
    LONG_FORM_JUDGE_AXIS_WEIGHT,
    LONG_GEN_COHERENCE_AXIS_WEIGHT,
    CHAT_TURNS_AXIS_WEIGHT,
    REASONING_DENSITY_WEIGHT,
    COMPOSITE_FINAL_BOTTOM_WEIGHT,
    SKILL_GROUP_AGGREGATION,
    compute_composite,
    load_canary_scores_for_uid,
)

H2H_HIST = REPO / "state" / "h2h_history.json"
COMPOSITE_NOW = REPO / "state" / "composite_scores.json"


def _print_weights():
    print("\n=== v30.6 weights (active in composite ranking) ===")
    rows = []
    rows.append(("AXIS", "weight"))
    rows.append(("─────", "──────"))
    for k, w in AXIS_WEIGHTS.items():
        if w > 0:
            rows.append((k, f"{w:.3f}"))
    for k, w in BENCH_GROUP_AXIS_WEIGHTS.items():
        if w > 0:
            rows.append((k, f"{w:.3f}"))
    for registry, label in (
        (BENCH_AXIS_WEIGHTS, "BENCH"),
        (ARENA_V3_AXIS_WEIGHTS, "ARENA"),
        (CANARY_AXIS_WEIGHTS, "CANARY"),
    ):
        for k, w in registry.items():
            if w > 0:
                rows.append((f"{k} [{label}]", f"{w:.3f}"))
    for axis, weight in (
        ("judge_probe", JUDGE_AXIS_WEIGHT),
        ("long_form_judge", LONG_FORM_JUDGE_AXIS_WEIGHT),
        ("long_gen_coherence", LONG_GEN_COHERENCE_AXIS_WEIGHT),
        ("chat_turns_probe", CHAT_TURNS_AXIS_WEIGHT),
        ("reasoning_density", REASONING_DENSITY_WEIGHT),
    ):
        if weight > 0:
            rows.append((axis, f"{weight:.3f}"))
    width = max(len(r[0]) for r in rows)
    for k, w in rows:
        print(f"  {k:<{width}}  {w}")
    total = sum(
        w for k, w in (
            list(AXIS_WEIGHTS.items())
            + list(BENCH_GROUP_AXIS_WEIGHTS.items())
            + list(BENCH_AXIS_WEIGHTS.items())
            + list(ARENA_V3_AXIS_WEIGHTS.items())
            + list(CANARY_AXIS_WEIGHTS.items())
            + [
                ("judge_probe", JUDGE_AXIS_WEIGHT),
                ("long_form_judge", LONG_FORM_JUDGE_AXIS_WEIGHT),
                ("long_gen_coherence", LONG_GEN_COHERENCE_AXIS_WEIGHT),
                ("chat_turns_probe", CHAT_TURNS_AXIS_WEIGHT),
                ("reasoning_density", REASONING_DENSITY_WEIGHT),
            ]
        )
        if w > 0
    )
    print(f"\n  TOTAL POSITIVE WEIGHT: {total:.3f}")
    print(f"  COMPOSITE_FINAL_BOTTOM_WEIGHT (alpha): {COMPOSITE_FINAL_BOTTOM_WEIGHT}")
    print(f"  SKILL_GROUP_AGGREGATION: {SKILL_GROUP_AGGREGATION}")


def _legacy_final(rec: dict) -> tuple[float | None, float | None, float | None]:
    return rec.get("final"), rec.get("worst_3_mean"), rec.get("weighted")


def _student_dict_from_axes(axes: dict) -> dict:
    """Recompute compute_composite from a per-axis dict by injecting the
    raw axis values directly through the student dict shape used by
    ``compute_axes``.

    We can't fully recompute from the round's student row because the
    h2h_history rounds drop the raw probe data (only axis values are
    persisted). So we reuse the existing axis values for the legacy
    axes and let canary_scores override the canary ones. Skill-group
    axes will recompute from sub-axes via _axis_skill_group_mean which
    needs the raw bench dicts -- so for the dry-run we build a minimal
    student dict that puts each pass_frac into the expected payload
    shape.
    """
    s = {}
    for axis, val in axes.items():
        if axis.endswith("_bench") and val is not None:
            s[axis] = {"n": 50, "correct": int(val * 50), "pass_frac": float(val)}
        elif axis == "capability" and val is not None:
            s["capability"] = {"pass_frac": float(val), "teacher_pass_frac": 1.0}
        elif axis == "kl" and val is not None:
            s["kl_global_avg"] = 1.0
        elif axis == "on_policy_rkl" and val is not None:
            s["on_policy_rkl"] = {"mean_rkl": 0.1}
        elif axis == "top_k_overlap" and val is not None:
            s["top_k_overlap_mean"] = float(val)
        elif axis == "length" and val is not None:
            s["length_axis"] = {"penalty": float(val)}
        elif axis == "degeneracy" and val is not None:
            s["think_probe"] = {
                "prompts_tested": 8,
                "prompts_terminated": int(float(val) * 8),
                "prompts_degenerate": 0,
                "self_bleu_across_prompts": 0.5,
                "teacher_self_bleu": 0.5,
            }
        elif axis == "judge_probe" and val is not None:
            s["judge_probe"] = {"normalized": float(val), "n_valid": 6}
        elif axis == "chat_turns_probe" and val is not None:
            s["chat_turns_probe"] = {"normalized": float(val), "n_valid": 6}
        elif axis == "long_form_judge" and val is not None:
            s["long_form_judge_probe"] = {
                "normalized": float(val), "n_valid": 4,
                "coherence_factor": float(axes.get("long_gen_coherence") or 0.5),
            }
    return s


def main() -> None:
    composite_now = json.loads(COMPOSITE_NOW.read_text())
    _print_weights()

    print("\n\n=== Composite recomputation: legacy v30.2 vs v30.6 ===")
    print(f"{'UID':>5} {'is_king':>7} {'model':<55} {'old_final':>10} {'new_final':>10}  {'delta':>7}  canary")
    print("─" * 120)
    rows = []
    king_uid_str = None
    for uid_str, rec in composite_now.items():
        if rec.get("is_king"):
            king_uid_str = uid_str
            break
    # Sort by old final desc
    items = sorted(composite_now.items(), key=lambda kv: -(kv[1].get("final") or 0.0))
    for uid_str, rec in items[:24]:
        try:
            uid_int = int(uid_str)
        except ValueError:
            continue
        axes = rec.get("axes") or {}
        student = _student_dict_from_axes(axes)
        canary = load_canary_scores_for_uid(uid_int)
        new = compute_composite(student, king_kl=1.0, king_rkl=None, canary_scores=canary)
        new_final = new.get("final")
        old_final = rec.get("final")
        is_king = "★" if rec.get("is_king") else ""
        delta = (new_final - old_final) if (new_final is not None and old_final is not None) else None
        canary_summary = ""
        if canary:
            canary_summary = " ".join(
                f"{k.replace('canary_',''):s}={canary.get(canary.get('__counts__', {}).get(k.replace('canary_',''), '') or k.replace('canary_',''))}"
                for k in ("canary_gsm8k","canary_humaneval","canary_bbh","canary_ifeval","canary_mmlu_pro")
                if canary.get(k.replace("canary_","")) is not None
            )
            # simpler: show real held-out scores
            canary_summary = " ".join(
                f"{key}={float(canary.get(key)):.2f}"
                for key in ("gsm8k","humaneval","bbh","ifeval","mmlu_pro")
                if canary.get(key) is not None
            )
        rows.append((uid_int, is_king, rec.get("model","")[:54],
                     old_final, new_final, delta, canary_summary))
        print(f"{uid_int:>5} {is_king:>7} {rec.get('model','')[:54]:<55} "
              f"{old_final or 0:>10.4f} {new_final or 0:>10.4f}  "
              f"{(delta or 0):>+.4f}  {canary_summary}")

    # King-specific deep dive
    if king_uid_str is not None:
        king_uid = int(king_uid_str)
        rec = composite_now[king_uid_str]
        axes = rec.get("axes") or {}
        canary = load_canary_scores_for_uid(king_uid)
        student = _student_dict_from_axes(axes)
        new = compute_composite(student, king_kl=1.0, king_rkl=None, canary_scores=canary)
        print(f"\n\n=== KING DEEP DIVE: UID {king_uid} ({rec.get('model')}) ===")
        print(f"  old final: {rec.get('final')}    old worst_3: {rec.get('worst_3_mean')}    old weighted: {rec.get('weighted')}")
        print(f"  new final: {new.get('final')}    new worst_3: {new.get('worst_3_mean')}    new weighted: {new.get('weighted')}")
        print("\n  King canary (held-out evalscope):")
        for k in ("gsm8k", "humaneval", "bbh", "ifeval", "mmlu_pro", "arc"):
            v = (canary or {}).get(k)
            cnt = (canary or {}).get("__counts__", {}).get(k)
            print(f"    {k:>12s}: {v}    n={cnt}")
        bottom = sorted(((k, v) for k, v in (new.get("axes") or {}).items() if v is not None),
                        key=lambda kv: kv[1])[:8]
        print("\n  v30.6 worst 8 axes (driving worst_3_mean):")
        for k, v in bottom:
            print(f"    {k:>30s}: {v:.4f}")


if __name__ == "__main__":
    main()
