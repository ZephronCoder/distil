"""v31 procedural-axis package.

The v31 redesign collapses 21 hand-rolled bench axes into 11 axes, each
backed by a published research-grade procedural-benchmark methodology.
Spec lives in ``reports/2026-05-09-v31-procedural-redesign.md``.

Each axis is implemented as a self-contained module that exports a
``generate_items(block_seed: int, n_items: int) -> list[dict]`` function
returning items in the same shape as ``_generate_math_items`` (each item
is a dict with at least ``src``, ``question``, ``gold``). The dispatcher
in ``scripts/pod_eval_vllm.py`` registers each module in
``_BENCH_SAMPLE_GENERATORS`` with a SHADOW key prefix so the items are
computed and scored but the corresponding bench axis lands in the
composite at weight 0 until the promotion gate fires.

Promotion gate (per ``reports/2026-05-09-v31-procedural-redesign.md``):
1. Pearson r ≥ 0.5 with the corresponding held-out canary on at least
   4 paired UIDs in ``axis_correlation.json``.
2. Reference Qwen3.5-4B scores in [0.30, 0.80] on the axis.
3. Std-dev across 4 consecutive rounds < 0.10.

Modules currently shipped:

* ``math_gsm_symbolic`` — GSM-Symbolic methodology (Mirzadeh et al.
  Apple, 2024 / arXiv 2410.05229). Symbolic templates, P0/P1/P2
  difficulty levels, GSM-NoOp distractor variant.

Modules planned (one per future sprint):
* ``code_humaneval_plus`` — EvalPlus 80× test augmentation.
* ``ifeval_verifiable`` — Google IFEval kwarg surface.
* ``reasoning_logic_grid`` — Zebra puzzles.
* ``reasoning_dyval_arith`` — DyVal DAG generator.
* ``long_context_ruler`` — RULER 13-task suite.
* ``knowledge_multi_hop_kg`` — MuSiQue / Wikidata multi-hop.
* ``math_competition`` — AMPS-Hard.
* ``math_robustness`` — GSM-Plus 5-perturbation suite.
* ``truthfulness_calibration`` — TruthfulQA + GSM-Plus critical-thinking.
* ``consistency_paraphrase`` — cross-axis paraphrase consistency.
"""

from __future__ import annotations
