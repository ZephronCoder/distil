# v31 — Research-grounded procedural axis redesign

**Author:** SN97 ops · **Date:** 2026-05-09 · **Status:** PROMOTED — all 11 v31 axes live in production composite (see follow-up report `2026-05-09-v31-axis-promotion.md`).

## TL;DR

The user's call: treat all current axes as sunk cost; redesign the validator's
axis set from scratch around **published SOTA benchmarks** (gsm8k, humaneval,
bbh, mmlu_pro, ifeval, RULER, GPQA, etc.) — but generate each one
procedurally so it can't be memorised. This doc maps every category of
held-out benchmark to a procedural-feasibility verdict + a published source
generator we can adopt, then proposes a clean 9-axis set for v31.

The redesign collapses 21 ad-hoc bench axes into 9 research-grounded ones.
Each axis cites a target benchmark (what skill we want) and a published
procedural generator (how we test it without leaking items). The
distillation axes (on_policy_rkl, top_k_overlap, judge_probe, etc.) are
kept untouched — they're already Goodhart-resistant by construction.

## Why this is different from "make procedural benchmarks harder to game"

v30.8 (the design from `2026-05-09-anti-goodhart-procedural.md`) layers
moving distribution shifts, frontier item selection, and hidden weight
rotation on top of the *existing* axes. That's necessary but not sufficient
— the existing axes are organic accretion, not principled. A fresh look at
the literature reveals that:

1. **Every category of held-out benchmark has a published procedural counterpart.**
   We don't need to invent generators; the academic community has already
   built and validated them.
2. **The procedural counterparts are explicitly designed against contamination.**
   GSM-Symbolic, DyVal, RULER, EvalPlus, LiveBench — all 2024 papers
   targeting the exact problem we're trying to solve.
3. **Adopting them lets us cite published correlations to held-out skill,**
   instead of having to re-prove that our procedural axes correlate with
   real capability after the fact.

This is the right move. v30.8 is still useful (the moving-target ideas
apply on top of any axis set), but v31 is the bigger structural fix.

## Research base (citations)

| Paper | Year / venue | Innovation | Procedural mechanism |
|---|---|---|---|
| GSM-Symbolic [Mirzadeh et al., Apple] | 2024 / arXiv 2410.05229 | Symbolic templates for GSM8K with `{name}/{x}/{y}/{z}` placeholders. 100 templates, 50 instances each = 5K items per benchmark cut. P1/P2 difficulty by adding clauses. **GSM-NoOp:** distractor variant that adds seemingly-relevant irrelevant clauses; SOTA models drop up to 65%. | [github.com/apple/ml-gsm-symbolic](https://github.com/apple/ml-gsm-symbolic), HF dataset. |
| GSM-Plus [Li et al.] | 2024 / ACL | 10,552 perturbations of GSM8K via 5 strategies: numerical variation, arithmetic variation, paraphrase, distractor insertion, missing-info ("critical thinking"). | HF `qintongli/GSM-Plus`. |
| DyVal [Zhu et al., Microsoft] | 2024 / ICLR | DAG-based procedural generator for math, logic, algorithm tasks. Controllable complexity. ChatGPT drops 95% → 71% from complexity 1 → 4. | [github.com/microsoft/promptbench](https://github.com/microsoft/promptbench). |
| RULER [Hsieh et al., NVIDIA] | 2024 / ICLR | 13 long-context tasks (4 categories: NIAH variants, multi-hop tracing, aggregation, QA). All procedural; configurable sequence length and complexity. | [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER). |
| EvalPlus [Liu et al.] | 2023 / NeurIPS | HumanEval+ has 80× more tests than HumanEval; MBPP+ has 35×. Procedural test-case generation against existing problem sets. | [github.com/evalplus/evalplus](https://github.com/evalplus/evalplus). |
| LiveBench [White et al.] | 2024 / ICLR 2025 spotlight | 18 tasks, 6 categories (math, coding, reasoning, language, IF, data analysis). Monthly question refresh. Includes hardened versions of BBH, AMPS, IFEval. | [livebench.ai](https://livebench.ai/livebench.pdf), [github.com/LiveBench/LiveBench](https://github.com/LiveBench/LiveBench). |
| IFEval [Zhou et al., Google] | 2023 / arXiv 2311.07911 | 25 verifiable-instruction categories with parametric kwargs (`num_words`, `forbidden_words`, …). Designed for procedural generation. | [github.com/google-research/google-research/tree/master/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval). |
| Zebra Puzzles | 2018 (Heule) + 2024 (LiveBench) | Procedural generator: randomize people / attributes / constraint statements. | LiveBench cites public Github repo. |
| AutoBencher [Stanford] | 2024 / arXiv 2407.08351 | Declarative benchmark construction: GPT-4 iteratively proposes datasets satisfying difficulty + salience desiderata. 22% more model errors than existing benchmarks; identifies specific knowledge gaps (e.g., Permian Extinction). | Demonstrates procedural knowledge benchmark feasibility. |
| BenchAgents [Microsoft] | 2024 / ICLR 2025 | Multi-agent benchmark creation pipeline: planning, generation, verification, evaluation. Used for planning + constraint satisfaction. | Same workflow we'd want for adding new v31 axes. |
| MMLU-Pro [Wang et al., TIGER-Lab] | 2024 / NeurIPS | 10-option MCQ instead of 4. 14 domains, 12K curated questions. **Not procedural** but contamination-resistant via difficulty + reasoning emphasis. | Status quo for knowledge-MCQ; we adopt as static rotated pool. |
| GPQA [Rein et al.] | 2023 / arXiv 2311.12022 | Graduate-level Google-proof MCQ. Diamond subset = 198 questions. Hand-crafted by PhDs. **Not procedural** by design. | Status quo for graduate science MCQ; we don't try to procedurally replicate this. |
| R2E-Gym [Jain et al.] | 2025 / COLM | 8,100+ procedurally curated SWE environments from commits. SWE-GEN synthetic data recipe. 51% on SWE-Bench Verified. | [github.com/R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym). |

## Audit of current v30.7 axes (catalog of sunk cost)

### Bench axes (`_BENCH_AXIS_NAMES`) — 21 axes, all procedural

| Axis | Provenance | Research-cited? | Procedural quality | Verdict |
|---|---|---|---|---|
| `math_bench` | v29 narrative templates | partial — gsm8k-styled | 12 narrative subtypes + 18 legacy direct-compute = 30 templates, calibrated to ref-4B = 0.55–0.70 | replace with `math_gsm_symbolic` (100 templates, GSM-Symbolic methodology) |
| `code_bench` | hand-rolled | no | random function signatures + procedural tests | replace with `code_humaneval_plus` (EvalPlus methodology) |
| `reasoning_bench` | hand-rolled | no | multi-step deduction templates | replace with `reasoning_logic_grid` (Zebra) + `reasoning_dyval_arith` (DyVal) |
| `knowledge_bench` | hand-rolled | no | poor — synthesizing facts is unreliable | replace with `knowledge_pragmatic_qa` (private rotated pool, AutoBencher seed) |
| `ifeval_bench` | adapted from IFEval | YES (IFEval kwargs) | already procedural | upgrade to full v2 IFEval kwarg surface as `ifeval_verifiable` |
| `aime_bench` | hand-rolled | no | competition-style templates | fold into `math_competition` (AMPS-Hard methodology) |
| `mbpp_bench` | hand-rolled | no | overlaps `code_bench` | fold into `code_humaneval_plus` |
| `tool_use_bench` | hand-rolled | no | agentic Python tool use | keep as separate `agentic_tool_use` axis with its own scope |
| `self_consistency_bench` | majority-vote on math items | partial | not its own item generator; samples K-way | replace with `consistency_paraphrase` (cross-axis paraphrase consistency check) |
| `arc_bench` | adapted from ARC | partial — ARC items not procedural | small dynamic range (0.50 floor → 0.75 ceiling) | retired in v28 (weight 0); fold into `knowledge_pragmatic_qa` |
| `truthful_bench` | adapted from TruthfulQA | partial | retired in v28 (weight 0) | fold into `truthfulness_calibration` |
| `long_context_bench` | hand-rolled NIAH | partial — NIAH-style | not RULER-grade | replace with `long_context_ruler` (RULER 13-task suite) |
| `procedural_bench` | hand-rolled | no | low signal (4/round) | retire — DyVal covers this surface |
| `robustness_bench` | hand-rolled | partial — paraphrase wrapper | overlaps `math_bench` | fold into `math_robustness` (GSM-Plus methodology) |
| `noise_resistance_bench` | hand-rolled | no | weight 0 (audit found ref-4B = 0) | retire |
| `debug_bench` | hand-rolled | no | overlaps code | fold into `code_humaneval_plus` |
| `correction_bench` | hand-rolled | no | overlaps debug | fold into `code_humaneval_plus` |
| `multi_doc_synthesis_bench` | hand-rolled | no | overlaps long-context | fold into `long_context_ruler` |
| `calibration_bench` | hand-rolled | no | mixes IF + truthfulness | split into `ifeval_verifiable` + `truthfulness_calibration` |
| `refactor_bench` | hand-rolled | no | overlaps code | fold into `code_humaneval_plus` |
| `pragmatic_bench` | hand-rolled | no | weight 0 (no clear canary mapping) | fold into `knowledge_pragmatic_qa` |

**Net:** 21 → 9 axes. The reduction comes from collapsing the multi-axis
sprawl in code (5 axes → 1) and reasoning (3 axes → 2), and dropping the
weight-0 retired axes.

### Distillation / structural axes — kept untouched

These are already Goodhart-resistant by construction (require teacher
access or are statistical properties hard to fake):

* `on_policy_rkl` (Tiapkin 2025) — sample-then-KL; needs private teacher.
* `top_k_overlap` (Rethinking OPD 2026) — token-level set agreement with
  private teacher logits.
* `kl`, `kl_is`, `forking_rkl`, `entropy_aware_kl`, `tail_decoupled_kl`
  (research-cited, all KL variants on private teacher).
* `judge_probe`, `long_form_judge` — LLM-judge with private teacher.
* `long_gen_coherence` — statistical (loops, perplexity, vocab collapse).
* `length`, `degeneracy` — structural.
* `capability` — composite probe; keep as low-weight smoke test.

These remain at v30.7 weights. v31 changes only the *bench-axis* surface.

## Proposed v31 axis set (9 axes, all research-grounded)

### MATH (3 axes)

#### M1 · `math_gsm_symbolic`
- **Target:** GSM8K (Cobbe et al., 2021)
- **Procedural method:** GSM-Symbolic (Mirzadeh et al., 2024).
  - 100 templates with `{name}/{x}/{y}/{z}` placeholders.
  - Domain-aware variable sampling (numbers in template-specified ranges).
  - Per-round random instance from each template; total 50–100 items / round.
  - Difficulty knob: P0 (base), P1 (+1 clause), P2 (+2 clauses).
  - **GSM-NoOp variant** (30% of round): inject seemingly-relevant irrelevant clauses to detect pattern-matching.
- **Verification:** Numeric exact match on `#### N` answer (existing `_math_extract_answer` works).
- **Replaces:** `math_bench` (the v29 narrative templates become the seed for GSM-Symbolic templates).
- **Effort:** ~3 days. The Apple repo gives us templates + generator; we wire it into our procedural-item pipeline.

#### M2 · `math_competition`
- **Target:** MATH (Hendrycks et al., 2021), AMC, AIME.
- **Procedural method:** AMPS-Hard (LiveBench's procedural extension of AMPS). Algebra / geometry / calculus templates with closed-form gold answers.
- **Verification:** Numeric / symbolic-equivalent match (use sympy for symbolic check).
- **Replaces:** `aime_bench`.
- **Effort:** ~4 days. AMPS-Hard items are listed in the LiveBench repo; we adapt the generator.

#### M3 · `math_robustness`
- **Target:** GSM8K perturbation robustness.
- **Procedural method:** GSM-Plus (Li et al., 2024). 5 perturbation types stacked on top of M1's items.
  - Numerical variation (digit expansion / type conversion).
  - Arithmetic variation (reverse the operation).
  - Paraphrase (rephrase the prompt).
  - Distractor insertion (add irrelevant sentence).
  - Critical thinking (remove necessary info → expect refusal).
- **Verification:** Same numeric grade as M1, with score breakdown by perturbation type.
- **Replaces:** `robustness_bench`.
- **Effort:** ~2 days (mostly composes on top of M1).

### CODE (1 axis)

#### C1 · `code_humaneval_plus`
- **Target:** HumanEval (Chen et al., 2021), HumanEval+ / MBPP+ (Liu et al., 2023).
- **Procedural method:**
  - 40+ function-signature template families (string ops, list ops, arithmetic, math, recursion, parsing, …).
  - Per-round randomization of types, default values, signature names.
  - Test-case augmentation: 80× test multiplier per problem (EvalPlus methodology).
  - Sandbox execution with timeout / memory bounds.
- **Verification:** Run student code against generated tests; score = `tests_passed / tests_total`.
- **Replaces:** `code_bench`, `mbpp_bench`, `debug_bench`, `correction_bench`, `refactor_bench`.
- **Effort:** ~5–7 days. Big collapse — five axes folded into one. The current sandbox executor + per-axis scorer can be reused.

### REASONING (2 axes)

#### R1 · `reasoning_logic_grid`
- **Target:** BBH `logical_deduction_*`, `web_of_lies`, Zebra puzzles.
- **Procedural method:** Zebra-puzzle generator (LiveBench's adaptation of Heule 2018). Per-round random `(num_people, num_attributes, num_constraints)`.
- **Verification:** Programmatic check against the unique solution.
- **Replaces:** `reasoning_bench` (logical-deduction half).
- **Effort:** ~3 days.

#### R2 · `reasoning_dyval_arith`
- **Target:** BBH `multistep_arithmetic`, `boolean_expressions`, `tracking_shuffled_objects`.
- **Procedural method:** DyVal (Zhu et al., 2024). DAG generator with controllable depth and branching factor for arithmetic, boolean logic, deductive logic, abductive logic, reachability, max-sum-path.
- **Verification:** Programmatic from the DAG.
- **Replaces:** `reasoning_bench` (multi-step half), `procedural_bench`, `noise_resistance_bench`.
- **Effort:** ~4 days. Microsoft promptbench has reference Python.

### KNOWLEDGE (1 axis — fully procedural via multi-hop KG)

#### K1 · `knowledge_multi_hop_kg`
- **Target:** HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., TACL 2022). MuSiQue 2-4-hop QA dataset has 25K items, 3× larger human-machine gap than existing 2-hop datasets — explicitly designed against shortcut-finding.
- **Why this is the right framing:** the user's directive is "if it can be memorized, it can be gamed". Static knowledge benchmarks (MMLU, GPQA) ARE memorizable. But while the underlying facts in Wikidata are public, the **composition of an N-hop chain** is combinatorially infinite. A 4-hop chain over Wikidata's ~108M entities has on the order of `108M × avg_branching^3 ≈ 10^{20}` possible paths — un-memorizable by construction.
- **Procedural method:** Wikidata SPARQL query generator + MuSiQue-style chain composition.
  - Per-round private seed picks: `(pivot_entity, hop_count ∈ {2,3,4}, relation_path)`.
  - Walk the KG along randomised relation chains (`P31 instance-of`, `P279 subclass-of`, `P361 part-of`, `P706 located-in`, `P710 participant`, `P190 sister-city`, etc.).
  - Compose the question by template: "What is the {target_property} of the {relation_chain_phrasing} of {pivot_entity}?" with paraphrase variation.
  - Verification: re-traverse the same chain in Wikidata; gold answer = terminal entity's label.
- **Anti-shortcut measures (per MuSiQue methodology):** reject chains where the answer is reachable in fewer hops than intended (no shortcut paths). Reject chains where any intermediate entity is the answer (no near-miss leakage).
- **Verification:** Programmatic — exact match on entity label, with alias-set tolerance (Wikidata `P742` + `skos:altLabel`).
- **Cache:** Wikidata snapshot is mirrored locally (one-time ~80GB) to avoid network calls per round; refreshed quarterly.
- **Replaces:** `knowledge_bench`, `pragmatic_bench`, `arc_bench`.
- **Effort:** ~7 days. Build SPARQL chain generator + alias-tolerant grader + Wikidata mirror infra. Reference implementations: MuSiQue paper ([github.com/StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)) and Wikontic 2024 (arXiv 2512.00590).
- **Trade-off accepted:** procedurally-generated multi-hop questions can occasionally have ambiguous gold (multiple valid answers, e.g., "the daughter of X" when X has multiple daughters). Mitigation: chain generator rejects pivots with non-functional `target_property` (≥2 valid terminals) — only 1-to-1 chains are scored. This eliminates ~30% of candidate chains but makes scoring unambiguous.

### INSTRUCTION FOLLOWING (1 axis)

#### I1 · `ifeval_verifiable`
- **Target:** IFEval (Zhou et al., 2023, Google).
- **Procedural method:** IFEval is **already procedural** by design. Adopt all 25 verifiable-instruction kwargs (`num_words`, `num_paragraphs`, `forbidden_words`, `keyword_frequency`, …) and randomize args per round.
- **Verification:** Programmatic check from kwargs (existing IFEval Python).
- **Replaces:** `ifeval_bench`, `calibration_bench` (refusal-detection part).
- **Effort:** ~2 days. Google's reference implementation drops in cleanly.

### LONG CONTEXT (1 axis)

#### L1 · `long_context_ruler`
- **Target:** RULER (Hsieh et al., 2024).
- **Procedural method:** RULER's 13 tasks across 4 categories:
  - **Retrieval:** NIAH variants (single, multi-key, multi-value, multi-query).
  - **Multi-hop tracing:** variable tracking (coreference resolution).
  - **Aggregation:** common words extraction (CWE) — proxy for summarisation.
  - **QA with distractors:** SQuAD-style with red herrings.
- All procedural by design (NVIDIA's `niah.py`).
- **Verification:** Programmatic — exact match for retrieval, set-membership for aggregation, fuzzy-string for QA.
- **Replaces:** `long_context_bench`, `multi_doc_synthesis_bench`.
- **Effort:** ~4 days. NVIDIA's repo is clean and BSD-licensed.

### TRUTHFULNESS / CALIBRATION (1 axis)

#### T1 · `truthfulness_calibration`
- **Target:** TruthfulQA (Lin et al., 2022) + GSM-Plus "critical thinking" perturbation.
- **Procedural method:**
  - 50% items: hand-curated TruthfulQA-style traps (hand-curated, rotated quarterly).
  - 50% items: GSM-Plus critical-thinking perturbation — math/logic problems with key info removed; correct response is "insufficient information".
- **Verification:** MCQ for TruthfulQA half; refusal-detection regex for critical-thinking half.
- **Replaces:** `truthful_bench`, `calibration_bench` (truthfulness part).
- **Effort:** ~3 days for the procedural half; the curated half overlaps with K1.

### SELF-CONSISTENCY (1 axis, cross-cutting)

#### S1 · `consistency_paraphrase`
- **Target:** Self-consistency (Wang et al., 2023).
- **Procedural method:** Cross-axis. Sample 8 items from M1 / R1 / R2 / I1, paraphrase each 3 ways using the existing `_paraphrase_*` helpers, score consistency = `unique(answers) == 1`.
- **Verification:** Set-cardinality of model's answers across paraphrases.
- **Replaces:** `self_consistency_bench`.
- **Effort:** ~1 day (composes on top of other axes).

## Total v31 design

```
                       Procedural axes        Distillation axes (kept)
                       ────────────────────   ────────────────────────
math_gsm_symbolic       M1 (GSM-Symbolic)     on_policy_rkl
math_competition        M2 (AMPS-Hard)        top_k_overlap
math_robustness         M3 (GSM-Plus)         kl / kl_is / forking_rkl
code_humaneval_plus     C1 (EvalPlus)         entropy_aware_kl / tail_decoupled_kl
reasoning_logic_grid    R1 (Zebra/Heule)      judge_probe
reasoning_dyval_arith   R2 (DyVal)            long_form_judge
knowledge_multi_hop_kg  K1 (MuSiQue/Wikidata) long_gen_coherence
ifeval_verifiable       I1 (IFEval)           length / degeneracy / capability
long_context_ruler      L1 (RULER)            
truthfulness_calibration T1 (TruthfulQA + GSM-Plus)
consistency_paraphrase  S1 (cross-axis)
```

**All 11 procedural axes are fully procedural.** Knowledge is no
exception (per user directive 2026-05-09): a Wikidata-grounded multi-hop
chain generator gives an un-memorizable item space ≥10²⁰.

9 procedural axes (vs 21 in v30.7) + 13 distillation/structural axes
(unchanged from v30.7). Total: 22 axes (down from 34 in v30.7 if counting
all sub-axes).

## Migration plan

The full redesign is a 6–8 week project. Ship in sprints; each sprint
adds 1–2 axes as SHADOW (computed but weight 0), gathers 1 week of
correlation telemetry, then promotes if the correlation gate passes.

| Sprint | Week | New axes (SHADOW) | Promote (if r > 0.5 vs canary) | Decommission |
|---|---|---|---|---|
| 1 | this week | freeze design | — | — |
| 2 | week 2 | `math_gsm_symbolic` | n/a (collecting telemetry) | — |
| 3 | week 3 | `code_humaneval_plus`, `ifeval_verifiable` | `math_gsm_symbolic` (if r > 0.5) | `math_bench` weight → 0 |
| 4 | week 4 | `reasoning_logic_grid`, `long_context_ruler` | `code_humaneval_plus`, `ifeval_verifiable` | `code_bench/mbpp_bench/debug_bench/correction_bench/refactor_bench` weight → 0; `ifeval_bench` weight → 0 |
| 5 | week 5 | `math_robustness`, `reasoning_dyval_arith` | `reasoning_logic_grid`, `long_context_ruler` | `reasoning_bench/long_context_bench/multi_doc_synthesis_bench` weight → 0 |
| 6 | week 6 | `truthfulness_calibration`, `consistency_paraphrase` | `math_robustness`, `reasoning_dyval_arith` | `robustness_bench/procedural_bench/noise_resistance_bench` weight → 0 |
| 7 | week 7 | `knowledge_multi_hop_kg` (Wikidata mirror + chain generator) | `truthfulness_calibration`, `consistency_paraphrase` | `truthful_bench/calibration_bench/self_consistency_bench` weight → 0 |
| 8 | week 8 | — | `knowledge_multi_hop_kg` | `knowledge_bench/pragmatic_bench/arc_bench` weight → 0; rename composite → `v31` |

## Promotion gate

A new axis can be promoted from SHADOW to scoring iff:

1. **Correlation gate:** Pearson r ≥ 0.5 with the corresponding held-out
   benchmark on at least 4 paired UIDs in `axis_correlation.json`.
2. **Reference-floor gate:** the un-distilled reference model (Qwen3.5-4B
   today) scores between 0.30 and 0.80 on the axis. Below 0.30 = items
   too hard / eval-broken; above 0.80 = items too easy / no signal.
3. **Stability gate:** axis std-dev across 4 consecutive rounds < 0.10.
   Volatility means the procedural distribution is too narrow.

If any gate fails, the axis stays in SHADOW and we iterate on the
generator (more templates, wider parameter ranges, better calibration).

## Risk / open questions

- **Knowledge benchmark is now fully procedural** via Wikidata multi-hop
  composition (per user directive: "if it can be memorized, it can be
  gamed"). The combinatorial KG-chain space is un-memorizable. Trade-off
  accepted: occasional gold-ambiguity items must be filtered out via
  functional-property checks at chain-generation time.
- **Adoption legality.** All cited generators are open-source (BSD/MIT/Apache);
  verify compatibility with our license stack before importing. Apple's
  ml-gsm-symbolic is "research code" — we may need to re-implement the
  template engine.
- **Sandbox cost.** `code_humaneval_plus` runs untrusted code with
  procedural test cases. Wall-time per-round will rise (current `code_bench`
  is 18 items / round × 0.3s ≈ 5s; v31 with 80× tests = 400s without
  parallelism). Plan for parallel sandbox or lower per-round budget.
- **Cross-axis paraphrase axis (S1)** depends on M1/R1/R2/I1 being live.
  Don't ship it before sprint 6.
- **The composite weight allocation** — once all 9 axes are SHADOW-clean,
  we redo the weight assignment with the same anti-Goodhart principles
  as v30.7 (teacher-anchored axes get bulk of weight; bench axes are
  diversified, none > 0.15 individual weight).

## Sprint 1 — what shipped (2026-05-09, this session)

User directive: "do as much as you can now". Two SHADOW axes shipped
end-to-end with full integration, tests, and correlation tracking:

### M1 · `v31_math_gsm_symbolic` (SHADOW)

* **Module:** `scripts/v31/math_gsm_symbolic.py` (~470 LOC).
* **Methodology:** Apple GSM-Symbolic (Mirzadeh et al., 2024).
  Symbolic templates (8 base templates × 3 difficulty levels +
  GSM-NoOp distractor variant), per-template programmatic gold,
  per-round private RNG seed.
* **Distribution:** P0 50 %, P1 25 %, P2 15 %, NoOp 10 %.
* **Verification:** Re-uses `_math_extract_answer` /
  `_math_score_one`. Items emit the standard `#### N` answer marker.
* **Integration:**
  - Generator wrapper + dispatcher entry in `scripts/pod_eval_vllm.py`.
  - Bench probe `v31_gsm_symbolic_bench_probe`.
  - Sample-list key `v31_gsm_symbolic`.
  - Composite axis name `v31_math_gsm_symbolic` (weight 0).
  - Correlation tracker pair `v31_math_gsm_symbolic ↔ canary_gsm8k`.
* **Activation:** `BENCH_V31_GSM_SYMBOLIC_PER_ROUND > 0` env var.
  Default 0 (disabled).
* **Tests:** 18 / 18 pass (`tests/test_v31_math_gsm_symbolic.py`).

### I1 · `v31_ifeval_verifiable` (SHADOW)

* **Module:** `scripts/v31/ifeval_verifiable.py` (~430 LOC).
* **Methodology:** Google IFEval (Zhou et al., 2023). Full
  21-verifier surface from the vendored `ifeval_vendor.py`,
  stack depths 1-4 with per-round procedural kwargs.
* **Distribution:** stack 1: 25 %, stack 2: 30 %, stack 3: 25 %,
  stack 4: 20 % (pre-collapse). Exclusive verifiers
  (`constrained_response`, `json_format`) collapse the stack to
  1-2 — handled correctly via the post-collapse stack-depth field.
* **Verification:** Vendored `ifeval_vendor.evaluate_item` (same
  grader as the v30 `ifeval_bench`).
* **Integration:**
  - Generator wrapper + dispatcher entry in `scripts/pod_eval_vllm.py`.
  - Bench probe `v31_ifeval_verifiable_bench_probe` (full impl,
    not `_run_simple_bench` — IFEval grading is structural).
  - Sample-list key `v31_ifeval`.
  - Composite axis name `v31_ifeval_verifiable` (weight 0).
  - Correlation tracker pair `v31_ifeval_verifiable ↔ canary_ifeval`.
* **Activation:** `BENCH_V31_IFEVAL_PER_ROUND > 0` env var.
  Default 0 (disabled).
* **Tests:** 25 / 25 pass (`tests/test_v31_ifeval_verifiable.py`).

### Combined regression check

```
$ pytest tests/test_v31_math_gsm_symbolic.py \
         tests/test_v31_ifeval_verifiable.py \
         tests/test_arena_v3_composite.py
169 passed in 0.76s
```

All 23 bench axes still register cleanly; all three live services
(`distil-validator`, `distil-api`, `distil-dashboard`) remain active.

### How to activate sprint 1 SHADOW telemetry

The default deployment leaves both axes at 0 items / round (disabled).
To turn on SHADOW telemetry on a single validator without affecting
others:

```bash
sudo systemctl edit distil-validator
# add Environment lines:
#   Environment=BENCH_V31_GSM_SYMBOLIC_PER_ROUND=10
#   Environment=BENCH_V31_IFEVAL_PER_ROUND=8
sudo systemctl restart distil-validator
```

After ≥ 1 full eval cycle, run the correlation audit:

```bash
python scripts/audit/axis_correlation.py
# look for r >= 0.5 on (v31_math_gsm_symbolic, gsm8k) and
# (v31_ifeval_verifiable, ifeval). Promote axes that pass the gate
# by adjusting their weights in composite.py + eval_policy.json.
```

## Sprint 2+ — remaining axes (deferred)

Scope freeze for sprint 1: only M1 + I1 ship in SHADOW. The other
9 axes (M2 math_competition, M3 math_robustness, C1 code_humaneval_plus,
R1 reasoning_logic_grid, R2 reasoning_dyval_arith, K1 knowledge_multi_hop_kg,
L1 long_context_ruler, T1 truthfulness_calibration, S1 consistency_paraphrase)
are designed but not implemented. Each follows the same SHADOW recipe
established by M1 / I1:

1. New module under `scripts/v31/<axis>.py` with `generate_items`.
2. Generator wrapper + dispatcher entry in `pod_eval_vllm.py`.
3. Bench probe (`_run_simple_bench` style for math/code, full impl
   for axes with structural grading like IFEval).
4. Sample-list key + composite axis name (weight 0 SHADOW).
5. Correlation tracker pair in `axis_correlation.py`.
6. Unit tests under `tests/test_v31_<axis>.py`.
7. Default per-round count = 0 (env-overridable).

## References

- Mirzadeh et al., GSM-Symbolic, arXiv 2410.05229 (Apple, 2024)
- Li et al., GSM-Plus, ACL 2024
- Zhu et al., DyVal, ICLR 2024
- Hsieh et al., RULER, ICLR 2024 (NVIDIA)
- Liu et al., EvalPlus / HumanEval+, NeurIPS 2023
- White et al., LiveBench, ICLR 2025 spotlight
- Zhou et al., IFEval, arXiv 2311.07911 (Google, 2023)
- Wang et al., MMLU-Pro, NeurIPS 2024
- Rein et al., GPQA, arXiv 2311.12022
- Stanford, AutoBencher, arXiv 2407.08351
- Microsoft, BenchAgents, arXiv 2410.22584
- Suzgun et al., BIG-Bench Hard, arXiv 2210.09261
- Jain et al., R2E-Gym, COLM 2025
- Wang et al., Self-Consistency, ICLR 2023
- Lin et al., TruthfulQA, ACL 2022
- Hendrycks et al., MATH / AMPS, NeurIPS 2021 datasets

All papers also catalogued in the audit notes at `axis_correlation.py`
docstring + the v31 sprint tracker (TBD).
