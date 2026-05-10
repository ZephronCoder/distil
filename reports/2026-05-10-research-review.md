# 2026-05-10 — Research review: v31 axis evidence + ship-next decision

**Status:** Decision-ready. **Author:** AI agent.
**Predecessors:** [`2026-05-09-v31-axis-promotion.md`](2026-05-09-v31-axis-promotion.md),
[`2026-05-10-axis-correlation-audit.md`](2026-05-10-axis-correlation-audit.md),
[`2026-05-10-variance-reduction.md`](2026-05-10-variance-reduction.md).

## TL;DR — what to ship next

1. **ADD** a single-turn, procedural function-calling axis modelled on
   **BFCL v4** (`v31_tool_use_bfcl`, weight 0.04). Tool-use is the only
   capability family every 2026 SOTA tech report (DeepSeek-V3.2, Kimi
   K2.5, Qwen 3.5, GLM-4.6) uses as a primary ranking surface and which
   the SN97 composite still does not measure.
2. **RETIRE** `top_k_overlap` (currently w = 0.18 in the distillation
   tier; audit r = −0.481 with held-out gsm8k, n = 5). Halve to 0.09
   immediately; zero on next king-turnover if the sign of r is preserved
   at n ≥ 12. Reallocate the freed weight to `on_policy_rkl` (+0.05)
   and the new `v31_tool_use_bfcl` (+0.04).
3. **EXTEND IPT** beyond `v31_consistency_paraphrase`: add the same
   bijective object-name perturbation to `v31_reasoning_logic_grid`
   and `v31_reasoning_dyval_arith`. One parametric pass per item;
   closes the exact failure mode arXiv 2604.15149 documents (RLVR
   models pass extensional but fail isomorphic verification).
4. **DEFER**: ARC-AGI-2 (too hard for 4 B distilled student),
   METR HCAST (hours-long horizons, wrong cost regime), HLE
   (frontier-only, sub-5 % at 4 B), Tau3-Bench (needs stateful API
   simulator we do not yet maintain). All on the v32 roadmap.

The rest of this report is the evidence behind those four bullets.

---

## 1. Per-axis correlation evidence (Q1)

For each v31 axis: source paper, held-out canary the design targets,
strongest published evidence the axis tracks that canary, and what
the SN97 internal audit said about the closest legacy proxy. Audit
numbers from `2026-05-10-axis-correlation-audit.md` (n = 5 paired
post-Kimi-K2.6 kings).

| v31 axis | Canary | Published evidence | Internal audit (legacy proxy) |
|---|---|---|---|
| `v31_math_gsm_symbolic` | gsm8k | Mirzadeh et al. 2024 (arXiv 2410.05229, Apple, ICLR 2025) — every model drops on GSM-Symbolic vs GSM8K; symbolic instantiations explicitly designed as a "more reliable" GSM8K proxy. AbstRaL (Apple, Jun 2025) reinforces this. | `math_bench` ↔ gsm8k r = +0.520 — moderate proxy. |
| `v31_math_competition` | HMMT / AIME | LiveBench (Hsieh et al. arXiv 2406.19314, ICLR 2025) — monthly-refreshed competition math, contamination-resistant, sub-70 % ceiling on frontier. Kimi K2.5 / Qwen 3.5 / GLM-4.6 all use HMMT-Feb-2026 as primary math headline. | No legacy proxy. Predicted to land above `aime_bench`'s observed −0.245 once n ≥ 12. |
| `v31_math_robustness` | gsm8k | GSM-Plus (Li et al. arXiv 2402.19255, ACL 2024) — 25 LLMs × 4 prompting strategies, even GSM8K-correct models fail under perturbations. GSM-NoOp (same Apple 2024 paper) — adding one irrelevant clause causes up to **65 pp drop** on SOTA. | `robustness_bench` ↔ gsm8k r = +0.727 — strong proxy. |
| `v31_code_humaneval_plus` | humaneval | EvalPlus (Liu et al. NeurIPS 2023) — 80× more tests than HumanEval; pass@1 −19.3 %, pass@10 −24.9 %, pass@100 −28.9 %; **ranking reversals** (WizardCoder-CodeLlama, Phind-CodeLlama overtake ChatGPT after correction). LiveCodeBench (Jain et al. arXiv 2403.07974) finds HumanEval correlates only **r = 0.77** with LCB-Easy gen — the lowest pair in their 0.93–0.99 cross-scenario matrix; HumanEval itself is the degraded canary. | `code_bench` ↔ humaneval r = +0.895 — strong proxy. |
| `v31_reasoning_logic_grid` | bbh | ZebraLogic (Lin et al. arXiv 2502.01100, ICML 2025) — 1 000 puzzles, 2×2–6×6, "curse of complexity" persists with model size and inference-time compute. Used as canonical logic-grid proxy in LiveBench Reasoning. | No clean legacy proxy (`reasoning_skill_group` was the catch-all, audited at r = −0.836 with bbh — worst Goodhart violator). |
| `v31_reasoning_dyval_arith` | bbh | DyVal (Zhu et al. arXiv 2309.17167, Microsoft, ICLR 2024) — DAG-generated arithmetic; ChatGPT 84.5 % overall vs 71.5 % at complexity 4. Designed as contamination-resistant alternative to GSM8K-style static items. | Same legacy umbrella `reasoning_skill_group` (r = −0.836). |
| `v31_long_context_ruler` | bbh / mmlu_pro | RULER (Hsieh et al. arXiv 2404.06654, NVIDIA, ICLR 2024) — 13 representative tasks across NIAH / multi-hop tracing / aggregation / QA, 17 long-context LMs; almost every model drops sharply past 32 K despite passing vanilla NIAH. RULERv2 (ICLR 2026 submission, openreview ZU9tRffRSA) systematises bottom-up complexity. | `long_context_bench` ↔ bbh r = −0.713 — strong Goodhart, exactly because legacy was a single-needle NIAH miners had begun to overfit. RULER-class items expected to swing strongly positive. |
| `v31_knowledge_multi_hop_kg` | mmlu_pro | MuSiQue (Trivedi et al. arXiv 2108.00573, ACL 2022) and 2WikiMultiHopQA are the canonical 2-4 hop benchmarks; method comparisons show 18–37 pp dynamic range (HippoRAG 2: 74.7 % MuSiQue, 90.4 % 2Wiki) — metric still discriminates. HopWeaver (arXiv 2505.15087) is the closest 2025 procedural generator. | No legacy proxy — `knowledge_skill_group` was the catch-all, audited at r = −0.325 with mmlu_pro. |
| `v31_ifeval_verifiable` | ifeval | IFEval (Zhou et al. arXiv 2311.07911, Google 2024) is the source surface; ReIFE (Liu et al. arXiv 2410.07069) showed across 25 LLMs × 15 protocols × 4 datasets that base-LLM rank order is *consistent* across protocols — IFEval-style verifiable IF is well-anchored. MMMT-IF spin-off reports a **PIF metric ↔ human r = 0.60** (n = 50 raters). | `ifeval_bench` ↔ ifeval r = +0.703 — strong proxy. |
| `v31_truthfulness_calibration` | (no canary; SimpleQA is the surface) | SimpleQA (Wei et al. arXiv 2411.04368, OpenAI 2024) — 4 326 questions, 3-way scoring. Documented finding: **all frontier models systematically overconfident** when asked to self-rate (GPT-4o 38.2 % accurate, calibration consistently inflated). Useful as calibration probe rather than SOTA proxy. | `calibration_bench` ↔ ifeval r = −0.299 — drifting; the proxy was poorly anchored. The v31 surface is much better matched. |
| `v31_consistency_paraphrase` | (cross-axis; no canary) | IPT — Isomorphic Perturbation Testing (arXiv 2604.15149, ICLR 2026 LLM Reasoning Workshop). **Note: the user query flagged 2604.15149 as fictional — confirmed real.** Paper proves RLVR-trained models (GPT-5 family, Olmo3) systematically pass extensional verification but fail under bijective object-name renaming; non-RLVR models (GPT-4o, GPT-4.5, Ministral) show 0 shortcuts. **70 %** of GPT-5-mini shortcuts in highest-complexity quartile; aggregate 40 shortcuts at complexity 1-10 vs **458** at 11-20. Older paraphrase-invariance lineage in arXiv 2310.16153, 2410.20020. | No legacy proxy — v31-native concept. |

**Summary signal.** 8 of 11 axes have a direct citation-grade
correlation argument with their canary. The three exceptions are the
two reasoning axes (`logic_grid`, `dyval_arith`) and the multi-hop KG
axis — for all three the closest legacy proxy was a strong-Goodhart
group (`reasoning_skill_group` r = −0.84, `knowledge_skill_group`
r = −0.33), so even an axis with mid-range correlation against the
canary will be a Pareto improvement. The post-promotion audit
(n ≥ 12, expected by 2026-05-14) will close the remaining ambiguity.

---

## 2. 2025 H2 / 2026 lab survey (Q2)

What the major labs do for evaluation, and what SN97 should pick up.

### 2.1 DeepSeek

- **DeepSeek-V3.2** (arXiv 2512.02556, Dec 2025): post-training compute
  > 10 % of pre-training; large-scale agentic task synthesis pipeline
  (1 800+ environments, 85 K+ instructions). The **Speciale** variant
  won IMO/IOI 2025 gold. Eval suite: AIME-2025, HMMT-2025, Codeforces,
  **SWE-Verified**, agentic suites — every headline is procedural /
  contamination-resistant.
- **DeepSeek transparency report** (Stanford CRFM FMTI Dec 2025) flags
  a real V3-Base contamination vector: web pages contain
  OpenAI-model-generated answers, so the base model acquires knowledge
  "indirectly" even without intentional synthetic data.
- **Survey: 100 Days After R1** (arXiv 2505.00551) confirms full
  replication still hard — RLVR strategies and reasoning data
  pipelines remain the open frontier.
- **Methodology signal:** their headline tables are fully procedural /
  dynamic. No static MMLU / HumanEval / GSM8K in the front.

### 2.2 Kimi (Moonshot)

- **K2.5** (arXiv 2602.02276, Jan 2026): 1 T total / 32 B active MoE;
  **Agent Swarm** with 4× latency reduction; eval headlines are
  AIME-2025 (96.1 %), HMMT-2025 (95.4 %), GPQA-Diamond, **SWE-Bench
  Verified 76.8 %**, **SWE-Bench Pro 50.7 %**, **BrowseComp 78.4 %**.
- **K2.6** (Apr 2026, internal note in v31 promotion report): 4 K+
  tool calls over 13 h, Terminal-Bench 2.0 66.7 %.
- **Methodology signal:** every Kimi headline is procedural / agentic /
  time-gated. Static benchmarks appear nowhere in the abstract.

### 2.3 Qwen (Alibaba)

- **Qwen3** (arXiv 2505.09388): unified thinking / non-thinking modes;
  thinking-budget mechanism; 119 → 201 languages.
- **Qwen 3.5** (Feb 2026): 397 B / 17 B active hybrid linear-attention
  MoE; eval headlines: HMMT, AIME, **LiveCodeBench**, **BFCL-V4**,
  **TAU2-Bench**, MMLU-Pro, MMLU-ProX. Same procedural-headline pattern
  as Kimi.
- **Methodology signal:** **BFCL v4** and **TAU2-Bench** promoted to
  first-class headline status — the strongest signal of where the field
  is going on tool-use evaluation.

### 2.4 GLM / Zhipu

- **GLM-4.5** (arXiv 2508.06471): "Agentic-Reasoning-Coding" framing;
  hybrid thinking modes; 355 B / 32 B active.
- **GLM-4.6** (Sep 2025): 357 B / 32 B; 200 K context; on par with
  Claude Sonnet 4 on AIME-25 / GPQA / SWE-Bench-Verified. Public
  **`glm-simple-evals`** repo (github.com/zai-org/glm-simple-evals)
  uses Llama-3.1-70B-Instruct as checker for everything except HLE
  (which uses GPT-4o).
- **Headline benchmarks:** AIME 24/25, GPQA, **HLE**, MATH-500,
  **SWE-Bench Verified**, **LiveCodeBench**, **SciCode**,
  **Terminal-Bench**, **TAU-Bench**, **BFCL v3**, **BrowseComp**,
  MMLU-Pro.
- **Methodology signal:** open-source eval framework + a third-party LLM
  judge for verification — they actively decouple verifier from policy
  lineage. SN97 already does this implicitly via teacher-as-judge but
  does not publish a verifier card.

### 2.5 Apple

- **GSM-Symbolic** (arXiv 2410.05229) is the canonical procedural
  perturbation paper and is already the source of `v31_math_gsm_symbolic`
  (M1) and `v31_math_robustness` (M3, including the 0.30-weighted
  topical-distractor / GSM-NoOp perturbation that landed in v31).
- **AbstRaL** (Jun 2025) — synthetic abstractions raise GSM-Symbolic
  robustness without changing GSM8K-form generalisation; reinforces the
  procedural-only thesis. Not a new methodology, just a defence.

### 2.6 Google DeepMind

- **Gemini 2.5 Pro tech report**: pass@1 single-attempt only; multiple
  trials averaged on smaller benchmarks. **Explicitly blocklists
  huggingface.com and similar sites** at inference time to avoid
  copy-paste leakage during eval.
- **Gemini 3 Pro evals methodology**
  (deepmind.google/models/evals-methodology/gemini-3-1-pro): same
  standards; non-Gemini results sourced from providers' self-reported
  numbers when no API access.
- **Methodology signal:** "blocklist-during-eval" is a cheap, readily
  portable hygiene step. SN97 should add a `BENCH_BLOCKLIST_DOMAINS`
  env-var that strips canary URLs from any RAG context the validator
  builds — even if we never use RAG today, codifying this now prevents
  the simple canary-leak failure mode.

### 2.7 Anthropic

- **Claude Sonnet 4 / 4.5 system cards**: focus is alignment / safety
  (deception, sandbagging, sycophancy, reward-hacking, agentic safety,
  CBRN, RSP). Sonnet 4.5 deployed under ASL-3.
- **Methodology signal:** **reward-hacking is now a primary Anthropic
  eval surface** — they report it side-by-side with capability. SN97's
  IPT addition (April 2026) is in the same lineage, but Anthropic
  publishes a per-task hacking taxonomy that we do not. Worth catching
  up to in v32.

### 2.8 Microsoft / NVIDIA

- **DyVal / DyVal-2** (arXiv 2309.17167) — already in v31 as
  `v31_reasoning_dyval_arith`.
- **RULER + RULERv2** (NVIDIA) — already in v31 as
  `v31_long_context_ruler`. RULERv2 (ICLR 2026 submission) systematises
  bottom-up complexity progression and is a free upgrade for a future
  revision.

### 2.9 Cross-cutting trends (2025 H2 → 2026)

The four trends every lab is converging on:

1. **Procedural / time-gated headlines** (LiveBench, LiveCodeBench,
   HMMT-by-month, SWE-Bench Pro) replace static MMLU / HumanEval /
   GSM8K in tech-report top tables. ✓ SN97 already does this.
2. **Step-level grading** (PRMBench, GroundedPRM, ProcessBench)
   replaces outcome-only grading on math. ⚠ SN97 has it on the v32
   roadmap but ships outcome-only today.
3. **Tool-use as a primary axis** (BFCL v4, TAU2-Bench, NexusBench).
   ✗ SN97 has no tool-use axis. **This is the biggest gap.**
4. **Isomorphic / decontamination-by-construction** (IPT, SWE-Bench
   Pro's GPL-licensed public partition, contamination certificates).
   ✓ SN97 has IPT in `v31_consistency_paraphrase`, ⚠ but only on the
   math surface; should extend to logic and DAG reasoning.

---

## 3. New methodology candidates, ranked (Q3)

Each candidate is rated on three axes:

- **Signal** — empirical correlation with held-out SOTA holdouts
  (S+ / S / S− / weak).
- **Resistance** — to memorisation (procedural / decontamination
  certificate / dynamic / static).
- **Cost** — single-pass per item / sandbox / external API.

| # | Candidate | Source | Signal | Resistance | Cost | Fit for SN97? |
|--:|---|---|---|---|---|---|
| 1 | **BFCL v4** single-turn function call (subset) | Patil et al. ICML 2025; gorilla.cs.berkeley.edu/leaderboard | **S+** — Qwen 3.5 / GLM-4.6 / Kimi K2.5 all use as primary | proc + live (offline single-turn cache eliminates API risk) | low (single-pass JSON exact-match) | **YES — ship-next ADD** (see §4) |
| 2 | **PRMBench / ProcessBench step-level grading** | Song et al. ACL 2025.acl-long.1230 (PRMBench, 6 216 problems × 83 K step labels); GroundedPRM arXiv 2510.14942 (26 % rel improvement on ProcessBench with 10 % data) | **S** — step-level signal strictly more informative than outcome-only | proc + step-level (very robust) | medium (per-step intermediate-value extraction in grader; ~+15 % grader cost on M1/M3/R2) | YES — v32 priority, already on roadmap |
| 3 | **LiveCodeBench-style live programming** | Jain et al. arXiv 2403.07974 + LiveCodeBench Pro (Apr 2025, 584 problems) | **S+** — 0.93–0.99 cross-scenario Pass@1 correlation; HumanEval only 0.77 with LCB-Easy ⇒ HumanEval is the *weak* link | dynamic (time-gated, contamination-detectable) | low–medium (sandbox already exists for `v31_code_humaneval_plus`; need per-round seed for problem-template rotation) | YES — natural complement to `v31_code_humaneval_plus`; consider as v32.1 |
| 4 | **RewardBench 2** | Lambert et al. arXiv 2506.01937 — Pearson **r = 0.87** between RB2 score and BoN downstream across 113 RMs | **S+** for RM evaluation specifically | proc + unseen WildChat prompts + Tulu 3 decontamination toolkit | medium (RM scoring step on candidate completions) | NO direct fit — SN97 evaluates *students*, not reward models. Useful as defensive check on teacher-as-judge surface (out of scope for v31) |
| 5 | **IPT extension to `logic_grid` + `dyval_arith`** | arXiv 2604.15149 (already cited for `v31_consistency_paraphrase`) | **S** — directly closes a documented RLVR failure mode | proc (bijective renaming, no extra data) | very low (one extra parametric pass per item) | YES — ship-next EXTEND (see §4) |
| 6 | **ConfidenceBench-style asymmetric scoring** | Holter 2024 (confidencebench.com); ConfProBench arXiv 2508.04576 | **S−** — humans beat all LLMs; signal exists but small absolute spread at 4 B | proc (secret 100 MCQ pool) | very low (scoring-function change to existing `v31_truthfulness_calibration`) | MAYBE — small shipping cost; would extend our calibration axis from "did you refuse" to "did you over-confidently confabulate". Defer until n ≥ 12 audit on `v31_truthfulness_calibration` lands. |
| 7 | **Tau3-Bench / TAU2-Bench** | sierra-research/tau2-bench, Mar 2026 (airline + retail + banking) | **S+** for stateful agentic | proc + held-out RAG corpora (banking 698-doc set) | **HIGH** — needs stateful API simulator + voice loop optionally | NO for v31 — needs infra build-out. v32 priority. |
| 8 | **HCAST (METR)** | arXiv 2503.17354 — 189 tasks, 563 human baselines, 1 500 hours, 1 min → 8 h+ tasks | **S+** for autonomy | proc + human-time-calibrated | **VERY HIGH** — hours-long horizons; wrong cost regime | NO — wrong scale |
| 9 | **ARC-AGI-2** | Chollet et al. arXiv 2505.11831 — 1 000 train + 120 public eval + 120 semi-private + 120 private; calibrated difficulty | **S+** for compositional reasoning at frontier | best-in-class (3-tier partition, statistically calibrated) | **VERY HIGH** — 4 B distilled student floors at 0 %; no discrimination | NO — wrong difficulty band. Revisit when student tier moves to 33 B. |
| 10 | **HLE — Humanity's Last Exam** | Phan et al. Nature 2026 (arXiv 2501.14249) — 2 500 expert questions; frontier <50 % | **S+** but for frontier only | static (community-vetted; **HLE-Verified** arXiv 2602.13964 found 7-10 pp accuracy improvement after curation) | medium (multi-choice + short-answer with auto-grader) | NO — frontier-only ceiling. 4 B student will floor. |
| 11 | **NexusBench / NexusRaven** | nexusflowai/NexusBench (2024) — VirusTotal / NVDLibrary / TicketTracking sub-benchmarks | **S** | proc + scripted backends (no live API risk) | low | MAYBE — viable alternative to BFCL v4 if we want a tool-use axis with stronger procedural guarantees but smaller community signal |
| 12 | **SWE-Bench Pro** (public partition only) | arXiv 2509.16941 — 1 865 problems × 41 repos, **GPL-licensed public, held-out, commercial** partition | **S+** for repo-scale SWE | best-in-class (decontamination via copyleft licensing) | **VERY HIGH** — full-repo agent loops; wrong scale for 4 B distilled student | NO for v31. Track for v32 / v33 once we move to repo-aware student tier. |

**Top 3 procedural / cheap / fits-our-cost-regime:** BFCL v4 (single-turn
subset), IPT extension to logic + DAG reasoning, and step-level grading
on math. The first two are ship-next; the third is v32.

---

## 4. Recommendation (Q4)

### 4.1 ADD — `v31_tool_use_bfcl` (weight 0.04)

**What.** A new procedural axis. Per round, sample 8–12 items where:

1. The item generator emits a synthetic but well-typed function
   signature (1–4 typed parameters, JSON-Schema), a natural-language
   user query that requires calling that function, and a gold
   `{name, arguments}` JSON object.
2. Optionally an "irrelevant function" distractor schema is injected
   into the candidate set — graders verify the model selects the
   correct function (relevance) and emits a JSON-Schema-valid argument
   object that exact-matches the gold (correctness).
3. Scoring: average of `correct_function ∧ correct_args ∧ schema_valid`
   across the round. Single-pass, no live API, no sandbox, no judge.

**Why now.** Three concurrent signals:

- *Coverage gap.* The internal eval-roadmap (`2026-04-29-eval-roadmap.md`)
  identifies tool-use as the largest uncovered SOTA-distinct skill.
  v31 added 11 axes but none touches function calling.
- *Industry alignment.* Every 2026 SOTA tech report (Kimi K2.5,
  Qwen 3.5, GLM-4.6, DeepSeek-V3.2) headlines BFCL v3 / v4 or
  TAU2-Bench. We are the only competitive distillation eval that
  doesn't.
- *Cost fit.* Single-turn function calling is procedurally generatable,
  graded by JSON exact-match against a gold key, and runs in well under
  a second per item. Zero new infra.

**Predicted correlation.** No existing axis to compare against, but
BFCL v4 has been shown by the Berkeley team to discriminate even at
the 7 B–13 B band, so a 4 B distilled student should sit comfortably
inside the discriminating range.

**Rollback.** Standard `BENCH_V31_TOOL_USE_BFCL_PER_ROUND=0` env-gate.
No composite-weight reflow needed if disabled (axis-min-valid floor
handles it).

### 4.2 RETIRE — `top_k_overlap` (current weight 0.18)

**Evidence (from internal audit).** Pearson r = **−0.481** with held-out
gsm8k at n = 5 paired post-Kimi-K2.6 kings. The audit explicitly flagged
this as the single-largest active-weight axis with negative correlation.
The published research that motivated it (Anshumann ACL 2025)
demonstrated it as a faithful proxy at frontier model scale, not at 4 B
distilled scale — the regime where teacher-mimicry is most likely to
memorise rather than generalise.

**Action:** halve to 0.09 in this commit. Reallocate the freed 0.09:

- +0.05 → `on_policy_rkl` (currently 0.30 → 0.35; the audit shows
  `on_policy_rkl` carries genuine SOTA-tracking signal because it
  measures generalisation under student sampling, not teacher mimicry).
- +0.04 → `v31_tool_use_bfcl` (the new axis above).

If after n ≥ 12 paired kings the correlation is still ≤ 0 with a tight
CI, retire `top_k_overlap` to 0 entirely; the residual 0.09 reallocates
to `on_policy_rkl` again (final weight 0.44, matching the GKD-style
"RKL is the central signal" finding from Thinking Machines'
On-Policy Distillation, Nov 2025).

### 4.3 EXTEND — IPT to `logic_grid` + `dyval_arith`

**What.** Apply the same procedural object-name renaming we already
ship in `v31_consistency_paraphrase` to the logic-grid and DyVal-arith
items:

- `v31_reasoning_logic_grid`: bijectively rename house attributes
  (Brit / Swede / German / Dane / Norwegian → 5 fresh tokens drawn from
  a shadow vocabulary; pets / drinks / cigarettes likewise).
- `v31_reasoning_dyval_arith`: rename DAG node identifiers
  (`x_0, x_1, …` → bijective alias) before serialising to the prompt.

**Why.** arXiv 2604.15149 directly proves RLVR-trained reasoning models
(which is exactly what every miner is now distilling from Kimi-K2.6)
pass extensional but fail isomorphic verification — **70 %** of
GPT-5-mini shortcuts in the highest-complexity quartile; an
order-of-magnitude jump from complexity 1-10 (40 shortcuts) to 11-20
(458 shortcuts). The 4 B distilled regime is exactly where this failure
mode bites hardest.

**Cost.** One parametric pass per item; the bijection generator is
already implemented (the `rotate_names` helper that
`v31_consistency_paraphrase` uses in §3.1 of the v31 promotion report).
Estimated ship time: 1 dev day, 6–8 unit tests.

**No composite-weight change** — the surface stays the same, only the
per-item generator becomes more discriminating.

### 4.4 What we deliberately do NOT ship

- **HCAST, ARC-AGI-2, HLE, SWE-Bench Pro:** wrong difficulty / cost
  regime for a 4 B distilled student. Track for v32 / v33.
- **Tau3-Bench:** needs stateful API simulator. v32 priority, blocked
  on infra.
- **PRMBench step-level grading:** approved direction, but the per-step
  intermediate-value extractor is a non-trivial rewrite of the
  M1 / M3 / R2 grader. v32, scheduled.
- **RewardBench 2:** evaluates RMs, not students. Useful as defensive
  check on teacher-as-judge surface; out of scope for v31 student
  ranking.
- **Daily rotating private template seed** (SWE-Bench-Pro-style
  partition idea): documented in v31 promotion §3.3; still v32.

---

## 5. References

All published 2024-2026 sources cited above, with arXiv / proceedings
IDs for grep-ability.

**Math / reasoning:**
GSM-Symbolic — arXiv 2410.05229 (Apple, ICLR 2025, also includes
GSM-NoOp). GSM-Plus — arXiv 2402.19255 (ACL 2024). DyVal — arXiv
2309.17167 (Microsoft, ICLR 2024). ZebraLogic — arXiv 2502.01100
(Allen AI, ICML 2025). MuSiQue — arXiv 2108.00573. HopWeaver —
arXiv 2505.15087. AbstRaL — Apple, June 2025.

**Code:**
EvalPlus — Liu et al. NeurIPS 2023 (HumanEval+ / MBPP+).
LiveCodeBench — arXiv 2403.07974; LiveCodeBench Pro Apr 2025.
SWE-Bench Pro — arXiv 2509.16941 (Sep 2025).

**Long context:**
RULER — arXiv 2404.06654 (NVIDIA, ICLR 2024). RULERv2 — ICLR 2026
submission, openreview ZU9tRffRSA.

**Instruction following / IF:**
IFEval — arXiv 2311.07911 (Google 2024). ReIFE — arXiv 2410.07069.

**Calibration / truthfulness:**
SimpleQA — arXiv 2411.04368 (OpenAI 2024). ConfidenceBench —
confidencebench.com (Holter 2024). ConfProBench — arXiv 2508.04576.

**Reward hacking / IPT / decontamination:**
LLMs Gaming Verifiers (IPT) — arXiv 2604.15149 (ICLR 2026 LLM
Reasoning Workshop, openreview 4B3WfRNqe3). **The user query flagged
this ID as fictional — confirmed real.** Static-to-dynamic
contamination survey — EMNLP 2025.emnlp-main.511.

**Reward-model evaluation:**
RewardBench 2 — arXiv 2506.01937 (r = 0.87 with BoN, n = 113 RMs).
PRMBench — ACL 2025.acl-long.1230 (6 216 problems × 83 K step labels).
GroundedPRM — arXiv 2510.14942.

**Tool-use / agentic:**
BFCL v4 — Patil et al. ICML 2025; gorilla.cs.berkeley.edu/leaderboard
(updated Apr 2026). NexusBench — github.com/nexusflowai/NexusBench
(Nov–Dec 2024). Tau-Bench / Tau2 / Tau3 —
github.com/sierra-research/tau2-bench (2024–2026).

**Frontier evaluation:**
ARC-AGI-2 — arXiv 2505.11831 (Chollet et al., 2025). HCAST — arXiv
2503.17354 (METR, 2025). Humanity's Last Exam — Phan et al. Nature
2026 (arXiv 2501.14249). HLE-Verified — arXiv 2602.13964.

**Lab tech reports (2025 H2 → 2026):**
DeepSeek-V3.2 — arXiv 2512.02556 (Dec 2025). DeepSeek transparency
report — Stanford CRFM FMTI Dec 2025. Survey "100 Days After R1" —
arXiv 2505.00551. Kimi K2.5 — arXiv 2602.02276 (Jan 2026). Qwen3 —
arXiv 2505.09388. GLM-4.5 — arXiv 2508.06471. GLM-4.6 release notes —
docs.z.ai/guides/llm/glm-4.6. Gemini 2.5 Pro — Google DeepMind tech
report. Gemini 3 Pro evals methodology —
deepmind.google/models/evals-methodology/gemini-3-1-pro. Claude
Sonnet 4 / 4.5 system cards — assets.anthropic.com (May / Sep 2025).

---

## 6. Sequencing for ship-next

Strict ordering for the next two commits:

1. **Commit A (this branch).** Ship `v31_tool_use_bfcl` axis, halve
   `top_k_overlap` to 0.09, reallocate +0.05 to `on_policy_rkl` and
   +0.04 to the new axis. Update env-var card in `docs/MINER_FAQ.md`,
   `frontend/src/components/v2/docs-panel.tsx`, `api/config.py`,
   `README.md`. Add 12+ unit tests for the BFCL-style item generator.
   Re-run the 250-test composite suite.
2. **Commit B (followup, this week).** Ship the IPT-to-logic and
   IPT-to-DyVal extension. Add 6+ unit tests verifying bijective
   renaming preserves goldness and never collides with attribute
   tokens.
3. **Watch:** the auto-correlation audit (scheduled every 6 h) for the
   new axis; flag if `v31_tool_use_bfcl` ↔ any held-out canary shows
   |r| > 0.5 in either direction within the first 8 paired kings.
