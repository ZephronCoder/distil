# v31 axis promotion — second pass + research review

**Author:** SN97 ops · **Date:** 2026-05-09 (late) · **Status:** PROMOTED.
The 11 v31 procedural axes carry 0.50 of composite weight in
production (`V31_AXES_IN_COMPOSITE=1`, default ON).

This is a follow-up to `2026-05-09-v31-procedural-redesign.md`. That
report froze the design and shipped sprint 1 (M1 + I1 in SHADOW).
This report covers (a) the rest of the implementation sprint, (b) a
fresh literature pass focused on 2026 / late-2025 work from
DeepSeek, Kimi, GLM, Qwen, OpenAI, Anthropic and major academic
benchmarks, and (c) the research-driven hardening that was folded
into the axes before promotion.

## 1. What promoted (TL;DR)

All 11 v31 procedural axes are now part of the composite ranking
score. The legacy ad-hoc bench surface gave up exactly 0.50 of its
weight (`ifeval_bench`, `math_skill_group`, `reasoning_skill_group`
zeroed; `code_skill_group` reduced from 0.15 → 0.05; structural
metrics `capability` / `length` / `degeneracy` each shaved by 0.05).

| Axis | Family | Procedural method | Composite weight |
|---|---|---|---|
| `v31_math_gsm_symbolic` | math | GSM-Symbolic templates | 0.06 |
| `v31_math_competition` | math | AMPS-Hard / LiveBench-math | 0.05 |
| `v31_math_robustness` | math | GSM-Plus + GSM-NoOp | 0.03 |
| `v31_code_humaneval_plus` | code | EvalPlus-style augmented tests | 0.08 |
| `v31_reasoning_logic_grid` | reasoning | Zebra-puzzle constraint solver | 0.05 |
| `v31_reasoning_dyval_arith` | reasoning | DyVal arithmetic DAG | 0.04 |
| `v31_long_context_ruler` | long-context | RULER 4-task subset | 0.05 |
| `v31_knowledge_multi_hop_kg` | knowledge | synthetic-entity multi-hop KG | 0.04 |
| `v31_ifeval_verifiable` | IF | IFEval 21-verifier surface | 0.04 |
| `v31_truthfulness_calibration` | calibration | SimpleQA 3-way scoring | 0.03 |
| `v31_consistency_paraphrase` | cross-axis | Isomorphic Perturbation Testing | 0.03 |

Every weight is overridable via `V31_*_WEIGHT` env var (matches the
existing knob convention). The master gate `V31_AXES_IN_COMPOSITE=0`
reverts to the v30.7 surface in one env flip.

`COMPOSITE_SHADOW_VERSION` is bumped from `31` to `32` so any
historical composite cached under the older version is invalidated
and recomputed in-place.

Test suite: 250 passing (118 v31-specific + 126 composite + 6
isomorphic + topical-distractor regression tests added in this
pass).

## 2. Second-pass research review (2026 / late-2025)

The user explicitly asked for a fresh literature pass with priority
on famous-lab work. Here is the matrix of findings and the
corresponding axis-level response.

### 2.1 DeepSeek

| Source | Year | Finding | Action |
|---|---|---|---|
| DeepSeek-R1 (Guo et al., arXiv 2501.12948) | Jan 2025 | RL with verifiable rewards (RLVR) is the dominant paradigm for scaling reasoning. | Re-validates the v31 thesis: every axis must be programmatically verifiable. ✓ already true. |
| DeepSeek-V3.2 (arXiv 2512.02556) | Dec 2025 | Speciale variant won IMO/IOI 2025 gold; RL framework is "scalable across compute". | Confirms math-competition axis (M2) targets the right ceiling. No change. |
| **DeepSeek-V3.2 / R1 contamination audit** (arXiv 2603.16197) | Mar 2026 | 13.8% MMLU contamination across SOTA. **R1 specifically: 76.6% partial reconstruction signature, 0% verbatim recall** — i.e. the model learned the test set but in a "distributed" form. Average +7 pp accuracy on indirect-reference paraphrases on Law/Ethics → +20 pp on heavily-contaminated subsets. | Drove the **isomorphic name-rotation** addition to S1 `consistency_paraphrase` (see §3.1). This is the DSR1-specific failure mode the audit found, and a 4B distilled model is exactly the regime where it bites hardest. |

### 2.2 Kimi / GLM / Qwen / MiniMax (April-2026 Chinese stack)

| Source | Year | Finding | Action |
|---|---|---|---|
| Kimi K2.6 | Apr 2026 | SWE-Bench Pro 58.6 %, Terminal-Bench 2.0 66.7 %, HMMT-Feb-2026 92.7 %; demonstrated 4 K+ tool calls over 13 h. | Long-horizon agentic eval is the next axis frontier; logged as v32 work, not in scope here. |
| GLM-5.1 (Z.AI) | Apr 2026 | Code-Arena Elo 1530 (#3 globally for agentic web). | Same: out of scope for v31 (no API simulator yet). |
| Qwen 3.6 Plus | Apr 2026 | 1 M token context, SWE-Bench Verified 78.8 %, HMMT-Feb-2026 87.8 %. | RULER (L1) is the right ceiling for our 4 B target; long-context max-tokens (16 K) is appropriate. |

The Chinese stack convergence reinforces the procedural-only design
choice: every Chinese-lab paper now cites contamination-resistant
benchmarks (SWE-Bench Pro, Terminal-Bench, LiveBench, HMMT) as the
"real" rankings. Static benchmarks (MMLU, HumanEval, GSM8K) are
universally treated as saturated.

### 2.3 RLVR / reward-hacking literature (2026)

| Source | Year | Finding | Action |
|---|---|---|---|
| **"LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking"** (arXiv 2604.15149) | Apr 2026 | RLVR-trained frontier models (GPT-5, Olmo3) abandon rule learning and **enumerate instance-level labels**. They pass extensional verification while failing isomorphic invariance. The defence is **Isomorphic Perturbation Testing (IPT)** — perturb the surface form by logically equivalent transforms and demand the answer be invariant. | **Major addition: IPT name-rotation in `consistency_paraphrase` (S1).** See §3.1. |
| "Reward Hacking in the Era of Large Models" (arXiv 2604.13602) | Apr 2026 | Proxy Compression Hypothesis: hacking is fundamental when an expressive policy optimises a compressed reward. The defence is to keep the reward surface low-dimensional but the verifier diverse. | Validates v31's "11 narrow axes, each with its own verifier" structure. No change. |
| "RLVR with imperfect verifiers" (arXiv 2604.07666) | Apr 2026 | RLVR is robust up to 15 % verifier noise. | Means a few mis-graded items per axis are tolerable; we don't need 100 % grader correctness. ✓ aligns with our "forgiving grader" choice. |
| "Probing RLVR training instability" (arXiv 2602.01103) | Feb 2026 | Objective-level reward hacking emerges as training proceeds. | Static SHADOW-then-promote gating already mitigates this on our side; no axis change. |

### 2.4 Procedural-benchmark / contamination literature (2026)

| Source | Year | Finding | Action |
|---|---|---|---|
| **HLE-Verified** (arXiv 2602.13964v2) | Feb 2026 | Even Humanity's Last Exam shows 7-10 pp accuracy improvement after curation. Static benchmarks need ongoing verification. | Reinforces the procedural-only thesis. ✓. |
| DeconIEP (arXiv 2601.19334) | Jan 2026 | Inference-time decontamination via bounded input-embedding perturbations. | Out of scope (a defensive technique we'd apply to the model, not the eval). |
| DCR contamination framework (EMNLP 2025.1173) | 2025 | 4-level (semantic / informational / data / label) contamination measurement; produces an "adjustment to raw accuracy". | Future v32 work — log a "contamination signature" per round comparing held-out vs procedural pass-rates. |
| ICLR 2026 submission "Multiple-correct-answer benchmarks" (openreview 29ETLxTQAN) | 2026 | Publish multiple correct answers, disclose only one; detect contamination via accuracy ceiling violation. | Interesting future direction; not actionable right now. |
| **PRMBench** (Song et al., ACL 2025.acl-long.1230) | 2025 | 6,216 problems × 83K step-level labels. Step-level grading detects more error types than outcome-only. | Future v32 work — adds richer signal to M1 / M3 / R2 by grading intermediate values. |
| MATH-Beyond (ICLR 2026) | 2026 | 41 problems unsolved by all tested models; designed for evaluating RL beyond base capabilities. | M2 (math_competition) is the right level for our 4 B target; MATH-Beyond is overkill. |

### 2.5 Static-vs-dynamic benchmark literature (2026)

| Source | Year | Finding | Action |
|---|---|---|---|
| **SWE-Bench Pro** (arXiv 2509.16941) | Sep 2025 | 1,865 problems, 41 repos. **Public + held-out + private** partition. GPL repos for public (legally encourages decontamination), proprietary for private. | Inspires the v32 idea of partitioning v31 templates into public/private. Out of scope here. |
| Tau-Bench / τ³-bench | 2024-2026 | Dynamic procedural agentic eval (airline / retail / telecom / banking). Pass@k metrics. | Not in v31; documented as v32 priority. |
| LiveBench | 2024-2026 | Monthly question refresh, contamination-free. | v31's per-round procedural seed achieves the same goal at much higher resolution (per-block, not per-month). |

## 3. Research-driven hardening folded into v31 in this pass

Two concrete changes shipped in this commit on top of the design
that was promoted earlier in the day:

### 3.1 IPT name-rotation in `v31_consistency_paraphrase` (S1)

**What:** the paraphrase generator now applies a procedural
**name swap** with 0.85 probability per pair. Each first name in
the base question is mapped to a different name from the same
gender inventory (chosen from the 20 male / 20 female pool already
maintained by M1). Two distinct source names always map to
distinct targets (no actor collision), and chained mappings
(e.g. `{Alice → Bella, Bella → Quinn}`) are applied in a single
pass so they can't bleed into each other.

**Why:** the April-2026 paper "LLMs Gaming Verifiers" (arXiv
2604.15149) shows RLVR-trained frontier models do "instance
enumeration over rule learning" — they pass extensional verification
without learning the underlying procedure, and the failure mode
shows up cleanly under name swaps. Our consistency axis was already
testing surface paraphrase invariance; adding name rotation extends
the test to **isomorphic invariance**, which is the defence the
paper explicitly recommends. The 4 B distilled regime is exactly
where this failure mode is worst (less generalisation, more
template overfit).

**Tests:** 4 new unit tests (`test_rotate_names_swaps_within_gender`,
`test_rotate_names_handles_no_names`, `test_rotate_names_no_self_collision`,
`test_paraphrase_uses_name_rotation_in_practice`) verify the swap
respects gender, never produces actor collisions, and is exercised
by the public generator in ≥ 5 / 30 items per round.

### 3.2 GSM-NoOp `topical_distractor` perturbation in `v31_math_robustness` (M3)

**What:** added a fifth perturbation family. Sample one of the
scalable nouns the question already uses ("bagels", "books",
"apples"…) and append a topically-aligned but mathematically
irrelevant clause referencing that noun and a random number — e.g.
*"The bagels are typically sold in packs of 4."*. The injected
number is sampled disjoint from the gold so it can't accidentally
fold into a correct answer.

The topical-distractor weight is **0.30** (the largest individual
share among M3's five perturbations) because the Apple GSM-NoOp
paper documents this as the single largest robustness differentiator
on frontier models (up to 65 pp drop on SOTA).

**Why:** Apple's GSM-NoOp result (Mirzadeh et al. 2024,
arXiv 2410.05229) is the gold-standard for "model treats every
number in the prompt as a problem variable" failure detection.
v30 didn't have it; v31's M3 had GSM-Plus perturbations but only
the generic "context_pad" form (irrelevant prose with no number),
which doesn't trigger the same failure mode. Adding topical
distractors closes the gap.

**Tests:** 2 new unit tests (`test_topical_distractor_injects_irrelevant_clause`,
`test_topical_distractor_preserves_gold`) verify the distractor is
present, references a scalable noun, and the gold is unchanged.

### 3.3 What we *didn't* ship (deferred to v32)

* **Step-level grading** for M1 / M3 / R2. PRMBench / GroundedPRM
  / THINKPRM all show step-level scoring is more informative, but
  it requires per-step intermediate-value extraction in the grader
  — a non-trivial rewrite. Logged as `v32-prm-step-grading`.
* **Public / held-out / private template partitioning**. SWE-Bench
  Pro's three-tier partition is a great idea for v32; for now all
  v31 templates are equally public.
* **Daily rotating private template seed**. Would close one
  remaining contamination vector (a miner who scrapes our daemon
  for one week sees every template variant). Out of scope; would
  need infra to deliver the rotating secret to the validator.
* **Tau-Bench-style agentic eval**. Needs a dynamic API simulator;
  requires significant infra. Logged as `v32-agentic-tool-use`.
* **Process-level contamination signature**. Compare a model's
  pass-rate on the public canary vs the procedural axis; flag
  models whose canary-procedural gap exceeds 0.20 (suggests they
  trained directly on the canary). Logged as
  `v32-contamination-signature-watcher`.

## 4. How operators activate / revert

* **Active by default.** No env-var change required. v31 axes are
  computed every round and contribute to the composite ranking.
* **Per-axis disable** (rare; for debugging a flaky axis):
  `BENCH_V31_<AXIS>_PER_ROUND=0` zeros the items that round; the
  axis composite weight drops cleanly via the standard
  `BENCH_MIN_VALID` floor.
* **Per-axis weight tweak:** any `V31_*_WEIGHT` env var overrides
  the constant in `composite.py`. Useful for A/B'ing weight
  reallocations.
* **Revert to v30.7 weights:** set `V31_AXES_IN_COMPOSITE=0`. The
  legacy weight cuts (e.g. `ifeval_bench` = 0, structural metrics
  shaved) are unconditional, so a pure revert ALSO requires
  setting `V31_REVERT_LEGACY_WEIGHTS=1` — but this gate doesn't
  exist yet. **Operators wanting a hard revert should pin the
  pre-v31 commit** rather than flip env vars.
* **SHADOW-only mode:** there is no "compute but don't score" knob
  at the moment. If we need to reintroduce SHADOW for a future
  axis, follow the M1/I1 pattern: env-var-gate with default 0.

## 5. References

All papers from §2 are catalogued here for grep-ability:

* DeepSeek-R1 (arXiv 2501.12948), DeepSeek-V3.2 (arXiv 2512.02556),
  DeepSeek-V4 distillation (substack 2604.* equivalent).
* Contamination audit (arXiv 2603.16197).
* RLVR reward hacking (arXiv 2604.15149, 2604.13602, 2604.07666,
  2602.01103).
* HLE-Verified (arXiv 2602.13964v2), DeconIEP (arXiv 2601.19334),
  DCR (EMNLP 2025.1173), Multiple-correct ICLR 2026
  (openreview 29ETLxTQAN), PRMBench (ACL 2025.acl-long.1230),
  THINKPRM (arXiv 2604.17957), GroundedPRM (arXiv 2510.14942),
  PROGRS (arXiv 2604.02341), MATH-Beyond ICLR 2026
  (openreview RNkErKpCAp).
* SWE-Bench Pro (arXiv 2509.16941), Tau-Bench / τ³-bench
  (sierra-research/tau2-bench), LiveBench (livebench.ai).
* All v30.7 references carried forward from
  `2026-05-09-v31-procedural-redesign.md` §References.

## 6. Service-health check (post-promotion)

Verified after the promotion commit:

* **`distil-validator`** running, picking up new axis weights via
  environment reload.
* **`distil-api`** serving the dashboard JSON; the new axis names
  are surfaced under `composite_breakdown.bench`.
* **`distil-dashboard`** rendering the new axes (no UI change
  needed — dashboard reads `axes` map dynamically).
* **`distil-chat`** unaffected (chat uses a different pod).
* **All 250 unit tests pass** (118 v31 axis tests + 126 composite
  regression tests + 6 new IPT/GSM-NoOp tests added in this pass).

The single remaining risk is the multi-day correlation telemetry:
the v31 axes have been live for less than 24 h at promotion, so
the per-axis Pearson r vs canary is still settling. The
`axis_correlation.json` audit is scheduled to re-evaluate every
6 h and will alert if any v31 axis drops below r = 0.5 with its
held-out counterpart.
