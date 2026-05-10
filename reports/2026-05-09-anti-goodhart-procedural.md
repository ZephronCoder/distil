# Anti-Goodhart procedural benchmarks (v30.8+ design)

**Author:** SN97 ops · **Date:** 2026-05-09 · **Status:** design / partially shipped (v30.7)

## TL;DR

v30.6 plugged the held-out evalscope canary (gsm8k, humaneval, bbh, ifeval,
mmlu_pro) into the composite as 0.50 of the score. That fixed the immediate
anti-correlation symptom (`composite.final r=-0.71` vs canary). But every one
of those benchmarks is **public**, so the moment they're a scoring component
the cheapest miner move is to train *to* the test (paraphrase-synthesis, item-
format match, train-set contamination). Goodhart unmoved.

v30.7 reverts the canary-in-score and redirects the freed 0.50 weight to
**teacher-anchored** axes that are hard to game without the (private)
teacher: `on_policy_rkl` 0.25→0.30, `top_k_overlap` 0.10→0.18, `judge_probe`
0.10→0.20, `long_form_judge` 0.10→0.20, `long_gen_coherence` 0.15→0.25.

That buys correctness but **does not solve** the deeper question the user
posed: *how do we make procedural benchmarks score-able such that optimizing
for the score actually produces a better model?* This doc lays out the design
for v30.8+.

## The fundamental constraint

Goodhart's Law says any metric optimized against ceases to be a good metric.
The corollary for an open subnet: **any rule a miner can read becomes a
training target.** So the goal isn't an "ungame-able metric" (impossible);
it's:

> Make the cheapest path to a high score *be* the underlying capability
> we want — i.e., engineer the proxy and the target to point in the same
> direction by construction.

Three classes of metric, ranked by Goodhart-resistance:

| Class | Example | Gameable by | Goodhart cost |
|---|---|---|---|
| Public benchmark | gsm8k pass_frac | Train on gsm8k. | High. |
| Procedural benchmark, public algorithm | `math_bench` (block-seeded items from a public template + operator pool) | Replicate the generator, train on millions of items, overfit to format. | Medium — generalization helps but format-overfit also wins. |
| Teacher-anchored signal | `top_k_overlap`, `on_policy_rkl`, `judge_probe` | Need real teacher access — not publicly available for the (fine-tuned) subnet teacher. | Low — gaming requires the capability we're measuring. |

v30.7 leans hard on the third class. The remaining work is to harden the
**second class** so adding more procedural diversity doesn't paint us back
into a Goodhart corner.

## Four anti-Goodhart levers for procedural benchmarks

### 1. Per-round private distribution shift (`epoch_secret`)

**Idea.** The procedural item generator is parameterised: operator pool,
parameter ranges, difficulty curves, prompt templates. Today every miner can
read the generator and replicate it. Add a private secret committed at round
start and **revealed post-hoc** (commit-and-reveal, like the existing prompt
cache).

The secret perturbs:
- Which subset of the operator pool is in scope this round.
- Parameter ranges (e.g. integer magnitude × 1–10×).
- Prompt template variants (formal vs informal, with/without examples,
  chain-of-thought required vs implicit).
- Difficulty mix (60/30/10 vs 30/40/30).

Miners can replicate the generator but **not the round's distribution**, so
they have to be robust across distributions. This converts a memorize-the-
generator strategy into "actually distill the teacher" — which is what we
want.

**Implementation sketch.**

```python
# scripts/validator/eval/runtime.py
EPOCH_SECRET = derive_epoch_secret(
    block_seed=current_block,
    private_key=load_private_key(),  # already exists for prompt cache
)
DIFFICULTY_MIX = sample_mix(EPOCH_SECRET[:8])
OPERATOR_SUBSET = sample_subset(OPERATORS, EPOCH_SECRET[8:24])
TEMPLATE_FAMILY = TEMPLATE_FAMILIES[int.from_bytes(EPOCH_SECRET[24:28], "big") % len(TEMPLATE_FAMILIES)]
```

**Reveal.** Post-round, write `EPOCH_SECRET` to `state/eval_data/eval_data_<block>.json`
alongside the per-prompt records, so any miner can verify the distribution
was honestly sampled (not adversarially picked against them).

**Cost.** Implementing well requires:
- A library of operator subsets / template families per skill axis (~3 days
  of work to backfill the existing `BENCH_*` generators).
- Verification harness so the reveal can be cheaply re-checked (~1 day).
- Dashboard tab showing current epoch secret + previous N to reassure
  miners about distribution honesty (~0.5 days).

### 2. Adversarial item selection

**Idea.** Today every UID sees the same procedural items (same `block_seed`).
That's good for paired comparisons but bad for Goodhart: items the king
already nails are wasted scoring opportunities, and items that flunk every
miner are noise.

Replace with **frontier sampling**:

1. Pre-flight pass: sample 4× the eval budget of items, run the *current
   king* + the teacher on them.
2. Keep items where teacher succeeds AND king's pass_frac is in the 0.20–0.80
   band (i.e., items at the edge of king capability).
3. Score every challenger on this filtered set.

Properties:
- Saturated capabilities (already at 1.0 across the board) drop out of
  scoring, so optimizing them gives zero ranking gain.
- Items that *no one* can do drop out, so over-investment in unsolvable
  hard cases gives zero ranking gain.
- The frontier shifts as the king improves. Gaming a frozen frontier
  doesn't help because next round's frontier is somewhere else.

**Implementation sketch.** ~80 lines added to `pod_eval_vllm.py`'s item
sampler; ~30s extra wall-time per round (the pre-flight teacher pass on
the filter pool).

### 3. Hidden axis-weight rotation

**Idea.** v30.7's weights are public. A miner can compute exactly how many
points each axis is worth and budget capability accordingly. If
`long_gen_coherence` is 0.25 and `code_skill_group` is 0.15, focus on
long-coherence training data over code data.

Instead, sample the round's axis weights from a **private prior** committed
ahead of time. The prior is public (`Dirichlet(α)` with publicly known α);
the realization is private until reveal.

Per-round weights vary. Aggregate ranking is averaged across N=10 rounds, so
single-round noise washes out. But **a model that overfits to one axis
loses ranking on the rounds where that axis is sampled low.**

The α prior sets the floor: `α = (3, 3, 3, 1, 1, 1, …)` keeps the top-3
axes always meaningful while occasionally re-sorting which is which.

**Implementation sketch.** Modify `get_effective_axis_weights()` to read
a per-round weight vector from `state/eval_progress.json` written at round
start by the validator service. ~50 lines.

**Risk.** Increases score variance; need to verify the dethrone gates stay
stable. Suggests landing as a **shadow** for 2 weeks before promoting.

### 4. Private OOD probe pool

**Idea.** Maintain a small (50-100 items) private benchmark set we never
publish. Use it ONLY to detect divergence — never as a score component
(symmetric anti-Goodhart: this is exactly the trap we're avoiding by yanking
the canary from the score).

Two readings per round:

- **King's pass_frac on private OOD pool.** If this drops > 5pp below the
  3-round trailing mean while composite climbs > 5pp, fire `goodhart_alarm`
  → king-canary streak gate fires → veto waived for any challenger.
- **Cross-axis correlation.** Like `axis_correlation.py` does today against
  the public canary, but anchored on the private pool. The private pool's
  per-axis correlation is the empirical signal for re-weighting axes.

The private pool's **items rotate quarterly** so even if a miner figures
out which items are in scope, the next quarter's scoring is on a fresh set.

**Implementation sketch.** ~200 lines: pool curation, encrypted on-disk
storage, eval runner, divergence detector, dashboard reveal-on-rotation.

## Procedural design rules (cumulative)

Some of these are already in v30.7. The rest are the v30.8+ work.

| # | Rule | v30.7 status | v30.8+ work |
|---|---|---|---|
| 1 | No public benchmarks in score | ✅ shipped | — |
| 2 | Worst-axis dominates aggregation (α=0.85) | ✅ shipped | — |
| 3 | Saturated sub-axes can't inflate group (`bottom_half_mean`) | ✅ shipped | — |
| 4 | Per-axis baseline-relative penalty (axis < base ⇒ docked) | ✅ shipped | — |
| 5 | Per-round private distribution shift | — | **#1 priority** |
| 6 | Frontier item selection (filter-out saturation + impossible) | — | #2 |
| 7 | Hidden per-round axis weights (smoothed across N=10 rounds) | — | #3 |
| 8 | Private OOD probe pool as divergence detector (NOT score) | — | #4 |
| 9 | Held-out canary as `king_canary_streak` GATE (4pp, 1-round streak) | ✅ tightened | — |
| 10 | Pareto majority dominance (challenger must beat on > 50% of axes) | ✅ shipped | — |

## Why this combination is robust

A miner trying to game v30.8 has to defeat all four:

- **Distribution shift:** can't pre-generate the round's exact items, so
  must train across the operator/parameter envelope.
- **Frontier selection:** training on saturated items gives zero ranking
  gain; must improve on the king's *current* failure modes.
- **Hidden weights:** can't focus on a single axis since other axes might
  be the heaviest this round; must distill broadly.
- **Private OOD:** any divergence between procedural ranking and OOD
  reality fires the dethrone gate.

The cheapest strategy under these rules is: distill the (private) teacher's
distribution well across the entire operator envelope, on adversarially-
selected frontier items, with no axis to over-prioritize.

That **is** what we want: a model that genuinely captures the teacher's
capability distribution, on items the field hasn't yet saturated, robustly
across input formats. It's also exactly what unbounded compute on the
existing distillation losses would converge to — meaning the metric and
the target are aligned by construction.

## Honest caveats

- None of this is a hard guarantee. Goodhart's Law is universal. These
  rules raise the cost of gaming faster than the cost of legitimate
  improvement, but a sufficiently determined miner with enough compute
  can still find seams.
- The gating mechanisms (king-canary streak, Pareto dominance, composite-
  floor veto) compound. A king that survives all of them is unlikely to be
  a pure Goodhart artifact.
- The private OOD pool is the **single point of failure**: if it leaks,
  every layer above it weakens. Treat it like a cryptographic secret:
  rotate quarterly, encrypt at rest, no logs.

## v30.7 vs v30.8+ recap

```
                    v30.6 (shipped + reverted)    v30.7 (LIVE)              v30.8+ (planned)
─────────────────── ────────────────────────────  ────────────────────────  ──────────────────────────────
Canary in score     YES (0.50 weight)             NO (0.0)                  NO
Canary as gate      streak gate (5pp / 2-round)   tightened (4pp / 1)       same as v30.7
Score components    procedural + canary           procedural only           procedural with shifts (#5–7)
Distribution        fixed per-round              fixed per-round            private shift (#5)
Item selection      uniform from skill pool       uniform                   frontier (#6)
Axis weights        public, fixed                 public, fixed             hidden, sampled (#7)
OOD detection       public canary correlation     same                      + private pool (#8)
Goodhart pressure   composite r=-0.71 → ?         alpha=0.85 + bottom-half   + 4 new layers
```

## Action items (this week)

1. **v30.7 ships.** ✅ done.
2. **Implement `epoch_secret`** for the math + code + reasoning generators.
   ~3 days. Land as a shadow (logged but not scoring) for 1 week.
3. **Curate private OOD pool**: 50 items each across math, code, reasoning,
   instruction-following. Hand-written or LLM-generated then human-filtered.
   ~2 days.
4. **Implement frontier item selection** in the eval runner. ~1.5 days.
   Land as a shadow.
5. **Hidden axis weights** as a research spike — 0.5 days to prototype, then
   2 weeks of telemetry before promoting.
6. **Document the new design** in MINER_FAQ.md so miners know the rules
   are being tightened *before* they invest training compute against the
   v30.7 surface.

## Open research questions

- How much of "model is genuinely better" can be captured by teacher-
  anchored axes alone? If `on_policy_rkl + top_k_overlap + judge_probe`
  saturates on a model that's still bad at the held-out canary, we have
  a teacher-quality problem (the teacher itself is leaving capability on
  the table) — distinct from a Goodhart problem on the metric.
- Does the private OOD pool need to grow in proportion to N_kings, or is
  a fixed 50-item pool durable for a year of king turnover?
- The hidden axis-weight rotation introduces noise. What's the right
  smoothing window so the ranking key stays stable but the overfit-resistance
  property holds?
