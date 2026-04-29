# Tweet Drafts — v30.2/v30.3 Eval Overhaul

Six options below, different angles. Each fits in 280 chars. Pick one or remix.

---

## Option 1 — "Better ranker + research-validated signals" (technical, builder-focused)

> SN97 v30.2 is live. New ranking key:
>
> `composite.final = 0.7 × worst_3_mean + 0.3 × weighted`
>
> Smooths single-axis noise (22% of leaderboard sat at exactly worst=0) while keeping anti-Goodhart pressure intact. Plus: top-K overlap, IS-KL, EOPD-style adaptive KL — all research-validated.
>
> Distillation has never been more honest.

**Char count:** ~290 — slightly over. Trim:

> SN97 v30.2 is live.
>
> New ranking: `composite.final = 0.7 × worst_3_mean + 0.3 × weighted`
>
> Smooths the single-axis-min noise (22% of UIDs sat at exactly 0) while keeping anti-Goodhart pressure. Plus research-validated signals: top-K overlap, IS-KL, EOPD adaptive KL.

(~270 chars)

---

## Option 2 — "Beat the teacher, not just match it" (incentive-focused, snappy)

> Pure distillation has a ceiling: the teacher.
>
> SN97 v30.2 ships `super_teacher` — a new axis that explicitly rewards student > teacher on 16 verifiable benches.
>
> Match teacher → 0.0
> Beat by +0.20 → ~0.96
>
> The path to SOTA-class small models is GRPO + post-distillation SFT. Mine accordingly.

**Char count:** ~290 — trim to:

> Pure distillation has a ceiling: the teacher.
>
> SN97 v30.2 ships `super_teacher` — explicitly rewards student > teacher on 16 verifiable benches. Match teacher = 0.0. Beat by +0.20 ≈ 0.96.
>
> Path to SOTA-class small models = GRPO + post-distillation SFT.

(~250 chars)

---

## Option 3 — "Goodhart's law, fought" (the why — academic-leaning, cites research)

> The 'optimize the metric, not the goal' problem is real.
>
> Pre-v30 SN97 saw Pearson r = -0.665 between validator math_bench and held-out GSM8K. Kings climbed validator while regressing on real benchmarks.
>
> v30.2 fixes this with skill groups, super-teacher, and 6 new research-validated signals.

**Char count:** ~285 — trim:

> Goodhart's law, in the wild: pre-v30 SN97 saw Pearson r = -0.665 between validator `math_bench` and held-out GSM8K.
>
> Kings climbed validator while REGRESSING on real benchmarks.
>
> v30.2 fixes it: skill groups, super-teacher axis, 6 research-validated distillation signals.

(~280 chars)

---

## Option 4 — "Single tweet, full v30.2 highlights" (broadcast-style)

> SN97 v30.2 deployed:
>
> ✓ New ranking: `final = 0.7 × worst_3_mean + 0.3 × weighted`
> ✓ super_teacher axis: rewards beating teacher
> ✓ Skill groups (code/math/reasoning/knowledge)
> ✓ 6 research-validated shadow signals
> ✓ +50% bench items, king re-eval per round
>
> Mining Guide v2 in the repo.

**Char count:** ~265

---

## Option 5 — "Quote-tweet style, hooks the reader" (storyteller)

> Pre-v30: 22% of SN97 miners sat at composite.worst = 0. Single-axis-min was floored by noise on the lowest-data axis. Ranking was effectively random in that cluster.
>
> v30.2 ships `composite.final = 0.7 × mean(bottom 3) + 0.3 × weighted`. The cluster fragments. Real progress wins.

**Char count:** ~290 — trim:

> Pre-v30: 22% of SN97 miners sat at composite.worst = 0. Single-axis min was floored by lowest-data noise — ranking was effectively random in that cluster.
>
> v30.2 ships `final = 0.7 × mean(bottom 3) + 0.3 × weighted`. Real progress wins again.

(~265 chars)

---

## Option 6 — "Research-paper namedrop" (technical credibility)

> SN97 v30.2 ships 6 new axes informed by 2026 research:
>
> • `top_k_overlap` (Rethinking OPD)
> • `kl_is` (Anshumann ACL 2025, unbiased KL)
> • `forking_rkl` (Wang et al. 2025)
> • `entropy_aware_kl` (EOPD, +1.37–5.05 Pass@8)
> • `tail_decoupled_kl` (TAD)
> • `super_teacher` (capacity-gap aware)

**Char count:** ~290 — trim:

> SN97 v30.2 ships 6 axes from 2026 distillation research:
>
> • `top_k_overlap` — Rethinking OPD's #1 predictor
> • `kl_is` — Anshumann ACL 2025 unbiased KL
> • `entropy_aware_kl` — EOPD adaptive blend
> • `tail_decoupled_kl` — TAD
> • `super_teacher`
> • `forking_rkl`

(~270 chars)

---

## My recommendation

**Option 4** for max info density and broad appeal — uses checkmarks for skimmability, mentions Mining Guide v2 for engagement.

**Option 2** for max engagement-per-character — the "ceiling" hook is sticky and the formula does the explaining.

**Option 3** for technical credibility — the negative-correlation number is concrete and grabs attention from people who know the literature.

If you want a thread instead of a single tweet, I can chain: 1 hook tweet → 1 ranking-key tweet → 1 super-teacher tweet → 1 research-axes tweet → 1 dashboard-link tweet. Just say the word.
