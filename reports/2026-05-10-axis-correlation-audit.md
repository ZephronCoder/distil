# Axis Correlation Audit — 2026-05-10

**Status:** Empirical validation of the v31 procedural axis promotion (2026-05-09) and the v31.2 noisy-axis retirement (2026-05-10).

**Inputs:**

- `state/h2h_history.json` — 5 most-recent post-cutover kings.
- `state/benchmarks/uid_*.json` — held-out evalscope canary (gsm8k, humaneval, bbh, ifeval, mmlu_pro).
- `scripts/audit/axis_correlation.py` — Pearson r per axis, joined by king UID.

**Sample:** n = 5 paired kings (post-Kimi cutover only; pre-cutover scores are in a different teacher regime and not comparable). The 95% CIs are wide on n=5 so the magnitudes are advisory; the signs and the relative ordering are the actionable signal.

## Headline

The legacy v30.2 `*_skill_group` axes that we retired from composite weight on 2026-05-10 are **anti-correlated** with the held-out canary in the post-cutover sample. The simpler v28 `*_bench` sub-axes that *constituted* those groups have mixed correlation (some strong positive, some near zero). This is direct empirical evidence that the `bottom_half_mean` aggregation of small-n sub-axes was injecting Goodhart noise into the ranking — exactly what `duohuang` flagged on Discord.

| Axis (legacy, retired or zero-weighted) | Held-out canary | Pearson r | Verdict |
|---|---|---:|---|
| `reasoning_skill_group` | bbh | **−0.836** | Strong Goodhart |
| `long_context_bench` | bbh | **−0.713** | Strong Goodhart |
| `top_k_overlap` (active, w=0.18!) | gsm8k | −0.481 | Worth watching |
| `correction_bench` | humaneval | −0.443 | Goodhart |
| `forking_rkl` | gsm8k | −0.330 | Goodhart |
| `knowledge_skill_group` | mmlu_pro | −0.325 | Goodhart |
| `aime_bench` | gsm8k | −0.245 | Drifting |
| `final` (legacy ranking) | gsm8k | −0.115 | Drifting |
| `worst_3_mean` (legacy ranking) | gsm8k | −0.093 | Drifting |
| `calibration_bench` | ifeval | −0.299 | Drifting |
| `code_skill_group` | humaneval | +0.186 | Weak |
| `math_skill_group` | gsm8k | +0.068 | Weak |
| `mbpp_bench` | humaneval | +0.000 | Weak |
| `debug_bench` | humaneval | +0.000 | Weak |
| `refactor_bench` | humaneval | +0.000 | Weak |

| Axis (legacy, weight > 0 — passing) | Held-out canary | Pearson r | Verdict |
|---|---|---:|---|
| `code_bench` (now telemetry) | humaneval | **+0.895** | Strong proxy |
| `ifeval_bench` (now telemetry) | ifeval | **+0.703** | Strong proxy |
| `robustness_bench` (now telemetry) | gsm8k | **+0.727** | Strong proxy |
| `math_bench` (now telemetry) | gsm8k | **+0.520** | Moderate proxy |

**Interpretation.** The handful of legacy bench axes that are still strong proxies are exactly the ones whose v31 procedural counterparts were promoted (math, code, IFEval, robustness). The Goodhart-drifting groups (`reasoning_skill_group`, `knowledge_skill_group`, `correction_bench`) are exactly the ones that v31 *replaced* with first-principles procedural items. The audit confirms that v31's surface choices match where the legacy proxy was failing.

## v31 procedural axes — pending data

n = 1 in the post-promotion window; the correlation audit will be run again once we have 4+ paired kings (i.e. by 2026-05-14 at the current king turnover rate) and the results re-anchored in `reports/`.

Predicted positive signal based on design:

- `v31_math_gsm_symbolic` ↔ gsm8k — same generative process as the canary's narrative-style word problem, with first-name + numeric variation. Should land near `math_bench`'s +0.52 floor.
- `v31_code_humaneval_plus` ↔ humaneval — direct EvalPlus-augmented test cases. Should track `code_bench`'s +0.895.
- `v31_ifeval_verifiable` ↔ ifeval — same 21-verifier surface. Should track `ifeval_bench`'s +0.70.
- `v31_long_context_ruler` ↔ bbh — RULER NIAH-class items. We expect this to **dramatically improve** over the −0.71 of the legacy `long_context_bench`, which used a too-narrow templating that miners had begun to overfit.

## Untouched concern — `top_k_overlap`

`top_k_overlap` is currently weighted **0.18** in the distillation tier but the audit shows r = −0.481 (n=5) with held-out gsm8k. Two scenarios are consistent with this:

1. The signal is real and `top_k_overlap` is rewarding teacher mimicry that doesn't translate to held-out skill.
2. n = 5 is too small and this is sampling noise; the published research that motivated it (Anshumann ACL 2025) shows it as a faithful proxy at much larger scale.

**Action:** keep the current weight, recompute after we have n ≥ 12 paired kings, and explicitly flag this axis in the next research-pass report. If the correlation stays below 0 with a tight CI, halve the weight to 0.09 and let `on_policy_rkl` (currently 0.30) absorb the freed weight.

## Followups

- Re-run this audit weekly while the post-cutover king history grows.
- Once n ≥ 12, threshold-promote any v31 axis with r ≥ 0.5 to a higher weight band (currently capped at 0.08 for `v31_code_humaneval_plus`).
- Add a CI check that fails the validator boot if any active-weight axis shows a sustained r ≤ −0.3 over the last 8 paired kings (early Goodhart alert).
- The `super_teacher` axis has 0 paired data (inactive in current schema). Decide on retire vs revive in the next research review.
