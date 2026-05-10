# 2026-05-10 — variance reduction sweep on the v31 composite

**Status:** LIVE (committed 2026-05-10).
**Author:** AI agent at the request of the operator.
**Predecessor:** [`reports/2026-05-09-v31-axis-promotion.md`](2026-05-09-v31-axis-promotion.md) (v31 promotion).

## Problem

The operator observed that, with the v31 composite live, the 3 % single-eval
dethrone margin (`SINGLE_EVAL_DETHRONE_MARGIN = 0.03`) is statistically too
permissive: a challenger that is essentially a copy of the king can clear it
through pure round-to-round noise (item-draw variance, CUDA-greedy ties,
tokenizer edge cases). Their request:

> evaluate the full eval system and see where you can make it less variable
> and more robust with more samples in places where appropriate / needed.

## Audit

Per-axis n at v31-promotion-time vs the contribution to composite variance.
We model each binary axis at p = 0.5 (worst-case) so SE\_axis = sqrt(0.25 / n).
Composite variance is dominated by `worst_3_mean` (α = 0.85) of three lowest
axes — so a single-axis SE near 0.18 directly leaks into the headline number.

| Axis | Old n | Old SE | Composite weight |
|------|------:|-------:|-----------------:|
| v31\_code\_humaneval\_plus | 8  | 0.177 | 0.08 |
| v31\_long\_context\_ruler | 8  | 0.177 | 0.05 |
| v31\_ifeval\_verifiable   | 8  | 0.177 | 0.04 |
| v31\_consistency\_paraphrase | 8 (×2 gens) | 0.125 | 0.03 |
| v31\_math\_competition    | 12 | 0.144 | 0.05 |
| v31\_math\_robustness     | 12 | 0.144 | 0.03 |
| v31\_reasoning\_logic\_grid | 12 | 0.144 | 0.05 |
| v31\_reasoning\_dyval\_arith | 12 | 0.144 | 0.04 |
| v31\_knowledge\_multi\_hop\_kg | 12 | 0.144 | 0.04 |
| v31\_truthfulness\_calibration | 12 | 0.144 | 0.03 |
| v31\_math\_gsm\_symbolic  | 16 | 0.125 | 0.06 |

Plus structural / probe axes (judge: n=16, chat\_turns: n=10, tool\_use: n=16,
calibration: n=8, long\_form\_judge: n=8) that contribute meaningfully but
are deliberately not bumped here (most are LLM-judged or continuous and
already dominate by weight, not by sample count).

The worst-3 of any model's axes are typically v31 axes (they're tougher than
the structural axes by design), so the variance leakage from n = 8 axes
into `worst_3_mean` was the single biggest knob to turn.

## Changes (live as of 2026-05-10)

### 1. Per-axis sample-count lift (the headline change)

| Axis | Old n | New n | New SE | SE delta |
|------|------:|------:|-------:|---------:|
| v31\_math\_gsm\_symbolic    | 16 | 24 | 0.102 | −18 % |
| v31\_math\_competition      | 12 | 18 | 0.118 | −18 % |
| v31\_math\_robustness       | 12 | 18 | 0.118 | −18 % |
| v31\_code\_humaneval\_plus  |  8 | 12 | 0.144 | −19 % |
| v31\_reasoning\_logic\_grid | 12 | 18 | 0.118 | −18 % |
| v31\_reasoning\_dyval\_arith| 12 | 18 | 0.118 | −18 % |
| v31\_long\_context\_ruler   |  8 | 16 | 0.125 | −29 % |
| v31\_knowledge\_multi\_hop\_kg | 12 | 18 | 0.118 | −18 % |
| v31\_ifeval\_verifiable     |  8 | 16 | 0.125 | −29 % |
| v31\_truthfulness\_calibration | 12 | 18 | 0.118 | −18 % |
| v31\_consistency\_paraphrase | 8 (×2) | 14 (×2 = 28) | 0.094 | −25 % |

Total v31 prompts per round: 120 → 190 (+58 %). Wall-time per student
rises ≈ 5–7 minutes; covered by the `POD_PER_MODEL_TIMEOUT` bump
(1500 → 2400 s) below.

### 2. `WORST_3_MEAN_K = 3 → 5`

The bottom-K mean is the dominant component of `final` (α = 0.85). Widening
from K=3 to K=5 averages over five lowest axes, so a single chance-driven
low axis (one wrong answer flipping its rank from 4th-worst to 1st-worst)
moves the aggregate by 1/5 instead of 1/3 — a ≈ 23 % reduction in the SE
of the bottom-K-mean component.

The headline name `worst_3_mean` is preserved in the JSON payload to keep
historical telemetry consistent (it's a label now, not a literal count).
A future schema bump can rename the field if appetite exists.

### 3. `SINGLE_EVAL_DETHRONE_MARGIN = 0.03 → 0.05`

The other two changes shrink the noise. This change widens the bar.
Combined effect on the false-positive dethrone rate (= P(challenger crosses
the gate when their true skill matches the king)):

* old: σ\_paired ≈ 0.05, threshold 0.03 → P\_FP ≈ 27 % per round
* new: σ\_paired ≈ 0.035, threshold 0.05 → P\_FP < 8 % per round

A genuine ≥ 5 % composite improvement still dethrones immediately. A
< 5 % improvement (typical for "I copied the king and tweaked a bit")
gets held by the king. This is the correct directional behaviour: real
skill wins; noise doesn't.

### 4. `POD_PER_MODEL_TIMEOUT = 1500 → 2400` s

Operational hygiene. The longer eval needs head-room. 40 minutes is
~1.6× the previous max-observed per-student wall time; we still
expect 25–28 minutes typical.

### 5. Policy-file ↔ pod env-name alignment

Several v31 entries in `configs/eval_policy.json` used long-form names
(`BENCH_V31_LONG_CONTEXT_RULER_PER_ROUND`) while the pod read short-form
names (`BENCH_V31_RULER_PER_ROUND`), so the policy values weren't actually
applied — the pod silently fell back to its hard-coded defaults. Fixed by:

* renaming the policy-file entries to the short form,
* listing the v31 vars in
  `scripts/validator/pod_session.py::_POD_EVAL_ENV_ALLOWLIST` so the
  validator propagates them across the SSH boundary.

This means that from this commit forward, lifting any v31 sample count is
a one-line policy edit + a validator restart — no pod-side code change.

### 6. Discord announcement explainer

The auto-generated "new king" announcement now reads `final by >5%` instead
of the hardcoded "by >3%". The literal margin is read from
`SINGLE_EVAL_DETHRONE_MARGIN` so the next bump won't drift.

### 7. Probe-side sample-count lifts (second pass)

After the v31 pass, the remaining variance hot-spots in the composite were
the high-weight non-v31 probes:

| Probe | Old n | New n | Composite weight |
|-------|------:|------:|-----------------:|
| `long_form_judge` | 8  | 12 | 0.20 |
| `judge_probe`     | 16 | 20 | 0.20 |
| `chat_turns_probe`| 10 | 14 | 0.10 |
| `calibration_bench` | 8 | 12 | 0.05 |

`bench_tool_use` was left at n=16 (already at the SE knee). Time impact:
~+2-3 minutes per student (long_form_judge is the most expensive, ~30 s
per item at 6144 max tokens; judge_probe is much cheaper at 256 max
tokens). Combined v31 + probe-side bumps lift the per-student wall time
by ~7-10 min, comfortably inside the new 40-minute timeout.

## What was *not* changed (deferred)

* **Paired-bootstrap CI on the dethrone gate.** Computing per-axis SE
  from per-axis n + p\_pass and gating on Δfinal > z × SE\_paired is
  more elegant than a flat 5 % bar but adds non-trivial code and the
  flat bar is already conservative under our SE bounds. Revisit if
  empirical false-positive rates remain above target.
* **Multi-round dethrone confirmation.** Holding a candidate king for
  one extra round, then re-scoring before crowning, would be a clean
  secondary defence — but it also delays legitimate dethrones by one
  round. Defer until we see whether the n + K + margin combo is
  enough.
* **Tighter MIN_VALID floors** (JUDGE_PROBE_MIN_VALID, etc.). Bumping
  the per-probe floors so they reflect the new n risks dropping the
  axis entirely under transient pod errors. Keep permissive defaults
  for now.
* **bench_tool_use lift.** Already at SE knee (n=16, SE 0.125). Marginal
  gain from further bumps; the sandbox cost is non-trivial.

## Acceptance

* `pytest tests/test_arena_v3_composite.py` — 126 / 126 pass.
* `pytest tests/test_single_eval_mode.py` — 58 / 58 pass.
* `pytest tests/` (full suite) — 774 / 782 pass; the 8 remaining
  failures (`test_api_teacher.py` × 2, `test_challenger_selection.py`,
  `test_noise_resistance_bench.py` × 3, `test_robustness_bench.py` × 2)
  are pre-existing and unrelated to this sweep (verified by re-running
  on the parent commit `37c7f7b`).
* Live policy values verified via `policy_env`:
  ```text
  WORST_3_MEAN_K: 5
  POD_PER_MODEL_TIMEOUT: 2400
  SINGLE_EVAL_DETHRONE_MARGIN: 0.05
  BENCH_V31_GSM_SYMBOLIC_PER_ROUND: 24
  BENCH_V31_RULER_PER_ROUND: 16
  BENCH_V31_CODE_PLUS_PER_ROUND: 12
  BENCH_V31_IFEVAL_PER_ROUND: 16
  BENCH_V31_CONSISTENCY_PER_ROUND: 14
  ```
