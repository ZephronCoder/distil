# Discord eval triage — 2026-05-07

Window: 2026-05-06 23:44 UTC → 2026-05-07 04:18 UTC.

Source: recent `#distil-97` / Arbos mention sessions mirrored from the public
SN97 bot.

## Findings

### 1. Eval pod startup error: missing `eval_progress_io.py`

Reported by `duohuang` at 03:21 UTC with a validator log tail showing the pod
bootstrap failed before writing full results.

Status: fixed before this report. The validator now uploads `eval_progress_io.py`
and `eval_prompt_accounting.py` through the pod-runtime manifest, and the live
round launched successfully afterward. At 04:44-04:48 UTC all core services were
active and the eval was advancing through students.

### 2. Content-hash copy check false positive on attention-only LoRA

Reported by `tiny`, `HotShot`, and `Topaz`. The concrete example compared
`Foremost04/hope_king_v1` against `talent-richer/hope_king`: layernorm / MLP /
norm tensors matched, but `model.layers.0.self_attn.q_proj.weight` differed.
The old shard-invariant content hash sampled only the structural tensors, so it
could mark an attention-only merged LoRA as a re-sharded copy.

Status: fixed in code and state.

- `eval/model_checker.py` content hashes are now versioned as `v2:` and require
  at least one attention tensor in the digest.
- The old unversioned content hashes no longer match new submissions.
- Added a regression test that verifies same structural tensors + different
  attention weights produce different content hashes.
- Cleared the validated false DQ for UID 86 (`Foremost04/hope_king_v1`); it now
  shows as queued instead of disqualified.

Unrelated historical content-hash DQs were left untouched because no evidence
was provided that those were attention-only LoRA false positives.

### 3. Tokenizer / validator-load complaints

Reported by `affine@lucker`: whether `tokenizer_config.json` must match Kimi
K2.6, and whether the validator model-load behavior was a bug.

Status: no code bug found. The actual checker requires the teacher tokenizer
artifacts and `tokenizer_config.json` (excluding `chat_template`, which is
checked separately) to match the teacher. This is intentional for Kimi-tokenizer
compatibility. The FAQ now explicitly notes that merged LoRA is allowed and that
copy detection no longer treats attention-only LoRA as a content-hash copy.

### 4. KL dethrone gate complaints

Reported by `A_Tensor`: the 3% KL improvement gate feels too restrictive when
the top KL scores are tightly clustered.

Status: triaged as a policy/scoring decision, not a runtime bug. No emergency
change was made during an active eval round.

### 5. Status / scoring explanation questions

Several messages requested UID status, current king axis tables, worst-score
calculation, weighted score calculation, teacher identity, and training guidance.
These were support/documentation questions rather than system defects.

Status: no code change required.

## Verification

- Live eval progressed from 5/10 to 6/10 students during triage.
- `/api/health`, `/api/eval-progress`, and `/api/queue` agreed on current
  progress and showed no eval failure.
- `pytest -q tests/test_model_checker_content_hash.py` passed.
