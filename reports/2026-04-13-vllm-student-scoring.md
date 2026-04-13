# vLLM Student Scoring (`--vllm-student-scoring`)

**Date:** 2026-04-13
**Author:** OpenClaw subagent
**Status:** Complete, not deployed

## Summary

Implemented an experimental vLLM-based student scoring path in `scripts/pod_eval_vllm.py`.

When `--vllm-student-scoring` is enabled, non-king students can now be scored by:
- starting a student vLLM server,
- sending full token-id sequences to `/v1/completions` with `max_tokens=0` and `prompt_logprobs=128`,
- extracting prompt-position top-k student logprobs,
- computing KL against teacher sparse logits,
- falling back to the original HF path if vLLM scoring fails.

Default behavior is unchanged because the new flag is off by default.

## Prior work checked

Searched `reports/` for related work and reviewed:
- `reports/2026-04-12-tokenizer-and-chunked-logits.md`
- `reports/2026-04-04-validator-rewrite.md`

Relevant context used:
- current sparse top-k teacher-logit handling in `pod_eval_vllm.py`
- existing report format in this repo

Note: task requested `REPORT_TEMPLATE.md`, but no such file was present under `/home/openclaw` or this repo, so this report follows the established local report style.

## Files changed

- `scripts/pod_eval_vllm.py`
- `reports/2026-04-13-vllm-student-scoring.md`

## What changed

### 1. vLLM server startup now supports revision pinning

Updated:
- `start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=16384, revision=None)`

Change:
- If a student revision is provided and not `main`, the vLLM server is started with `--revision <hash>`.

Why:
- keeps student scoring aligned with the repo’s existing revision pinning / anti-weight-swap logic.

### 2. Added sparse-vs-sparse KL function

New function:
- `compute_kl_sparse_vs_sparse(...)`

Purpose:
- computes KL when both teacher and student are only available as sparse top-k distributions.

Behavior:
- teacher values are treated as either logits or logprobs depending on caller input,
- student values are treated as logprobs from vLLM,
- KL is computed on the teacher top-k support,
- missing student entries use a floor logprob (`~log(1e-10)`),
- processing is chunked over sequence positions for memory control.

Why:
- existing `compute_kl_from_sparse()` expects dense student logits and cannot be used directly with vLLM prompt top-k output.

### 3. Added prompt-logprob parser for vLLM responses

New function:
- `_parse_vllm_prompt_logprobs(...)`

Handled formats:
- `{token_str: logprob}`
- `{token_id_str: {logprob, rank, decoded_token}}`
- OpenAI-style list entries like `[{token, logprob, ...}, ...]`

Why:
- prompt logprob payload shape can differ from generation-time `top_logprobs`, and may vary across vLLM/OpenAI-compatible response versions.

### 4. Added `score_student_via_vllm()`

New function:
- `score_student_via_vllm(...)`

Behavior:
- starts vLLM for the student on the existing eval port,
- uses lower GPU utilization (`0.15`),
- sends batched token-id prompts to `/v1/completions`,
- requests `max_tokens=0` and `prompt_logprobs=128`,
- aligns prompt logprobs to continuation positions,
- computes per-prompt KL via sparse-vs-sparse KL,
- supports early stopping callbacks,
- reports progress,
- always stops vLLM on exit,
- returns structured status/timing/error info.

### 5. Added CLI flag and integrated Phase 2 path

New arg:
- `--vllm-student-scoring`

Integration:
- enabled only when the flag is set,
- used for non-king students,
- king still follows the original HF path,
- activation fingerprinting is skipped on the vLLM path,
- if vLLM scoring errors, code falls back to the original HF student load + forward scoring path,
- incremental result saving and live progress reporting are preserved,
- early stopping is preserved via a vLLM batch-level callback.

### 6. Token map is prepared for Phase 2 when needed

Added:
- Phase 2 now builds `vllm_token_to_id = _build_token_to_id_map(tokenizer)` when the flag is enabled.

Why:
- prompt logprobs must be mapped back to token IDs to compare against teacher sparse token IDs.

## Validation performed

### Syntax

Passed:
```bash
cd /home/openclaw/distillation
python3 -m py_compile scripts/pod_eval_vllm.py
```

### Quick code sanity checks

Confirmed presence of:
- `compute_kl_sparse_vs_sparse`
- `_parse_vllm_prompt_logprobs`
- `score_student_via_vllm`
- `--vllm-student-scoring`
- Phase 2 integration branch
- vLLM startup revision support

## Important implementation notes

- The student vLLM path currently reuses the existing served model name `"teacher"` because `start_vllm_server()` already hardcodes `--served-model-name teacher`. This is harmless for internal API calls but a little ugly.
- The implementation assumes `/v1/completions` with `max_tokens=0` returns prompt logprobs. If a deployed vLLM build behaves differently, the code will fall back to HF scoring instead of silently producing results.
- Early stopping in the vLLM path runs between batches, not after every individual prompt.
- Batch size is currently hardcoded to `20` inside the new vLLM scoring call site.

## Not done

- No validator restart
- No pod deployment
- No production run
- No live A/B comparison against HF scoring yet
- No attempt to add activation fingerprinting to the vLLM scoring path

## Suggested next test

Run one controlled eval with the flag on and compare against HF scoring on the same prompt set:

```bash
cd /home/openclaw/distillation
python3 -m py_compile scripts/pod_eval_vllm.py
# then run pod_eval_vllm.py once with and once without --vllm-student-scoring
# compare kl_global_avg and per-prompt KLs for a small student set
```

## Replication

```bash
cd /home/openclaw/distillation
python3 -m py_compile scripts/pod_eval_vllm.py
git diff -- scripts/pod_eval_vllm.py
```
