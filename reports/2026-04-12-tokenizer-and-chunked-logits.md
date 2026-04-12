# Strict Tokenizer Enforcement + Chunked Logit Extraction

**Date:** 2026-04-12
**Files changed:**
- `eval/model_checker.py`
- `scripts/pod_eval_vllm.py`
- `scripts/remote_validator.py`

## Problem

1. **Tokenizer bypass:** The tokenizer encoding check in `model_checker.py` had a fail-open `except` block — if `verify_tokenizer_match()` threw any exception, the model was allowed through. This let models with corrupted/modified tokenizers (e.g. `sniper918/sn97-xxxn` with wrong `tokenizer_class`, extra `added_tokens`, different `tokenizer.json` hash) pass validation.

2. **Logit truncation gaming:** `pod_eval_vllm.py` Phase 1b truncated long sequences to `max_logit_len` (2048 tokens). Attackers could exploit this by making models that only need to match the first 2048 tokens of teacher output.

## Changes

### model_checker.py — Strict tokenizer enforcement

1. **New function `verify_tokenizer_files()`**: Does byte-for-byte SHA256 comparison of `tokenizer.json` between student and teacher (`TEACHER_MODEL`). Also compares `tokenizer_config.json` after removing `chat_template` (which is checked separately). Any mismatch = immediate DQ with clear reason.

2. **New helper `_is_transient_error()`**: Identifies transient network errors (429, timeout, connection) that should not DQ a model.

3. **Fail-closed exception handling**: Both `verify_tokenizer_files()` and `verify_tokenizer_match()` now fail-closed on exceptions. Only transient network errors are allowed through. All other errors (corrupted files, parse failures, etc.) result in DQ.

4. **Call order**: `verify_tokenizer_files()` runs BEFORE the encoding-based check (step 7a vs 7b). If files don't match, the model is DQ'd immediately without needing to download and instantiate the tokenizer.

### pod_eval_vllm.py — Chunked forward pass

1. **Replaced truncation with chunked KV-cache forward pass**: Sequences longer than `HF_CHUNK_SIZE` (4096 tokens) are processed in chunks using `past_key_values` to maintain KV cache between chunks. All logit chunks are concatenated to produce full-sequence logits.

2. **Added `HF_CHUNK_SIZE = 4096` constant** at top of file.

3. **`--max-logit-len` arg kept for backward compat** but deprecated and ignored.

4. **Logging**: Reports number of chunked sequences when chunking is used.

### remote_validator.py — Cleanup

1. Removed `MAX_LOGIT_LEN` constant definition (line 61).
2. Removed `--max-logit-len` from the eval command construction (line 726).

## NOT done

- No services restarted
- No uploads to eval pod
- Changes take effect on NEXT eval cycle only

## Replication

```bash
cd /home/openclaw/distillation
git log -1 --oneline  # verify commit
python3 -c "import ast; ast.parse(open('eval/model_checker.py').read())"
python3 -c "import ast; ast.parse(open('scripts/pod_eval_vllm.py').read())"
python3 -c "import ast; ast.parse(open('scripts/remote_validator.py').read())"
```
