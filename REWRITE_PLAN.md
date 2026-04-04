# Distil SN97 Validator Rewrite Plan

## Current State
- `remote_validator.py`: 2261 lines, one massive function
- `pod_eval_vllm.py`: 1139 lines, one massive main()
- `eval/scoring.py`: 326 lines (OK but has dead code)
- `eval/model_checker.py`: 620 lines (OK but can be trimmed)
- `eval/dataset.py`: 287 lines (OK)
- `eval/kl_divergence.py`: 319 lines (OK but has legacy functions)
- `api/server.py`: 1488 lines (separate concern, leave for now)

## Problems Identified

### Architecture
1. **`remote_validator.py` is a 2261-line god function** — main() is ~2000 lines with no decomposition
2. **State management is ad-hoc** — 12+ JSON files read/written with manual Path juggling everywhere
3. **No separation of concerns** — chain interaction, pod management, eval orchestration, scoring, leaderboard management, announcements all in one function
4. **Duplicated code** — JSON read/write patterns, retry loops, disk cleanup commands appear 5+ times

### Dead/Unused Code
1. `update_ema()` in scoring.py — EMA was replaced by winner-take-all, but function still exists
2. `load_ema_scores/save_ema_scores` aliases — dead
3. `commitment_changed()`, `load_commitment_cache()`, `save_commitment_cache()` — unused
4. `compute_winner_weights()` — still exists but main code builds weights manually
5. `compute_kl_divergence()` (CPU fallback) in kl_divergence.py — labeled "simulation/testing only"
6. `evaluate_kl_with_continuation()` in kl_divergence.py — "legacy, kept for backward compat"
7. `load_prompts_from_hf()` and `sample_prompts_seeded()` in dataset.py — legacy functions
8. `verify_tokenizer()` in model_checker.py — replaced by `verify_tokenizer_match()`
9. Commented-out cumulative dethronement code in remote_validator.py
10. Multiple scripts that may be stale: `cosine_similarity_check.py`, `multi_shard_analysis.py`

### Fragility
1. **No disk cleanup before teacher load** (FIXED in fd31b6f but structurally messy)
2. **SSH failures crash the whole loop** — single exception in lium.exec kills the epoch
3. **State files can get inconsistent** — the 40-line `validate_state_consistency()` is a band-aid
4. **Pod reconnection is brittle** — pod is found once at startup, never re-discovered
5. **Teacher model name hardcoded in 3+ places** across both files
6. **No structured logging** — mix of print() and logger.info() with inconsistent formatting
7. **Parallel eval path is complex and rarely used** — 8-GPU path adds 100+ lines of complexity for a marginal case

### Community Feedback (caseus + others)
1. Disk cleanup before teacher loading ✅ (fixed)
2. Add logging through the API so status is visible without SSH
3. Make the system reproducible — prompts, scores, and eval should be auditable
4. Reduce crash-loop cycles — fail fast, clean up, retry intelligently

## Rewrite Plan

### New File Structure
```
scripts/
  remote_validator.py    — Main orchestrator (slimmed to ~800 lines)
  pod_eval_vllm.py       — GPU eval script (slimmed to ~700 lines)

eval/
  __init__.py
  dataset.py             — Prompt sampling (keep, trim legacy)
  kl_divergence.py       — KL computation (keep, trim legacy)
  model_checker.py       — Architecture/integrity checks (keep, trim)
  scoring.py             — Score/state management (keep, trim dead code)
  state.py               — NEW: Centralized state management class
  pod.py                 — NEW: Pod lifecycle (connect, cleanup, health)
  chain.py               — NEW: Chain interaction (metagraph, commitments, weights)
```

### Key Changes

#### 1. remote_validator.py — Break up the god function
- Extract `class ValidatorState` → `eval/state.py` (all 12+ JSON files, atomic writes, validation)
- Extract pod management → `eval/pod.py` (connect, reconnect, cleanup, health check, upload/download)
- Extract chain interaction → `eval/chain.py` (metagraph fetch, commitment parsing, weight setting)
- Main loop becomes: fetch_chain → precheck_models → select_challengers → run_eval → process_results → set_weights
- Each step is a standalone function that can be tested independently

#### 2. pod_eval_vllm.py — Simplify
- Remove parallel multi-GPU path (we use 1×B200, not worth the complexity)
- Consolidate disk cleanup into one `ensure_disk_space()` called at every phase boundary
- Remove the --persistent-vllm path (we don't use it)
- Simplify the teacher loading: try vLLM → fallback HF, both with disk checks
- Remove dead args and unused code paths

#### 3. eval/scoring.py — Trim dead code
- Remove `update_ema()`, `load_ema_scores`, `save_ema_scores`
- Remove `commitment_changed()`, `load_commitment_cache()`, `save_commitment_cache()`
- Remove `compute_winner_weights()` (weights are built directly in validator)

#### 4. eval/kl_divergence.py — Trim legacy
- Remove `compute_kl_divergence()` (CPU fallback)
- Remove `evaluate_kl_with_continuation()` (legacy)
- Keep: `compute_kl_from_logits()`, `generate_teacher_continuations()`, `evaluate_student_kl()`

#### 5. eval/dataset.py — Trim legacy
- Remove `load_prompts_from_hf()` (legacy HF streaming)
- Remove `sample_prompts_seeded()` (legacy)
- Keep: `sample_prompts_from_dataset()`, `format_prompt()`

#### 6. eval/model_checker.py — Trim
- Remove `verify_tokenizer()` (replaced by `verify_tokenizer_match()`)

### What NOT to change
- `api/server.py` — separate concern, works, leave it
- Core KL computation logic — proven correct
- Paired t-test dethronement — working correctly
- Block-hash-seeded prompt sampling — anti-gaming mechanism
- Winner-take-all weight setting — subnet design

### Testing Strategy
- Run with `--once` flag to test single epoch
- Compare output against current validator's behavior
- Verify state files are compatible (same format)
