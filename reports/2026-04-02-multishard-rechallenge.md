# Multi-Shard Analysis Tool + Periodic Re-challenge + Prompt Count Bump

**Date:** 2026-04-02  
**Branch:** `improvements/validator-fixes-v2`  
**Repo:** `/home/openclaw/distillation`

## Summary

Implemented four changes requested via Discord:

1. **Multi-shard analysis script** (`scripts/multi_shard_analysis.py`) — BC's request for paired statistical benchmarking of all past/current kings
2. **Periodic re-challenge system** (in `scripts/remote_validator.py`) — s0wa48's observation that models are scored once and never re-evaluated
3. **Prompt count increase** 60 → 120 — Arbos's suggestion to reduce H2H variance
4. **Dashboard score clarity notes** (`scripts/dashboard_notes.md`) — Documenting global vs H2H score distinction for s0wa48's confusion

## Changes

### 1. `scripts/multi_shard_analysis.py` (NEW)

Standalone script for multi-shard paired KL analysis. Features:
- Evaluates N models across the same M shards from `karpathy/climbmix-400b-shuffle`
- Uses the **same KL computation** as the validator (`eval/kl_divergence.py`)
- Computes per-shard KL, mean, std dev, 95% bootstrap CI
- Paired t-test between all model pairs (same shards = valid paired comparison)
- Outputs JSON + markdown table
- Resume support for long runs (`--resume`)
- Deterministic shard selection via seed

**Usage:**
```bash
python scripts/multi_shard_analysis.py \
    --models "user/model-a,user/model-b" \
    --num-shards 50 \
    --prompts-per-shard 40 \
    --teacher Qwen/Qwen3.5-35B-A3B \
    --output results/
```

### 2. Periodic Re-challenge (in `scripts/remote_validator.py`)

- New constants: `RE_CHALLENGE_INTERVAL = 30` epochs, `RE_CHALLENGE_TOP_N = 3`
- Every 30 epochs, selects the top 3 historically best models (by `best_kl` from `model_score_history`) excluding the current king
- Adds them to the challenger pool for re-evaluation on fresh prompts
- Tracks history in `state/rechallenge_history.json`
- Logs prominently with 🔄 emoji
- If a re-challenged model beats the king, normal dethronement flow applies

### 3. Prompt Count: 60 → 120

- Changed `EVAL_PROMPTS` from 60 to 120 in `scripts/remote_validator.py`
- Halves the variance of H2H comparisons
- Comment documents the rationale (Arbos quote)

### 4. `scripts/dashboard_notes.md` (NEW)

- No dashboard HTML/JS files exist in the repo yet
- Created documentation explaining global vs H2H score distinction
- Includes UI implementation guidelines for when the dashboard is built
- Explains why cross-shard scores are not comparable (~10-20% variance from prompt difficulty)

## Files Modified/Created

| File | Action |
|------|--------|
| `scripts/multi_shard_analysis.py` | Created |
| `scripts/dashboard_notes.md` | Created |
| `scripts/remote_validator.py` | Modified (EVAL_PROMPTS, RE_CHALLENGE constants, re-challenge logic) |
| `reports/2026-04-02-multishard-rechallenge.md` | Created (this file) |

## Replication

```bash
cd /home/openclaw/distillation
git checkout improvements/validator-fixes-v2
# View changes
git diff HEAD
# Test multi-shard script (dry run, needs GPU)
python scripts/multi_shard_analysis.py --help
```
