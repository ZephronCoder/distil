# Prompt Reproducibility + Round Verification + Test Improvements

**Date:** 2026-04-02
**Branch:** `improvements/validator-fixes-v2`
**Author:** subagent (reproducibility task)

## Summary

Created tools for reproducing evaluation prompts from any past round, independently verifying rounds, and expanded test coverage for KL divergence, scoring, model checking, and state consistency.

## Files Created/Modified

### New Files
1. **`scripts/reproduce_prompts.py`** — Regenerate exact eval prompts from a block number + block hash
2. **`scripts/verify_round.py`** — Independently verify any past evaluation round
3. **`tests/test_kl_scoring_edge_cases.py`** — 53 unit tests for KL, scoring, model checker, and edge cases

### Existing Files (unchanged)
- `tests/test_eval_pipeline.py` — Left as-is (integration/E2E tests); new unit tests are in a separate file

## Details

### 1. `scripts/reproduce_prompts.py`

Replicates the exact shard selection and prompt sampling logic from `eval/dataset.py`:
- Takes `--block-number` and `--block-hash` as inputs
- Computes `shard_idx = int(hash_hex[:8], 16) % 6542` (same as dataset.py)
- Uses `random.Random(hash_hex)` to shuffle shard indices deterministically
- Filters by min/max char length (200/4000 defaults)
- Can fetch block hash from substrate RPC (`--substrate-url`) or fall back to sha256
- Outputs JSON to stdout or file (`--output`)
- Includes `--metadata` flag for shard info

### 2. `scripts/verify_round.py`

4-step verification pipeline:
1. **Prompt regeneration** — Regenerates prompts and verifies count matches
2. **Metadata validation** — Checks king is in results, KL scores are reasonable, epsilon threshold is correct
3. **Scoring logic** — Verifies winner-take-all with epsilon is consistent
4. **Local re-evaluation** (optional, `--rerun`) — Guidance for re-running with GPU

Fetches H2H data from API (`/api/h2h-latest`, `/api/h2h-history`).

### 3. `tests/test_kl_scoring_edge_cases.py`

53 tests across 6 test classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestKLDivergence` | 8 | Identical distributions (KL≈0), different (KL>0), start_pos, 2D input, known values, asymmetry, top-k fallback, zero gen_len |
| `TestEarlyStopping` | 8 | Clearly worse, close scores, KL=0, negative KL, very high KL, single score, zero variance, fraud threshold |
| `TestScoring` | 11 | Winner-take-all, max_kl filter, zero KL excluded, stale failures, no candidates, single candidate, UID expansion, DQ exclusion, EMA update/first, failure tracking |
| `TestModelChecker` | 9 | MoE vs dense params, nested config, empty config, DQ by hotkey:block, legacy UID DQ, coldkey/HF flagging, duplicate hash, commitment change |
| `TestPromptSampling` | 7 | Shard determinism, distribution, hash computation, seeded sampling, format_prompt sanitization, binary rejection |
| `TestStateConsistency` | 6 | Score round-trip, empty defaults, corrupted JSON, history capping, per-commitment DQ, commitment cache |

All 53 tests pass in ~3 seconds (CPU only, no downloads needed).

## How to Use

```bash
# Reproduce prompts from a known round
python scripts/reproduce_prompts.py --block-number 7879382 \
    --substrate-url wss://entrypoint-finney.opentensor.ai:443

# Verify a round
python scripts/verify_round.py --round-block 7880403 \
    --api-url https://api.arbos.life \
    --substrate-url wss://entrypoint-finney.opentensor.ai:443

# Run unit tests
python3 -m pytest tests/test_kl_scoring_edge_cases.py -v
```

## Test Results

```
53 passed, 1 warning in 2.95s
```

The single warning is a harmless PyTorch std() degrees-of-freedom warning when computing std on a single-element tensor.
