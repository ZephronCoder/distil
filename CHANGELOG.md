# Changelog

All notable changes to Distil (SN97) are documented here.

## [2026-04-13]
### Added
- Concurrent teacher generation (4x parallel vLLM requests) for faster eval rounds
- vLLM-based student scoring (opt-in, `--vllm-student-scoring` flag)
- Pre-dethronement integrity check — models must be public on HuggingFace before crowning
- Shard-invariant hashing to detect re-sharded model copies
- API transparency endpoints: `/api/evaluated_uids`, `/api/dq_reasons`, `/api/model_hashes`
- `/api/compare` endpoint for side-by-side miner comparison
- `/api/eval-stats` for round timing and KL trends
- Reference baseline (Qwen3.5-4B) on dashboard leaderboard
- `MIN_COMPLETION_TOKENS=10` filter for teacher completions
- Round timing data (`elapsed_seconds`, `n_students`) in H2H records
- `TOP_N_ALWAYS_INCLUDE` increased from 2 to 5
- Git commit hash logged at validator startup

### Fixed
- UID 237 exploit: model deleted after crowning, now caught by integrity check
- Dashboard p-value display (was hardcoded, now uses `paired_test_alpha` from round data)
- Chat pod auto-restart (was using nonexistent script)
- Stale eval process kill before new rounds
- Failure counter reset on new commitment
- Leaderboard API enrichment from live state

### Changed
- `PAIRED_TEST_ALPHA`: 0.02 → 0.03
- `EVAL_PROMPTS_H2H`: 150 → 300

## [2026-04-12]
### Added
- Include base model (Qwen3.5-35B-A3B) as reference in every eval round
- vLLM generation progress reported to `eval_progress.json` for dashboard
- Eliminate Phase 1b via vLLM logprobs — top-k sparse storage, HF batching, reasoning parser
- GPU splitting on 2×RTX4090 — rent 1 of 2 GPUs at $0.14/hr
- Auto-benchmarks on king change + `/api/benchmarks` endpoint
- `/api/pod-logs` and `/api/eval-data` endpoints
- Separate chat/bench pod from eval pod
- Extract t-test p-value to round level in H2H history API
- `max_new_tokens` raised to 8192 (effectively uncapped generation)

### Fixed
- Socket timeouts and reconnect logic for Lium connections
- Handle empty sparse logprobs in scoring (0-token generations)
- vLLM `gpu_memory_utilization` 0.45 → 0.90 (KV cache was going negative)
- vLLM `max_model_len` 4096 → 16384 for uncapped completions
- Cap HF logit extraction at 2048 tokens to prevent Phase 1b timeout
- Chunked logit extraction + strict tokenizer enforcement
- Announcement uses live params and prod API for earnings
- Stop chat vLLM before teacher gen to prevent HF fallback
- Annotate exploit-era rounds in API + dashboard

### Changed
- Remove `min_chars=500` pre-filter — all prompts now evaluated
- 150 prompts with p<0.02 threshold (later raised to 300 on Apr 13)
- Contenders reduced from top-4 to top-1 per round

## [2026-04-11]
### Added
- Revision-based integrity checks — pin exact model revisions during eval
- Chat link on dashboard for trying the king model

### Fixed
- Integrity DQs stay permanent — cheaters must pay new submission fee
- Continuous integrity checks + tighten p-value to 0.01
- King-failed promotion must use fresh scores only
- Unbound variable in `model_checker`

### Changed
- Published benchmarks paper v2.1 — deeper analysis, full datasets, H200 results

## [2026-04-10]
### Fixed
- rsync filter order — exclude `announcement.json` before include `*.json`

## [2026-04-09]
### Added
- R2 eval data export for external analysis
- `check_model.py` now matches production decoding exactly

### Fixed
- Exclude `announcement.json` from rsync + write to prod directly
- Remove HF username flags from scoring code
- Prevent duplicate announcement overwrites

## [2026-04-08]
### Changed
- Revert eval prompts from 480 to 300 — community flagged slow rounds
- Prompt pre-filtering with `min_chars` 200→500, oversample 480→~300 usable
- Restore `--revisions` flag for pod eval

## [2026-04-07]
### Added
- Increase eval prompts from 180 to 300 for stronger t-test
- Architecture enforcement: require `Qwen3_5ForConditionalGeneration`
- `check_model.py` for miner self-verification of architecture

### Fixed
- Eval timeout set to 2 hours flat
- Re-enable top-4 contenders in every eval round
- Disable re-evals — models evaluated once per round
- Copy base tokenizer files in chat server bootstrapper

## [2026-04-06]
### Added
- T-test column on dashboard + hotkey tracking in score history
- One-sided t-test + Bonferroni correction (later reverted — too conservative)
- Token-safe truncation + oversample prompts
- `block_hash` in H2H history for auditability
- Activation-space duplicate detection (caseus's proposal)
- vLLM king chat bootstrap + fresh commitments cache logic
- Coldkey-level and HF username hard bans in precheck (later reverted to per-hotkey only)
- Keep prod commitments fresh + surface vLLM compatibility

### Fixed
- UID recycling — legacy hash no longer blocks new miners
- Pin model revisions on pod eval + DQ UID 147
- Dethrone king if it fails to produce a fresh score
- Disable thinking by default for OpenAI chat proxy
- Stale DQ auto-cleanup

## [2026-04-05]
### Fixed
- Models tested against old kings were never re-selected for H2H
- Validator logs API returns chronological order (oldest first)
- Full state reset via `--resume` removal, revert false dethronement
- Clean orphaned UIDs from `evaluated_uids` at epoch start
- Reset model hash when miners re-commit at new block
- Silence paramiko SSH logs drowning validator output
- Stale `eval_results.json` never cleared between rounds + chat model mismatch detection

## [2026-04-04]
### Added
- OpenAI-compatible `/v1/` endpoints for Open WebUI integration
- Auto-restart chat server when king changes
- Dashboard improvements: validator logs, leaderboard, miner rounds endpoint
- Real-time validator logging API endpoint
- Raise prompt `max_chars` from 4000 to 10000 for richer eval signal

### Fixed
- Replace lium SDK with direct SSH for chat proxy (more reliable)
- Remove `max_tokens` cap, sanitize chat error messages
- `check_model.py` matches production, remove unnecessary truncation
- Aggressive disk cleanup before teacher loading
- Increase vLLM startup timeout to 15 min for fresh pods
- `.total_mem` → `.total_memory` (torch API change)

### Changed
- Full validator rewrite — modular architecture, dead code removal, fault tolerance

## [2026-04-03]
### Added
- Hard cap on challengers per round (15 maintenance, 80 initial)
- `/api/eval-status` endpoint for miner eval visibility
- Deploy script for production frontend

### Fixed
- Leaderboard uses H2H scores (not global), top-k shadow metrics
- Sanitize `inf`/`nan` in API responses (UID 41 Infinity crash)
- Atomic state writes and safe state sync
- Prevent re-evaluation of models already in `evaluated_uids` but missing score
- Top-5 leaderboard uses scores from previous H2H round only
- Corrupt disk cache JSON handling
- Skip eval when no new challengers
- Stale bare-UID DQ entries blocking new miners
- Validator reads oldest commitment (permanent, one per hotkey)
- H2H winner selection used stale global scores instead of round scores
- Announcement spam — claims log prevents rsync re-posting
- Commitments API + new miner/commitment endpoints

### Changed
- Chunked compiled KL divergence (2.4× faster, 10× less memory)
- B200 eval support — enforce-eager, grouped_mm patch, flash-attn
- Disable teacher cache save when disk full on 230GB pods
- Atomic cache save to prevent corrupt `teacher_cache.pt`
- Stop vLLM before HF teacher logit extraction (OOM)

## [2026-04-02]
### Added
- Paired t-test dethronement replaces fixed epsilon threshold
- Top-5 contenders always included in every eval round
- Asymmetric cumulative dethronement (later removed in favor of t-test)
- `test_miner.py` pre-submission validator for miners
- Leaderboard API with stale detection and model existence checks
- `@distil・97` role ping on king changes
- Full eval mode — all models in one round on same prompts
- Smart challenger selection replacing periodic re-challenge
- Min-token filter + cosine similarity tool
- Batch 15 models per round during initial eval

### Fixed
- VRAM OOM with chunked processing
- EVAL_PROMPTS reference mismatch
- UID 18 infinite re-eval loop

### Changed
- Enforce original Qwen chat template (security)
- API hardening: rate limiting, chat injection fix
- Merge caseus's training script
- Prompt reproducibility + round verification

## [2026-03-31]
### Added
- vLLM-accelerated teacher generation for faster + reproducible eval
- Persistent vLLM teacher — keep server alive between rounds
- Chat with king model via GPU pod (streaming, thinking separation)
- Automated benchmark script (king vs baseline via Vast.ai)
- Validator status card — shows current state, phase, and queue
- Live GPU log streaming to dashboard
- Comprehensive pre-submission model checker (`check_model.py`)
- Round resumption — reuse prompts + partial results after crash
- Persistent model score history — never re-eval known-bad models
- Multi-layer fraud tracking (coldkey + HF username flagging)
- 7-layer comprehensive anti-cheat system
- Per-model timeout (10 min) replaces fixed global timeout
- Subprocess probe load to catch model segfaults gracefully
- Resume eval from partial results + cached teacher logits
- 5-minute timeout for student model loading
- Health endpoint shows king, scores count, eval progress, last eval block
- Published paper: "Does Lower KL Divergence Produce Better Models?"

### Fixed
- **CRITICAL**: VRAM fraud check measured total GPU, not student delta
- **CRITICAL**: scoring now checks DQ list — DQ'd UIDs never get weight
- Block trust_remote_code exploit + DQ UID 210
- Weight assignment uses H2H winner only, not stale global scores
- King selection from `h2h_latest`, not global scores
- Disk fill-up recovery + restore scores
- Prevent fraudulent KL=0 from poisoning early stopping
- Preserve state across restarts (no more from-scratch)
- Detect MoE fraud + min file size check
- Pre-load disk check (emergency cleanup at >85% full)
- DQ by hotkey instead of UID (UIDs get recycled)
- Set_weights returns tuple not bool
- Commitments is dict keyed by UID, not list of dicts
- Pre-flight state validation to prevent wasted eval rounds
- Granular progress for every eval phase — no dead zones
- Per-commit DQ — miners can re-register after disqualification
- Use real on-chain block hash for prompt selection (security)
- Disable vLLM when tmux not available on pod

### Changed
- Eval prompts tuned through 40→60→80→120 during mass re-eval
- `MIN_PROMPTS_EARLY_STOP`: 3 → 7

## [2026-03-30]
### Added
- H2H round results table for dashboard transparency
- Switch to climbmix-400b-shuffle dataset for instant prompt sampling
- Expand prompt sampling range from 50K to 500K items
- Live eval progress with teacher generation phase
- Save results incrementally after each student
- Parallel 2-GPU eval support
- Sample prompts from full 1.5T-token FineWeb dataset each epoch
- Increase prompt pool from 500 to 10,000 — prevents overfitting
- Auto-update script and simplified validator setup
- Validator Guide section in README

### Fixed
- Don't DQ models on transient HF errors (429/timeout/503)
- Handle metagraph sync returning n=1 with retry
- Skip king-only evals and don't save king-only H2H rounds
- Guard against IndexError when metagraph.n is stale
- Preserve king global score across H2H evals
- Flush prompts file before upload to prevent empty file
- Force exit after eval to prevent process hang
- Treat UIDs without scores as unevaluated
- Clean student HF cache between models during eval
- Auto-clean HF model cache after each eval epoch
- Retry logic for SFTP uploads to Lium pod
- Input sanitization + API memory fix (subprocess for bittensor calls)

### Changed
- Switch from SWE-bench to FineWeb pretraining dataset
- Use HF API metadata instead of downloading shards for integrity check
- Reduce samples to 20 per epoch (was 100)
- 60 prompts per eval, switch to B200 pod for faster inference
- 80 prompts per eval with FineWeb streaming memory fix

## [2026-03-29]
### Added
- **Initial release** — Distillation subnet for Qwen model compression
- King-of-the-hill validator architecture
- GPU-based KL divergence evaluation on rented pods (Lium/Vast.ai)
- Duplicate model hash detection
- Logit fingerprinting to detect functional copies
- Confidence-based early stopping in GPU eval
- Score history with trendline chart + `/api/history` endpoint
- Model info API endpoint
- 1% epsilon threshold for king dethronement
- Disqualification transparency + README
- `/api/scores` endpoint for eval data
- `/api/price` for subnet alpha price + TAO/USD
- Disk-backed API cache with background refresh
- PM2 validator wrapper script
- Discord announcements for king changes (tok/s benchmark, earnings)
- Live eval progress: per-prompt KL, CI, completed results
- Event-driven eval + atomic announcement claims

### Changed
- Winner-take-all scoring replaces EMA — raw KL per epoch
- Permanent commitments with tokenizer enforcement and KL floor
- Fast full-distribution KL eval with same-tokenizer enforcement
- Remote validator with first real on-chain eval + weight setting

### Security
- Remove hardcoded TMC key, add `.env` support
- Model integrity checks on HuggingFace submissions
