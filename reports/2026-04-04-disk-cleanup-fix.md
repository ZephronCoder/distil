# Disk Cleanup Fix — Eval Pipeline Stability

**Date:** 2026-04-04
**Author:** OpenClaw (subagent)
**Root cause analysis by:** caseus (github.com/winglian)

## Problem

The eval pipeline (remote_validator.py + pod_eval_vllm.py) crash-looped daily due to disk exhaustion on the Lium GPU pod. The ~230GB pod disk filled up because:

1. `disk_check_and_clean()` only ran before student loads, never before teacher loading
2. `start_vllm_server()` and HF fallback had zero disk awareness
3. Failed vLLM downloads left partial files in /tmp — next attempt failed the same way
4. `remote_validator.py` pre-eval cleanup only triggered at >80% and only cleaned student caches
5. `teacher_cache.pt` (~45GB) was never cleaned between rounds
6. Stale logit caches from previous blocks accumulated

## Changes

### pod_eval_vllm.py

1. **Aggressive disk cleanup before teacher loading** (before vLLM path):
   - `disk_check_and_clean(threshold=70)` — lower threshold than default 85
   - Stale teacher cache removal if prompts_hash doesn't match
   - `/tmp` cleanup of large files (>1GB `teacher_*`, `vllm_*`, `tmp*`)
   - Disk usage logging before and after cleanup

2. **Disk check inside `start_vllm_server()`** (top of function):
   - Checks disk before model download starts
   - At >85%, runs `disk_check_and_clean(threshold=80)` plus /tmp cleanup
   - Logs disk state before vLLM startup

3. **Disk check before HF fallback teacher load**:
   - `disk_check_and_clean(threshold=70)` before `load_model()` call
   - Disk usage logging

4. **Disk usage logging at key phases**:
   - After teacher unload (disk + VRAM)
   - Before each student load (disk state)

### remote_validator.py

5. **More aggressive pre-eval cleanup via SSH**:
   - ALWAYS cleans non-teacher student caches (removed >80% threshold gate)
   - Cleans stale /tmp files >1GB older than 30 minutes
   - Cleans stale logit caches (eval_gpu0.json, eval_gpu1.json, eval_teacher_only.json)

6. **teacher_cache.pt cleanup between rounds**:
   - New rounds (not resuming) now delete teacher_cache.pt (~45GB freed)
   - pod_eval regenerates teacher logits for new prompt set anyway

7. **Post-eval cleanup enhanced**:
   - Also removes teacher_cache.pt after round completion
   - Cleans /tmp large files >1GB

## Files Modified

- `scripts/pod_eval_vllm.py` — 4 insertion points
- `scripts/remote_validator.py` — 3 insertion points

## Verification

Both files compile cleanly with `python3 -m py_compile`.

## Notes

- The eval is currently running on the pod — changes take effect on next round
- No validator restart needed; remote_validator.py re-uploads pod_eval_vllm.py each epoch
