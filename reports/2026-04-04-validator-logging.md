# Validator Logging API Endpoint

**Date:** 2026-04-04
**Status:** ✅ Complete
**Commit:** `feat: add real-time validator logging API endpoint`

## Summary

Added real-time structured validator logging accessible via a new API endpoint. Miners can now see validator activity (round starts, precheck results, eval progress, per-student KL scores, round outcomes, errors) without SSH access.

## What Changed

### 1. `eval/state.py` — `log_event()` function
- New `VALIDATOR_LOG_FILE = "validator_log.json"` constant
- `log_event(msg, level="info", state_dir="state")` — appends structured entries to `state/validator_log.json`
- Each entry: `{"ts": unix_timestamp, "level": "info|warn|error", "msg": "..."}`
- Caps at 100 entries (FIFO), uses atomic writes for safety
- Stateless function (no ValidatorState dependency) — can be called from anywhere

### 2. `api/server.py` — `GET /api/validator-logs`
- New endpoint: `GET /api/validator-logs?limit=50`
- Default limit=50, max limit=200
- Returns entries most-recent-first
- 5s cache TTL for fast refresh
- Uses existing `_safe_json_load` for robustness
- OpenAPI docs with full description

### 3. `scripts/remote_validator.py` — log_event calls at key points
- **Epoch start:** `"Starting epoch N"`
- **Precheck results:** `"Prechecked N models: X valid, Y DQ, Z error"`
- **Round start:** `"Starting h2h round N, king=UID X, challengers=[...]"`
- **Eval start:** `"Running eval on pod: king vs N challengers, M prompts"`
- **Per-student results:** `"UID X: KL=0.095, +/-N% vs king"` (both king and challengers)
- **Round end:** `"Round complete. Winner: UID X, KL=0.085. Weights set."` or new king announcement
- **Disk cleanup:** `"Pod disk: 45% used after cleanup"`
- **Chain errors:** `"Chain unreachable: ..., retrying in 5min"`
- **Pod cleanup errors:** logged at warn level
- **Epoch errors:** `"Epoch error: ..."` at error level

## Files Modified

| File | Change |
|------|--------|
| `eval/state.py` | Added `log_event()`, `VALIDATOR_LOG_FILE`, `VALIDATOR_LOG_MAX_ENTRIES` |
| `api/server.py` | Added `GET /api/validator-logs` endpoint |
| `scripts/remote_validator.py` | Added `log_event` import + 10 log_event calls at key points |

## Deployment

- Deployed to prod via `scripts/deploy-prod.sh api`
- API healthy (HTTP 200) after restart
- `state/validator_log.json` will be created on next validator epoch
- Syncs to prod every 15s via `distil-sync`
- No validator restart needed — changes take effect on next restart

## API Usage

```bash
# Get last 50 log entries (default)
curl https://api.arbos.life/api/validator-logs

# Get last 20 entries
curl https://api.arbos.life/api/validator-logs?limit=20

# Response format
{
  "entries": [
    {"ts": 1743789600.0, "level": "info", "msg": "Round complete. Winner: UID 42, KL=0.085. Weights set."},
    {"ts": 1743789590.0, "level": "info", "msg": "UID 42: KL=0.085, -3.41% vs king"},
    ...
  ],
  "count": 50
}
```

## Notes

- The validator log file doesn't exist yet — it'll be created on the next validator epoch
- The log_event function is intentionally stateless (takes `state_dir` as param) to avoid coupling with ValidatorState instance lifecycle
- Log entries are capped at 100 to keep the file small for fast sync/reads
