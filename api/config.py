import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.runtime import (
    ALLOWED_ORIGINS,
    CACHE_TTL,
    CHAT_POD_APP_PORT,
    CHAT_POD_HOST,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    DASHBOARD_URL,
    DISK_CACHE_DIR,
    NETUID,
    STATE_DIR,
    TMC_BASE,
    TMC_HEADERS,
)

CHAT_POD_PORT = CHAT_POD_APP_PORT

# ── Tunables (single source of truth, duplicated in several places before) ──
# 2026-04-28: STALE_EVAL_BLOCKS retired (was a misleading time-based cooldown
# in the dashboard ``eval_status`` text). Single-eval mode re-evaluates a UID
# only when its on-chain commitment changes; there is no time-based re-test.
# Kept here only so any external consumer that imports the constant still
# resolves (returns the legacy default), but the API routes no longer use it.
STALE_EVAL_BLOCKS = 50  # deprecated, retained for backward-compat import only
EPOCH_BLOCKS = 360
CHAT_RESTART_COOLDOWN = 120
CHAT_SERVER_SCRIPT = "/root/chat_server.py"
MAX_COMPARE_UIDS = 10
MAX_BATCH_UIDS = 64
ANNOUNCEMENT_CLAIMS_KEEP = 50

API_DESCRIPTION = f"""
# Distil - Subnet {NETUID} API

Public API for [Distil]({DASHBOARD_URL}), a Bittensor subnet where miners compete to produce the best knowledge-distilled small language models.

## How It Works

Miners submit distilled models and a validator scores each commitment on a **multi-axis composite** (`scripts/validator/composite.py`) anchored on **11 procedurally-generated v31 axes**:

- **Math**: `v31_math_gsm_symbolic` (0.06), `v31_math_competition` (0.05), `v31_math_robustness` (0.03 — GSM-NoOp distractor injection)
- **Code**: `v31_code_humaneval_plus` (0.08 — sandbox-graded, EvalPlus-augmented test cases), `v31_ifeval_verifiable` (0.04 — constraint-driven IFEval)
- **Reasoning**: `v31_reasoning_logic_grid` (0.05 — zebra-puzzle constraint-sat), `v31_reasoning_dyval_arith` (0.04 — arithmetic on dynamic DAGs), `v31_long_context_ruler` (0.05 — NIAH at variable context)
- **Knowledge**: `v31_knowledge_multi_hop_kg` (0.04 — procedural 2–3 hop KG)
- **Honesty**: `v31_truthfulness_calibration` (0.03 — Brier-scored calibration), `v31_consistency_paraphrase` (0.03 — IPT name-rotation consistency)
- **Distillation**: `on_policy_rkl` (0.30), `top_k_overlap` (0.18), `kl` (0.05), `capability` (0.05), `length` (0.05), `degeneracy` (0.05)
- **Quality**: `judge_probe` (0.20), `long_form_judge` (0.20), `long_gen_coherence` (0.25), `chat_turns_probe` (0.10)
- **Discipline**: `reasoning_density` (0.05), `calibration_bench` (0.05)
- **Telemetry tier (composite weight 0)**: legacy `*_bench` and `*_skill_group` axes still RUN every round but no longer touch ranking — they exist for dashboard visibility and `axis_correlation.json`.

**Ranking key** (v32.5, 2026-05-13): `composite.final = 0.75 × worst_3_mean + 0.25 × weighted` where:
- `worst_3_mean` = equal-weighted mean of the 3 lowest non-broken axes (K reverted from 5 to 3 in v32.5 — the API field name now matches the math again)
- `weighted` = standard weighted convex combination of all active axes

The king is whoever has the highest `composite.final`. A challenger dethrones only when its final beats the incumbent's by `SINGLE_EVAL_DETHRONE_MARGIN` (default 5% since 2026-05-10, was 3%). The legacy `composite.worst` (single-axis min) is retained as telemetry but is no longer the dethrone gate.

## Quick Start

```bash
curl https://api.arbos.life/api/health
curl https://api.arbos.life/api/scores
curl https://api.arbos.life/api/price
```
"""
