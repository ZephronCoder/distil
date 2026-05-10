# SN97 Autonomous Operations Runbook

Goal: run the subnet unattended for 30+ days while still producing a SOTA distilled model that earned the crown legitimately.

Read this end-to-end so you can leave the subnet alone for a month and trust the watchdogs.

## Layered defence

Time-to-detect ceiling: 3 minutes for chat, 5 minutes for everything else. Each timer has a single responsibility and writes its state to a file the others can read.

- chat-keeper.timer (3 min): probes the chat pod and re-launches chat_server.py when the king's vLLM has fallen over.
- sn97-bot-snapshot.timer (1 min): refreshes state/sn97_snapshot.json so the Discord bot answers with current numbers.
- sn97-healthcheck.timer (5 min): the catch-all. Probes validator, API, dashboard, OpenClaw, chat surface, GPU log freshness, disk, and journal failure rate.
- distil-openclaw-config-guard.timer (5 min): audits OpenClaw config for tool/thread regressions.
- distil-benchmark-sync.timer (75 s): refreshes the held-out canary for the king.
- distil-owui-patches.timer (5 min): keeps the chat UI on the SN97-branded patches.

## Self-repair (sn97-healthcheck.py)

Detection heuristics and the corresponding fix-or-noop:

- API non-2xx returned: restart distil-api (budget-gated).
- Dashboard not 200: restart distil-dashboard.
- eval_progress.json mtime older than 3 hours: restart distil-validator and clear stale pod state.
- gpu_eval.log silent for 15+ minutes: restart distil-validator (validator owns the writer).
- Disk at 80 percent or higher: reclaim HF hub caches older than 7 days except current student.
- OpenClaw 401 from Discord: notify only. Auth is a manual fix; restart-on-401 just thrashes.
- Chat pod down: notify only. chat-keeper.timer does the actual heavy lift via SSH.

Two safety rails:

1. Restart budget: at most 2 restarts/hour for any single unit. After that the script logs an issue and noops the repair so a flapping service cannot churn the host.
2. Download awareness: if chat_server.py is actively downloading a multi-GB HF model, the heal call returns early.

Snapshots: state/healthcheck.json (latest run) and state/incidents.jsonl (rolling event log, surfaced at /api/incidents).

## Goodhart drift watchdog (axis correlation)

Source: scripts/audit/axis_correlation.py.

The validator is a faithful proxy for SOTA capability only when its per-axis scores correlate with the corresponding held-out canary. The audit script computes the rolling Pearson r between every active axis and its held-out counterpart and writes state/axis_correlation.json.

Sustained r at or below -0.3 over n at or above 8 paired kings on a non-zero-weight axis is the dominant Goodhart alert. When that fires, treat it as the most important signal you can act on: that axis has stopped tracking the SOTA holdout we care about and is rewarding overfitting.

The most recent audit (post-Kimi cutover, n=5 paired kings) is reports/2026-05-10-axis-correlation-audit.md. Re-run weekly while the post-cutover history grows.

## Anti-Goodhart gates already shipped

These are not runtime watchdogs; they are design watchdogs that the validator enforces every round.

- 11 procedurally-generated v31 axes (math, code, reasoning, long-context, knowledge, honesty): every item is generated from the on-chain block-seed, so there is no static answer key to memorise.
- Isomorphic Perturbation Testing in v31_consistency_paraphrase: first names are rotated within gender between paired isomorphic problems; a model that memorised the canonical wording fails the rotated one.
- GSM-NoOp topical-distractor injection in v31_math_robustness: mathematically irrelevant clauses are injected; a memorised distribution that triggers on surface keywords fails.
- Worst-K mean ranking with K=5: a model cannot camp specialists; it has to be competent on its 5 weakest non-broken axes.
- 5 percent dethrone margin: a challenger needs to beat the king on composite.final by at least 5 percent so RNG variance cannot crown a copy.
- One-eval-per-commitment: non-king miners are scored exactly once per on-chain commitment, so a re-roll requires a fresh registration burn.
- King paired re-eval: the king is re-evaluated every round on the same procedural items as challengers, so a lucky-round king cannot hold the crown forever.
- SHA256 + activation-fingerprint copy detection: identical safetensors weights or near-identical activation distributions on the first prompts get DQ'd.

## Weekly check (10 minutes, every Monday)

1. journalctl -u distil-validator since 24 h ago filtered for error / warn should trend below 200/day. A sudden jump usually points at a vLLM or pod-side regression.
2. /api/incidents length should drop to near 0 between issues. A persistent non-zero is an unresolved repair the script gave up on.
3. state/axis_correlation.json: open the audit report or just cat the file and skim for any axis at r below -0.3 with n at or above 8.
4. python scripts/audit/axis_correlation.py | head -40: re-run so the next operator sees fresh data.
5. Discord #distil-97: scan the last 24 h. The bot answers in-thread; user complaints almost always pinpoint a specific axis or behaviour the audit didn't catch yet.
6. python scripts/sn97_healthcheck.py --format markdown: one-shot human-readable rollup.

## Failure modes the autonomous stack does NOT cover

These still need a human. They are rare but they bypass the watchdog.

- Chain hyperparameter changes (burn rate, immunity period): coldkey-signed transactions only.
- vLLM major version upgrade breaking the king's reasoning parser (e.g. the distil_kimi parser circular-import we hit on 2026-05-09 with vLLM 0.20.0+).
- HuggingFace outage: validator gracefully degrades but commitments piling up will fire at once when HF is back.
- Bittensor chain freeze or reorg: the validator's set_weights retries are bounded; a multi-hour freeze leaves the king static until the chain resumes.
- A new evaluation methodology from a major lab that is more robust than ours and we should adopt: the watchdog won't tell you about it; the research-pass cadence does.

If any of those happen while you are away, the dashboard will show stale data but the validator will not corrupt state.
