# Distil SN97 — Roadmap

Last updated: 2026-04-13

## ✅ Recently Shipped

- **Eval v3** — 300 prompts/round, sparse top-128 KL divergence, paired t-test (p<0.03)
- **Anti-exploit** — Pre-dethronement integrity checks, shard-invariant hashing, copy detection
- **Transparency** — `/api/evaluated_uids`, `/api/dq_reasons`, `/api/model_hashes`, `/api/eval-stats`, `/api/compare`
- **Dashboard** — Live eval progress, reference baseline (Qwen3.5-4B), p-value display fix
- **Performance** — Concurrent teacher generation (4x faster), vLLM student scoring (experimental)
- **Documentation** — Miner FAQ, CHANGELOG.md, API transparency endpoints

## 🔄 In Progress

- **Eval speed** — Targeting <1hr rounds (currently ~2-3hr). Concurrent generation deployed, vLLM student scoring under A/B testing
- **Downstream benchmarks** — Adding IFEval, reasoning tasks as secondary quality signal (KL remains primary)
- **Codebase cleanup** — Splitting monolithic files into modules for easier contribution

## 📋 Planned

- **Local eval guide** — Step-by-step for miners to reproduce scores locally
- **Training starter code** — Example distillation script with recommended hyperparameters
- **Eval data explorer** — Browse prompts, completions, and per-prompt KL on the dashboard
- **Historical leaderboard** — Track miner ranking over time, not just current snapshot
- **Multi-teacher** — Evaluate against multiple teachers for robustness (research phase)

## 💡 Under Consideration

- **Partial credit** — Proportional emissions instead of winner-takes-all (needs community input)
- **Dynamic prompt selection** — Weight harder prompts more heavily
- **Model quality metrics** — Benchmark scores alongside KL for a holistic view

---

Have ideas? Drop them in the `ა・distil・97` channel.
