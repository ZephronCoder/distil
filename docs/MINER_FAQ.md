# Subnet 97 (Distil) — Miner FAQ & Getting Started

## What is Subnet 97?

Distil is a Bittensor subnet where miners compete to distill knowledge from a large teacher model into smaller student models. The teacher is **Qwen/Qwen3.5-35B-A3B** (35B total params, ~3B active — it's a Mixture-of-Experts model). Your job: produce the most faithful small model (≤5.25B params), measured by KL divergence against the teacher's output distribution.

Lower KL = better. Winner takes all — the king gets 100% of emissions.

---

## Getting Started

### 1. Register on SN97

Register a hotkey on subnet 97 via the standard Bittensor registration flow (`btcli subnet register --netuid 97`).

### 2. Train Your Student Model

- **Architecture:** Must be `Qwen3_5ForConditionalGeneration` with `model_type: "qwen3_5"` in `config.json`
  - ⚠️ **NOT** `Qwen3_5ForCausalLM` / `qwen3_5_text` — this will get you disqualified
- **Max total params:** 5.25B (total, not active — MoE tricks won't help)
- **Tokenizer:** Must be identical to the teacher's tokenizer (vocab size 248,320). Don't modify `tokenizer.json` or `tokenizer_config.json`
- **No quantization:** bf16/fp16 only. GPTQ, AWQ, GGUF etc. are rejected
- **No custom code:** `.py` files in your repo (except `__init__.py`) will get you DQ'd
- **Format:** Safetensors required (no pytorch `.bin`-only models)

### 3. Upload to HuggingFace

Push your model to a **public** HuggingFace repo. It must stay public — private or deleted models get disqualified.

### 4. Commit Your Model

Submit your HuggingFace model repo via the commitment mechanism on-chain. 

**⚠️ Commitments are permanent.** One model per hotkey, forever. You cannot re-upload or swap models on the same hotkey. Choose carefully.

---

## How Evaluation Works

1. **300 prompts per round** sampled from the [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset, seeded by the current block
2. Both teacher and student generate completions (greedy, temp=0, max 8,192 new tokens)
3. KL divergence is computed between teacher and student output distributions
4. **King-of-the-hill:** The current best model (king) is re-evaluated alongside challengers every round
5. **Dethronement:** A challenger must beat the king with statistical significance (paired t-test, p < 0.03) to take the crown
6. **Top 5 contenders** (by KL score) are always included in every eval round
7. **Winner takes all** — the king receives 100% of emissions
8. A reference baseline (undistilled Qwen3.5-4B) is evaluated every round as UID -1 for comparison

---

## Upcoming Mechanism Change — Multi-Axis Composite Score

**What is changing.** KL divergence alone is insufficient: it can be over-optimized until the model produces gibberish under autoregressive sampling while still matching teacher logit shape on teacher-generated continuations. This has been demonstrated both in the literature (Tiapkin et al., 2025, "Teacher Hacking in Knowledge Distillation") and empirically on SN97.

Starting in the validator commit cycle following the 14-day grace period, the validator will rank models by a **multi-axis composite score** rather than KL alone. The composite aggregates four independent axes using a **worst-case rule** — you win only if you're competitive on *every* axis.

**The four axes.**

1. **KL axis** (still present). Teacher-forced KL divergence on teacher continuations, as today. Normalized against the current king: a student matching king KL scores 1.0 on this axis.
2. **Capability axis.** A small battery of verifiable prompts (arithmetic, yes/no, one-word factual) scored by exact-match extraction. The teacher is evaluated on the same prompts each round, and your score is normalized against teacher pass rate.
3. **Length axis.** Your mean generation length on a held-out probe versus the teacher's mean generation length. Rambling or looping students are penalized; hard-stopping students are rewarded up to a cap.
4. **Degeneracy axis.** The existing thinking-collapse probe (termination fraction, MAD-z-scored repetition versus the teacher, cross-rollout Self-BLEU).

**Aggregation: worst-axis rule.** The composite is `min(axis_1, axis_2, axis_3, axis_4)`. Gaming any one axis at the expense of others brings your composite down. This is the Coste 2024 / Pan et al. 2025 min-form credit assignment rule.

**Secondary: weighted mean.** We also log a weighted mean of the axes (KL and capability each 0.35, length and degeneracy each 0.15) as a softer aggregation used for display. The ranking key will be worst-axis.

**Shadow mode (now).** Until the switchover, the composite is logged alongside KL in the H2H output and API. KL still decides the king. Watch your composite score — if it's low, you'll lose once the switch happens.

**Why this is good for miners.** The on-chain commitment is permanent — it's in your interest that the thing the subnet rewards is actually correlated with model quality, because otherwise SN97 becomes a curiosity rather than a serious distillation program and alpha trends to zero. Under the new mechanism, a student that makes steady progress on real model quality *wins automatically*, whereas today it loses to a model that produces garbage but happens to match teacher logit shape.

**What to train for.**

- Reverse-KL under the student's own sampling (not forward-KL on teacher rollouts). See Thinking Machines, "On-Policy Distillation" (Nov 2025); GKD (Agarwal et al., 2024); MiniLLM (Gu et al., 2023).
- Supervised fine-tuning on a capability mix (GSM8K, IFEval, basic QA) interleaved with the distillation objective.
- Early termination behavior — don't emit `<think>` loops on trivial prompts.

---

## Training Tips

- **Base model:** Start from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or a compatible Qwen3.5 architecture
- **Objective:** Standard knowledge distillation — minimize KL(teacher ‖ student) on the teacher's output distribution
- **Long completions matter:** Eval uses `max_new_tokens=8192`, so your model needs to handle long generations well
- **Temperature:** Eval runs greedy (temp=0) — train accordingly
- **Don't modify the chat template:** It's checked against the reference Qwen3.5 template hash. Injected comments or modifications = DQ

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Wrong architecture" DQ | `config.json` has `Qwen3_5ForCausalLM` or wrong `model_type` | Change `architectures` to `["Qwen3_5ForConditionalGeneration"]` and `model_type` to `"qwen3_5"` in config.json. No weight changes needed. |
| "Integrity check failed" | Model weights changed after commitment | You can't re-upload on the same hotkey. Commitments are permanent. Register a new hotkey. |
| "Copy detected" | Model hash matches another miner's submission | Your weights are identical to another miner's. Train your own model. |
| "Model is now private" DQ | HuggingFace repo set to private or deleted | Keep your model repo public at all times. |
| "Vocab size mismatch" | Modified tokenizer | Use the exact same tokenizer as the teacher (Qwen3.5-35B-A3B). |
| "Quantized model detected" | Model has `quantization_config` in config.json | Remove quantization. Use bf16/fp16 weights only. |
| "Custom code files" DQ | `.py` files found in your repo | Remove all Python files from your HuggingFace repo. |
| "Tokenizer encoding mismatch" | Tokenizer produces different token IDs than teacher | Use the unmodified teacher tokenizer files. |
| "Chat template modified" | `chat_template` in tokenizer_config.json differs from reference | Use the original Qwen3.5 chat template without modifications. |

---

## Useful Links

- **Dashboard:** <https://distil.arbos.life>
- **API Health:** <https://api.arbos.life/api/health>
- **GitHub:** <https://github.com/unarbos/distil>
- **Discord:** Channel `ა・distil・97` in the Bittensor Discord

---

## API Endpoints for Miners

All endpoints are on `api.arbos.life`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/miner/{uid}` | Details for a specific miner |
| `GET /api/scores` | Current scores |
| `GET /api/leaderboard` | Leaderboard (who's king, top contenders) |
| `GET /api/compare?uids=2,34,36` | Head-to-head comparison between miners |
| `GET /api/eval-status` | Current eval round status |
| `GET /api/eval-data` | Raw eval data |
| `GET /api/eval-stats` | Eval statistics |
| `GET /api/pod-logs` | Pod logs (paginated) |

---

## Key Constants

| Parameter | Value |
|-----------|-------|
| Subnet UID | 97 |
| Teacher model | `Qwen/Qwen3.5-35B-A3B` |
| Max student params | 5.25B (total) |
| Required architecture | `Qwen3_5ForConditionalGeneration` |
| Required model_type | `qwen3_5` |
| Vocab size | 248,320 |
| Eval prompts (head-to-head) | 300 |
| Eval prompts (broad sweep) | 60 |
| Max new tokens | 8,192 |
| Max prompt tokens | 1,024 |
| Dethronement threshold | paired t-test, p < 0.03 |
| Top-N always included | 5 |
| Dataset | `karpathy/climbmix-400b-shuffle` |
| Reference baseline | `Qwen/Qwen3.5-4B` (UID -1) |
