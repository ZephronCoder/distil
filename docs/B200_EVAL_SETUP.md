# B200 Eval Pod Setup — Reproducibility Notes

**Date:** 2026-04-03
**GPU:** NVIDIA B200 (sm_100, 183GB VRAM, CUDA driver 580.x)
**Cost:** $2.99/hr on Lium

## Working Configuration

### Template
- **Lium template:** `64c96459-29f1-4578-8785-0c56ec3341cc`
- **Image:** `daturaai/pytorch:2.7.0-py3.12-cuda12.8.0-devel-ubuntu22.04`
- This gives Python 3.12, PyTorch 2.7.0+cu128, CUDA 12.8

### Dependencies (installed on pod)
```bash
pip install vllm accelerate transformers
```
This upgrades PyTorch to 2.10.0+cu128 (vLLM's dependency) which is fine — B200 sm_100 support works.

### Final versions (confirmed working)
- **PyTorch:** 2.10.0+cu128
- **vLLM:** 0.19.0
- **transformers:** 5.5.0
- **Python:** 3.12.10

## vLLM on B200

### ✅ Works
- vLLM **server mode** (`python3 -m vllm.entrypoints.openai.api_server`) — works fine
- Must use `--enforce-eager` to skip torch.compile/CUDAGraph (sm_100 kernel compilation hangs or fails)
- Server starts in ~15s for gpt2, ~5-10 min for Qwen3.5-35B-A3B

### ❌ Common Pitfalls
1. **Multiprocessing spawn guard**: vLLM uses `spawn` multiprocessing. Any script using `from vllm import LLM` must have `if __name__ == "__main__":` guard, or the child process re-executes the entire script and crashes.
2. **Old PyTorch**: PyTorch < 2.7.0 does NOT support B200 sm_100. The default `Pytorch (Cuda)` template (id `bfda7aa0`) has PyTorch 2.1.1+cu121 — **DO NOT USE** for B200.
3. **FlashInfer**: Works with `--enforce-eager`. Without it, kernel compilation may hang for 10+ minutes on first run.
4. **Template `vllm/vllm-openai:latest`**: Has vLLM pre-installed but **NO SSH server** — unusable with Lium's SSH-based exec.

### vLLM Server Launch Command (for eval)
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-35B-A3B \
  --port 9100 \
  --served-model-name teacher \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --enforce-eager \
  --no-enable-log-requests
```

### Startup Timeout
- vLLM server for 35B MoE model on B200: allow **15 minutes** (900s) for first startup
  - Model download: variable (depends on cache)
  - Weight loading: ~2-3 min
  - With `--enforce-eager`: no compilation step
  - The eval script (`pod_eval_vllm.py`) has a 900s health-check loop

## Multi-GPU Pods (4090, 3090, etc.)

### ⚠️ RAM Limitation
- `teacher_cache.pt` is ~60GB (120 prompts × full vocab logits for 35B model)
- Parallel Step 2 (split students across GPUs) loads this cache per process
- **Require ≥128GB RAM** for parallel mode, otherwise OOM kills processes
- The validator checks RAM and falls back to sequential mode if < 128GB

### Tensor Parallelism
- TP size must be a power of 2 (1, 2, 4, 8) — model dims must be divisible
- The eval script picks the largest power-of-2 ≤ GPU count
- 5×4090: uses TP=4 (one GPU idle)

## Troubleshooting

### "Engine core initialization failed"
- Usually multiprocessing spawn issue — add `if __name__ == "__main__":` guard
- Or PyTorch too old for the GPU architecture

### "8192 is not divisible by N"
- TP size must divide model dimension. Use power-of-2 TP.

### SSH "Error reading SSH protocol banner"
- The `vllm/vllm-openai` Docker image has no SSH. Use a PyTorch template instead.
- Some B200 executors take 4+ minutes for SSH to become available after RUNNING status.

### Teacher cache too large for RAM
- Sequential eval mode (1 GPU process at a time) uses ~60GB RAM
- Parallel mode (2 processes) needs ~120GB RAM
- Solution: the validator auto-detects RAM and disables parallel mode if < 128GB
