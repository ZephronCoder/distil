"""Model architecture/identity checks: MoE-aware param count, vocab/tokenizer
match against the teacher, safetensors hash for copy detection."""
import json
import hashlib
import logging
import os
import time
import requests as _requests
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, model_info as _raw_model_info
from eval.runtime import (
    STATE_DIR as RUNTIME_STATE_DIR,
    STUDENT_ARCH_ALLOWLIST,
    STUDENT_ARCH_NAMES,
    STUDENT_MODEL_TYPES,
    TEACHER_CONFIG_VOCAB_SIZE,
    TEACHER_MODEL,
)

logger = logging.getLogger("distillation.model_checker")


# HF model_info wrapper: thread HF_TOKEN through, retry transient 429/5xx.
_HF_TOKEN = os.environ.get("HF_TOKEN") or None
_MODEL_INFO_RETRY_DELAYS = (0.5, 1.5, 4.0)  # ~6s total


def model_info(model_repo, revision=None, files_metadata=False, token=None, **kwargs):
    """HF model_info with auth + bounded 429/5xx retries."""
    effective_token = token if token is not None else _HF_TOKEN
    last_exc = None
    for attempt, delay in enumerate((0.0,) + _MODEL_INFO_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return _raw_model_info(
                model_repo,
                revision=revision,
                files_metadata=files_metadata,
                token=effective_token,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if not any(k in msg for k in ("429", "rate limit", "too many", "503", "502", "timeout")):
                raise
            if attempt == len(_MODEL_INFO_RETRY_DELAYS):
                raise
            logger.debug(
                f"model_info({model_repo}@{revision}) attempt {attempt + 1} hit transient "
                f"{type(exc).__name__}; backing off {_MODEL_INFO_RETRY_DELAYS[attempt]}s"
            )
    if last_exc:
        raise last_exc

BASELINE_VOCAB_SIZE = TEACHER_CONFIG_VOCAB_SIZE
STATE_DIR = Path(RUNTIME_STATE_DIR)


def compute_moe_params(config: dict) -> dict:
    """Compute total/active params for MoE models.

    Returns {total_params, active_params, is_moe, num_experts, num_active_experts}.
    Supports nested configs (e.g. text_config for multimodal); text_config
    values are used as fallback.
    """
    text_cfg = config.get("text_config", {})
    def _get(key, default=0):
        v = config.get(key)
        if v is None or v == 0:
            v = text_cfg.get(key)
        return v if v is not None else default

    hidden = _get("hidden_size", 0)
    layers = _get("num_hidden_layers", 0)
    vocab = _get("vocab_size", 0)
    intermediate = _get("intermediate_size", hidden * 4)
    num_heads = _get("num_attention_heads", 0)
    kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)

    if not all([hidden, layers, vocab]):
        return {"total_params": 0, "active_params": 0, "is_moe": False}

    # Attention params per layer: Q + K + V + O
    attn_per_layer = (
        hidden * num_heads * head_dim  # Q
        + hidden * kv_heads * head_dim  # K
        + hidden * kv_heads * head_dim  # V
        + num_heads * head_dim * hidden  # O
    )
    total_attn = layers * attn_per_layer

    # Embeddings: input + output (may share weights)
    tie_word = config.get("tie_word_embeddings", False)
    embed_params = vocab * hidden * (1 if tie_word else 2)

    # Layer norms, biases, etc. (rough estimate)
    norm_params = layers * hidden * 4  # 2 norms per layer, 2 params each

    # MoE detection — check both top-level and text_config
    num_experts = _get("num_local_experts", 0) or _get("num_experts", 1)
    num_active = _get("num_experts_per_tok", 0) or _get("num_active_experts", num_experts)
    is_moe = num_experts > 1

    # FFN params per expert (SwiGLU: gate + up + down)
    # Some models use moe_intermediate_size for expert FFN
    expert_intermediate = _get("moe_intermediate_size", intermediate)
    ffn_per_expert = hidden * expert_intermediate * 2 + expert_intermediate * hidden

    if is_moe:
        # Some layers may be dense (shared experts)
        num_shared = _get("num_shared_experts", 0)
        shared_intermediate = _get("shared_expert_intermediate_size", intermediate)
        shared_ffn = hidden * shared_intermediate * 2 + shared_intermediate * hidden if num_shared else 0

        router_per_layer = hidden * num_experts
        total_ffn = layers * (num_experts * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
        active_ffn = layers * (num_active * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
    else:
        total_ffn = layers * ffn_per_expert
        active_ffn = total_ffn

    total_params = total_attn + total_ffn + embed_params + norm_params
    active_params = total_attn + active_ffn + embed_params + norm_params

    return {
        "total_params": total_params,
        "active_params": active_params,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_active_experts": num_active,
    }


def get_safetensors_param_count(model_repo: str, revision: str = None) -> float:
    """Get verified param count from safetensors metadata (billions). Returns -1 if unavailable."""
    try:
        info = model_info(model_repo, revision=revision)
        if info.safetensors and hasattr(info.safetensors, "total"):
            return info.safetensors.total / 1e9
    except Exception:
        pass
    return -1.0


_EMBED_KEYS = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "transformer.wte.weight",
)
_FLOAT_DTYPES = ("BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2")
_QUANT_TENSOR_MARKERS = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.",
    ".SCB",
    ".weight_format",
    ".g_idx",
    "qweight",
    "qzeros",
    "scales",
)


def _read_safetensors_header(session, url: str) -> Optional[dict]:
    """Fetch the JSON header of a safetensors file via HTTP range requests."""
    import struct
    pr = session.get(url, headers={'Range': 'bytes=0-7'}, timeout=30,
                     stream=True, allow_redirects=True)
    pr.raise_for_status()
    prefix = pr.raw.read(8)
    pr.close()
    if len(prefix) != 8:
        return None
    header_size = struct.unpack('<Q', prefix)[0]
    if header_size <= 0 or header_size > 8_000_000:
        return None
    header_len = 8 + header_size
    hr = session.get(url, headers={'Range': f'bytes=0-{header_len - 1}'},
                     timeout=60, stream=True, allow_redirects=True)
    hr.raise_for_status()
    blob = hr.raw.read(header_len)
    hr.close()
    return json.loads(blob[8:header_len].decode('utf-8'))


def get_embed_weight_shape(model_repo: str, revision: str = None) -> Optional[tuple[int, int, str]]:
    """Read shape + dtype of model.embed_tokens.weight from safetensors headers
    (no tensor data download). Catches configs that claim teacher vocab while
    shipping older/smaller checkpoint weights. Returns (vocab, hidden, dtype)
    or None."""
    try:
        from huggingface_hub import hf_hub_url
        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            return None
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            for fname in st_files:
                url = hf_hub_url(repo_id=model_repo, filename=fname, revision=revision)
                hj = _read_safetensors_header(session, url)
                if hj is None:
                    continue
                for key in _EMBED_KEYS:
                    if key in hj:
                        shape = hj[key].get("shape") or []
                        dtype = hj[key].get("dtype") or "?"
                        if len(shape) >= 2:
                            return int(shape[0]), int(shape[1]), dtype
        return None
    except Exception as e:
        logger.warning(f"Embed shape probe failed for {model_repo}: {e}")
        return None


def detect_safetensors_quantization(model_repo: str, revision: str = None) -> Optional[dict]:
    """Detect quantized weights by scanning tensor-name signatures (bnb/GPTQ/AWQ)
    in the first safetensors shard's header. Returns
    {"quantized": True, "scheme": <label>, "marker": <name>} or None."""
    try:
        from huggingface_hub import hf_hub_url
        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            return None
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            url = hf_hub_url(repo_id=model_repo, filename=st_files[0], revision=revision)
            hj = _read_safetensors_header(session, url)
            if hj is None:
                return None
            for key in hj:
                if key == "__metadata__":
                    continue
                lower = key.lower()
                for marker in _QUANT_TENSOR_MARKERS:
                    if marker in lower:
                        if "absmax" in marker or "quant_map" in marker:
                            scheme = "bitsandbytes"
                        elif "qweight" in marker or "qzeros" in marker:
                            scheme = "gptq/awq"
                        elif "scb" in marker:
                            scheme = "bitsandbytes-int8"
                        else:
                            scheme = "unknown_quant"
                        return {"quantized": True, "scheme": scheme, "marker": key}
        return None
    except Exception as e:
        logger.warning(f"Quantization probe failed for {model_repo}: {e}")
        return None


def compute_model_hash(model_repo: str, revision: str = None) -> Optional[str]:
    """Combine SHA256 of every safetensors shard into a single hex digest;
    catches re-sharded copies with identical total weights."""
    import hashlib
    try:
        info = model_info(model_repo, revision=revision, files_metadata=True)
        # Collect SHA256 from ALL safetensors files (sorted by filename for stability)
        shard_hashes = []
        for sibling in sorted(info.siblings or [], key=lambda s: s.rfilename):
            if sibling.rfilename.endswith(".safetensors"):
                sha = None
                if hasattr(sibling, "lfs") and sibling.lfs:
                    sha = sibling.lfs.get("sha256") or sibling.lfs.get("oid")
                if not sha and hasattr(sibling, "blob_id") and sibling.blob_id:
                    sha = sibling.blob_id
                if sha:
                    shard_hashes.append(f"{sibling.rfilename}:{sha}")
        if not shard_hashes:
            return None
        # If single shard, return its hash directly (backward compatible)
        if len(shard_hashes) == 1:
            return shard_hashes[0].split(":", 1)[1]
        # Multiple shards: hash the combined sorted list
        combined = "\n".join(shard_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Hash computation failed for {model_repo}: {e}")
        return None


def check_duplicate_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> Optional[int]:
    """Return UID of the original submitter for ``model_hash`` if duplicate,
    else None. Stored in weight_hashes.json (separate from model_hashes.json
    so save_model_hashes() can't overwrite and break dedup)."""
    hash_file = state_dir / "weight_hashes.json"
    legacy_file = state_dir / "model_hashes.json"
    for f in (hash_file, legacy_file):
        if not f.exists():
            continue
        try:
            hashes = json.loads(f.read_text())
            for uid_str, stored_hash in hashes.items():
                if stored_hash == model_hash and int(uid_str) != miner_uid:
                    return int(uid_str)
        except Exception:
            continue
    return None


def register_model_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
):
    """Register a model (weight) hash for a miner UID."""
    state_dir.mkdir(parents=True, exist_ok=True)
    hash_file = state_dir / "weight_hashes.json"
    hashes = {}
    if hash_file.exists():
        try:
            hashes = json.loads(hash_file.read_text())
        except Exception:
            pass
    hashes[str(miner_uid)] = model_hash
    hash_file.write_text(json.dumps(hashes, indent=2))


def check_duplicate_content_hash(
    content_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> Optional[int]:
    """Find another UID with the same content (re-shard-invariant) hash."""
    f = state_dir / "model_content_hashes.json"
    if not f.exists():
        return None
    try:
        hashes = json.loads(f.read_text())
        for uid_str, stored_hash in hashes.items():
            if stored_hash == content_hash and int(uid_str) != miner_uid:
                return int(uid_str)
    except Exception:
        pass
    return None


def register_content_hash(
    content_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
):
    """Register a content hash for a miner UID."""
    state_dir.mkdir(parents=True, exist_ok=True)
    f = state_dir / "model_content_hashes.json"
    hashes = {}
    if f.exists():
        try:
            hashes = json.loads(f.read_text())
        except Exception:
            pass
    hashes[str(miner_uid)] = content_hash
    f.write_text(json.dumps(hashes, indent=2))


_CONTENT_HASH_BASE_TARGETS = {
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
}

_CONTENT_HASH_ATTENTION_TARGETS = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.q_b_proj.weight",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.kv_b_proj.weight",
}


def compute_content_hash(model_repo: str, revision: str = None, sample_tensors: int = 4) -> Optional[str]:
    """Shard-invariant content hash from raw bytes of a fixed tensor set.
    v2 includes at least one attention tensor so attention-only LoRA merges
    don't false-positive against the base model. Returns hex digest or None."""
    import struct
    try:
        from huggingface_hub import hf_hub_url
        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            return None
        targets = set(_CONTENT_HASH_BASE_TARGETS | _CONTENT_HASH_ATTENTION_TARGETS)
        tensor_hashes = []
        found_attention = False
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            for fname in st_files:
                if not targets:
                    break
                url = hf_hub_url(repo_id=model_repo, filename=fname, revision=revision)
                pr = session.get(url, headers={'Range': 'bytes=0-7'}, timeout=30,
                                 stream=True, allow_redirects=True)
                pr.raise_for_status()
                prefix = pr.raw.read(8); pr.close()
                if len(prefix) != 8:
                    continue
                header_size = struct.unpack('<Q', prefix)[0]
                if header_size <= 0 or header_size > 8_000_000:
                    continue
                header_len = 8 + header_size
                hr = session.get(url, headers={'Range': f'bytes=0-{header_len - 1}'},
                                 timeout=60, stream=True, allow_redirects=True)
                hr.raise_for_status()
                blob = hr.raw.read(header_len); hr.close()
                hj = json.loads(blob[8:header_len].decode('utf-8'))
                for tname, tinfo in hj.items():
                    if tname == '__metadata__' or tname not in targets:
                        continue
                    offs = tinfo.get('data_offsets') or [0, 0]
                    if len(offs) != 2:
                        continue
                    abs_start = header_len + offs[0]
                    abs_end = header_len + offs[1] - 1
                    if abs_end < abs_start:
                        continue
                    size = abs_end - abs_start + 1
                    if size <= 0 or size > 200_000_000:
                        continue
                    br = session.get(url, headers={'Range': f'bytes={abs_start}-{abs_end}'},
                                     timeout=120, stream=True, allow_redirects=True)
                    br.raise_for_status()
                    data = br.raw.read(size); br.close()
                    if len(data) != size:
                        continue
                    th = hashlib.sha256(data).hexdigest()
                    tensor_hashes.append(f"{tname}:{th}")
                    found_attention = found_attention or tname in _CONTENT_HASH_ATTENTION_TARGETS
                    targets.discard(tname)
        if not tensor_hashes or not found_attention:
            return None
        tensor_hashes.sort()
        return "v2:" + hashlib.sha256("\n".join(tensor_hashes).encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Content hash failed for {model_repo}: {e}")
        return None


def compute_tensor_metadata_hash(model_repo: str, revision: str = None) -> Optional[str]:
    """Shard-invariant hash of safetensors tensor metadata. Header-only fetch
    via HTTP range requests, no shard download."""
    try:
        from huggingface_hub import hf_hub_url
        import struct

        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            logger.warning(f"No safetensors files in {model_repo}")
            return None

        all_tensors = []
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            for fname in st_files:
                url = hf_hub_url(repo_id=model_repo, filename=fname, revision=revision)

                prefix_resp = session.get(
                    url,
                    headers={'Range': 'bytes=0-7'},
                    timeout=30,
                    stream=True,
                    allow_redirects=True,
                )
                prefix_resp.raise_for_status()
                prefix = prefix_resp.raw.read(8)
                prefix_resp.close()
                if len(prefix) != 8:
                    raise ValueError(f"Incomplete safetensors header prefix for {fname}")

                header_size = struct.unpack('<Q', prefix)[0]
                if header_size <= 0 or header_size > 8_000_000:
                    raise ValueError(f"Unexpected safetensors header size {header_size} for {fname}")

                header_len = 8 + header_size
                header_resp = session.get(
                    url,
                    headers={'Range': f'bytes=0-{header_len - 1}'},
                    timeout=60,
                    stream=True,
                    allow_redirects=True,
                )
                header_resp.raise_for_status()
                header_blob = header_resp.raw.read(header_len)
                header_resp.close()
                if len(header_blob) < header_len:
                    raise ValueError(f"Incomplete safetensors header body for {fname}")

                header_json = json.loads(header_blob[8:header_len].decode('utf-8'))
                for tensor_name, tensor_info in header_json.items():
                    if tensor_name == '__metadata__':
                        continue
                    dtype = tensor_info.get('dtype', '')
                    shape = tuple(tensor_info.get('shape', []))
                    all_tensors.append((tensor_name, shape, dtype))

        all_tensors.sort(key=lambda t: t[0])
        canonical = json.dumps(all_tensors, separators=(',', ':'), sort_keys=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Tensor metadata hash failed for {model_repo}: {e}")
        return None

# Cache positive integrity results; failures aren't cached so miners
# get fast re-admit after fixing configs.
_INTEGRITY_CACHE: dict = {}
_INTEGRITY_CACHE_TTL = 900  # 15 min


def verify_model_integrity(
    model_repo: str,
    revision: str = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """
    Pre-weight-setting integrity check:
    1. Model is still publicly accessible on HuggingFace
    2. Repo revision hasn't changed since commitment (git SHA match)
    3. Falls back to weight hash if no stored revision SHA

    Returns dict with:
      pass: bool
      reason: str
      current_hash: str or None  (git SHA of repo HEAD, or weight hash for legacy)
    """
    cache_key = (model_repo, revision or "", expected_hash or "")
    cached = _INTEGRITY_CACHE.get(cache_key)
    if cached is not None:
        result, ts = cached
        if (time.time() - ts) < _INTEGRITY_CACHE_TTL and result.get("pass") and not result.get("transient"):
            return dict(result)  # copy so callers can't mutate cache
    result = _verify_model_integrity_uncached(model_repo, revision, expected_hash)
    if result.get("pass") and not result.get("transient"):
        _INTEGRITY_CACHE[cache_key] = (dict(result), time.time())
    return result


def _verify_model_integrity_uncached(
    model_repo: str,
    revision: str = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """Inner impl — see ``verify_model_integrity`` for contract."""
    try:
        # 1. Check model is still public (HEAD request to repo)
        info = model_info(model_repo, revision=revision)
        if info.private:
            return {
                "pass": False,
                "reason": f"Model {model_repo} is now private — must be public for transparency",
                "current_hash": None,
            }
        if info.disabled:
            return {
                "pass": False,
                "reason": f"Model {model_repo} has been disabled on HuggingFace",
                "current_hash": None,
            }
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} no longer exists on HuggingFace (404)",
                "current_hash": None,
            }
        if "403" in err or "restricted" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} is restricted/gated — must be publicly accessible",
                "current_hash": None,
            }
        # Transient errors should not DQ
        err_lower = err.lower()
        if any(k in err_lower for k in ["429", "rate limit", "too many", "timeout", "503", "502", "connection"]):
            return {
                "pass": True,
                "reason": f"transient_error: {err}",
                "current_hash": None,
                "transient": True,
            }
        return {
            "pass": False,
            "reason": f"Cannot verify model accessibility: {err}",
            "current_hash": None,
        }

    # 2. Compare info.sha (git SHA of resolved revision) to expected.
    current_repo_sha = getattr(info, 'sha', None)

    if expected_hash and current_repo_sha:
        # Check if expected_hash looks like a git SHA (40 hex chars) vs weight hash
        is_git_sha = len(expected_hash) == 40 and all(c in '0123456789abcdef' for c in expected_hash)
        if is_git_sha:
            if current_repo_sha != expected_hash:
                return {
                    "pass": False,
                    "reason": f"Model repo has new commits since evaluation! revision {current_repo_sha[:12]}... ≠ expected {expected_hash[:12]}...",
                    "current_hash": current_repo_sha,
                }
            return {
                "pass": True,
                "reason": "ok",
                "current_hash": current_repo_sha,
            }

    # 3. Legacy path: fall back to weight hash comparison if no git SHA stored
    if expected_hash and not (len(expected_hash) == 40 and all(c in '0123456789abcdef' for c in expected_hash)):
        # expected_hash is a weight hash — use old method
        current_hash = compute_model_hash(model_repo, revision)
        if not current_hash:
            return {
                "pass": False,
                "reason": f"Cannot compute model hash — safetensors may have been removed",
                "current_hash": None,
            }
        if current_hash != expected_hash:
            return {
                "pass": False,
                "reason": f"Model weights changed since commitment! hash {current_hash[:16]}... ≠ expected {expected_hash[:16]}...",
                "current_hash": current_hash,
            }
        return {
            "pass": True,
            "reason": "ok",
            "current_hash": current_hash,
        }

    # 4. No expected hash — first check. Prefer git SHA over weight hash.
    if current_repo_sha:
        return {
            "pass": True,
            "reason": "ok",
            "current_hash": current_repo_sha,
        }

    # Fallback: compute weight hash for models where SHA is unavailable
    current_hash = compute_model_hash(model_repo, revision)
    if not current_hash:
        return {
            "pass": False,
            "reason": f"Cannot compute model hash — safetensors may have been removed",
            "current_hash": None,
        }
    return {
        "pass": True,
        "reason": "ok",
        "current_hash": current_hash,
    }


# Fixed test strings for tokenizer verification.
TOKENIZER_TEST_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "日本語のテスト文字列です。Unicode handling matters.",
    "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) for all x in vocabulary",
]
def is_allowed_student_arch(
    model_type: str | None,
    archs: list[str] | None,
) -> tuple[bool, str]:
    """Check (model_type, architectures) against subnet-config's allowlist.
    Returns (ok, label). Order: exact (mt, arch) pair, model_type only,
    architecture only."""
    archs = list(archs or [])
    mt = model_type or ""
    if not STUDENT_ARCH_ALLOWLIST and not STUDENT_ARCH_NAMES:
        return True, "no_allowlist_configured"
    for entry in STUDENT_ARCH_ALLOWLIST:
        if not isinstance(entry, dict):
            continue
        e_mt = entry.get("model_type")
        e_arch = entry.get("architecture")
        if e_mt and e_arch and mt == e_mt and e_arch in archs:
            return True, f"pair:{e_mt}/{e_arch}"
    if mt and mt in STUDENT_MODEL_TYPES:
        return True, f"model_type:{mt}"
    for a in archs:
        if a in STUDENT_ARCH_NAMES:
            return True, f"architecture:{a}"
    return False, f"not_in_allowlist:{mt}:{','.join(archs) if archs else 'none'}"


def assess_vllm_compatibility(config: dict, repo_info=None) -> tuple[bool, str]:
    """Soft check for native vLLM compatibility (allowlist-based).
    Returns (vllm_native, label); also reports preprocessor_config.json
    presence so the chat-king wrapper can stub vision-wrapped configs."""
    model_type = config.get("model_type")
    archs = config.get("architectures") or []
    preproc_present = False
    if repo_info is not None:
        try:
            preproc_present = any(
                getattr(s, "rfilename", "") == "preprocessor_config.json"
                for s in (repo_info.siblings or [])
            )
        except Exception:
            pass

    ok, label = is_allowed_student_arch(model_type, archs)
    if ok:
        suffix = "no_preproc" if not preproc_present else "with_preproc"
        return True, f"{label}:{suffix}"
    return False, f"unsupported_or_unknown:{model_type}:{','.join(archs) if archs else 'none'}"


_teacher_py_hashes_cache: Optional[dict[str, str]] = None


def _get_teacher_py_hashes() -> dict[str, str]:
    """Map {filename: sha256} for every .py file shipped by the current
    teacher repo (lazy; empty dict if none)."""
    global _teacher_py_hashes_cache
    if _teacher_py_hashes_cache is not None:
        return _teacher_py_hashes_cache
    hashes: dict[str, str] = {}
    try:
        teacher_info = model_info(TEACHER_MODEL, files_metadata=True)
        for sibling in (teacher_info.siblings or []):
            fname = getattr(sibling, "rfilename", "") or ""
            if not fname.endswith(".py"):
                continue
            try:
                path = hf_hub_download(repo_id=TEACHER_MODEL, filename=fname)
                with open(path, "rb") as f:
                    hashes[fname] = hashlib.sha256(f.read()).hexdigest()
            except Exception as exc:
                logger.warning(
                    f"Failed to hash teacher .py {fname}: {exc}"
                )
    except Exception as exc:
        logger.warning(f"Failed to enumerate teacher .py files: {exc}")
    _teacher_py_hashes_cache = hashes
    return hashes


def _filter_teacher_identical_py_files(
    model_repo: str, revision: Optional[str], fnames: list[str],
) -> list[str]:
    """Return the subset of ``fnames`` that are NOT byte-identical to the
    teacher's shipped version of the same filename (caller fails closed)."""
    teacher_hashes = _get_teacher_py_hashes()
    unsafe: list[str] = []
    for fname in fnames:
        base = fname.rsplit("/", 1)[-1]
        teacher_hash = teacher_hashes.get(fname) or teacher_hashes.get(base)
        if not teacher_hash:
            unsafe.append(fname)
            continue
        try:
            student_path = hf_hub_download(
                repo_id=model_repo, filename=fname, revision=revision,
            )
            with open(student_path, "rb") as f:
                student_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as exc:
            logger.warning(
                f"Failed to hash student .py {model_repo}:{fname}: {exc}"
            )
            unsafe.append(fname)
            continue
        if student_hash != teacher_hash:
            unsafe.append(fname)
    return unsafe


_teacher_chat_template_hash_cache: Optional[str] = None


def _teacher_chat_template_hash() -> str:
    """SHA256 of the teacher chat template (lazy).
    Checks tokenizer_config.json::chat_template, falls back to
    chat_template.jinja; returns a no-match sentinel if neither exists."""
    global _teacher_chat_template_hash_cache
    if _teacher_chat_template_hash_cache is not None:
        return _teacher_chat_template_hash_cache
    template = ""
    try:
        cfg_path = hf_hub_download(repo_id=TEACHER_MODEL, filename="tokenizer_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        tmpl = cfg.get("chat_template", "")
        if isinstance(tmpl, list):
            tmpl = json.dumps(tmpl)
        template = tmpl or ""
    except Exception as exc:
        logger.warning(f"Teacher tokenizer_config fetch failed: {exc}")
    if not template:
        try:
            jinja_path = hf_hub_download(repo_id=TEACHER_MODEL, filename="chat_template.jinja")
            with open(jinja_path) as f:
                template = f.read()
        except Exception:
            template = ""
    if not template:
        logger.warning(
            f"Teacher {TEACHER_MODEL} has no chat_template; "
            "using sentinel that no student can match."
        )
        _teacher_chat_template_hash_cache = "__NO_TEMPLATE_PRESENT__"
        return _teacher_chat_template_hash_cache
    h = hashlib.sha256(template.encode()).hexdigest()
    _teacher_chat_template_hash_cache = h
    return h


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error that should not DQ."""
    err_str = str(exc).lower()
    return any(k in err_str for k in [
        "429", "rate limit", "too many requests",
        "timeout", "timed out",
        "connection", "connectionerror", "connecttimeout",
    ])


_TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",      # HuggingFace fast tokenizer
    "tokenizer.model",     # SentencePiece
    "tiktoken.model",      # tiktoken (Kimi, GPT family)
    "vocab.json",          # GPT2 BPE
    "merges.txt",          # GPT2 BPE
    "added_tokens.json",   # added/special tokens
    "special_tokens_map.json",
    "bpe.codes",           # sentencepiece variant
)


def _teacher_tokenizer_artifact_hashes() -> dict[str, str]:
    """SHA256 of each tokenizer-artifact file the teacher ships
    (subset of _TOKENIZER_ARTIFACT_NAMES); the returned set is required."""
    hashes: dict[str, str] = {}
    try:
        from huggingface_hub import list_repo_files as _list
        teacher_files = set(_list(TEACHER_MODEL))
    except Exception as exc:
        logger.warning(f"Failed to list teacher files: {exc}")
        teacher_files = set()
    for name in _TOKENIZER_ARTIFACT_NAMES:
        if name not in teacher_files:
            continue
        try:
            p = hf_hub_download(repo_id=TEACHER_MODEL, filename=name)
            with open(p, "rb") as f:
                hashes[name] = hashlib.sha256(f.read()).hexdigest()
        except Exception as exc:
            logger.warning(f"Failed to hash teacher {name}: {exc}")
    return hashes


def verify_tokenizer_files(model_repo: str, revision: str = None) -> dict:
    """Byte-for-byte tokenizer-file verification against the teacher.
    Adapts to whichever tokenizer artifacts the teacher ships
    (tokenizer.json / tiktoken.model / BPE / etc) and also compares
    tokenizer_config.json (chat_template handled separately).
    Returns {match: bool, reason: str (if not matching)}."""
    teacher_hashes = _teacher_tokenizer_artifact_hashes()
    if not teacher_hashes:
        return {
            "match": False,
            "reason": (
                f"Teacher {TEACHER_MODEL} ships no recognised tokenizer "
                f"artifact — cannot perform byte-match verification."
            ),
        }
    for fname, teacher_hash in teacher_hashes.items():
        try:
            student_path = hf_hub_download(
                repo_id=model_repo, filename=fname, revision=revision,
            )
        except Exception as exc:
            if _is_transient_error(exc):
                return {
                    "match": True,
                    "transient": True,
                    "reason": f"transient fetching student {fname}: {exc}",
                }
            return {
                "match": False,
                "reason": (
                    f"Missing required tokenizer artifact "
                    f"'{fname}' on student — teacher ships it, student doesn't."
                ),
            }
        with open(student_path, "rb") as f:
            student_hash = hashlib.sha256(f.read()).hexdigest()
        if student_hash != teacher_hash:
            return {
                "match": False,
                "reason": (
                    f"{fname} mismatch: student hash {student_hash[:16]}... "
                    f"!= teacher hash {teacher_hash[:16]}... "
                    f"(tokenizer differs from {TEACHER_MODEL})"
                ),
            }

    # Check tokenizer_config.json (excluding chat_template)
    try:
        teacher_cfg_path = hf_hub_download(
            repo_id=TEACHER_MODEL, filename="tokenizer_config.json",
        )
        student_cfg_path = hf_hub_download(
            repo_id=model_repo, filename="tokenizer_config.json", revision=revision,
        )
    except Exception as exc:
        if _is_transient_error(exc):
            return {
                "match": True,
                "transient": True,
                "reason": f"transient fetching tokenizer_config.json: {exc}",
            }
        # If neither has a tokenizer_config, pass — some Kimi forks might omit.
        if "404" in str(exc) or "not found" in str(exc).lower():
            return {"match": True}
        return {"match": False, "reason": f"tokenizer_config.json fetch failed: {exc}"}

    with open(teacher_cfg_path) as f:
        teacher_cfg = json.load(f)
    with open(student_cfg_path) as f:
        student_cfg = json.load(f)

    # Remove chat_template from both (checked separately)
    teacher_cfg.pop("chat_template", None)
    student_cfg.pop("chat_template", None)

    if teacher_cfg != student_cfg:
        diff_keys = []
        all_keys = set(teacher_cfg.keys()) | set(student_cfg.keys())
        for k in sorted(all_keys):
            if teacher_cfg.get(k) != student_cfg.get(k):
                diff_keys.append(k)
        return {
            "match": False,
            "reason": (
                f"tokenizer_config.json mismatch (excluding chat_template): "
                f"differing fields: {', '.join(diff_keys[:10])}"
            ),
        }

    return {"match": True}


def verify_tokenizer_match(model_repo: str, revision: str = None) -> dict:
    """Verify the student tokenizer encodes test strings to the same IDs as
    the teacher. Short-circuits to match=True for tiktoken/sentencepiece
    teachers (handled byte-wise in verify_tokenizer_files).
    SKIP_TOKENIZER_ENCODING_CHECK=1 bypasses (not recommended)."""
    if os.environ.get("SKIP_TOKENIZER_ENCODING_CHECK") == "1":
        return {"match": True}

    from tokenizers import Tokenizer as RawTokenizer
    from huggingface_hub import hf_hub_download as _hf_dl

    try:
        from huggingface_hub import list_repo_files as _list
        teacher_files = set(_list(TEACHER_MODEL))
    except Exception:
        teacher_files = set()

    if "tokenizer.json" not in teacher_files:
        # tiktoken / sentencepiece path — the byte-match verification
        # upstream already proved student == teacher on the artifact file(s).
        return {"match": True}

    try:
        teacher_path = _hf_dl(TEACHER_MODEL, "tokenizer.json")
        teacher_tok = RawTokenizer.from_file(teacher_path)
        student_path = _hf_dl(model_repo, "tokenizer.json", revision=revision)
        student_tok = RawTokenizer.from_file(student_path)
    except Exception as exc:
        if _is_transient_error(exc):
            return {"match": True, "transient": True, "reason": str(exc)}
        return {"match": False, "reason": f"tokenizer.json load failed: {exc}"}

    for test_str in TOKENIZER_TEST_STRINGS:
        teacher_ids = teacher_tok.encode(test_str).ids
        student_ids = student_tok.encode(test_str).ids
        if teacher_ids != student_ids:
            return {
                "match": False,
                "reason": (
                    f"Encoding mismatch on test string: "
                    f"teacher produced {len(teacher_ids)} tokens, "
                    f"student produced {len(student_ids)} tokens"
                ),
            }

    return {"match": True}


def check_model_architecture(
    model_repo: str,
    revision: str = None,
    max_total_params_b: float = 3.5,
) -> dict:
    """Check if a model meets distillation subnet requirements (total params
    cap, vocab match, MoE active-param reporting).
    Returns {pass, reason, params_b, active_params_b, vocab_size, ...}."""
    try:
        # 0. SECURITY: reject repos with custom .py code (blocks json.dump
        # monkey-patches etc). Teacher-identical .py files are allowed so
        # legitimate Kimi students with tokenization_kimi.py still pass.
        info = None
        try:
            info = model_info(model_repo, revision=revision, files_metadata=True)
            dangerous_files = []
            for sibling in (info.siblings or []):
                fname = sibling.rfilename
                if fname.endswith('.py') and fname != '__init__.py':
                    dangerous_files.append(fname)
            if dangerous_files:
                surviving = _filter_teacher_identical_py_files(
                    model_repo, revision, dangerous_files,
                )
                if surviving:
                    return {
                        "pass": False,
                        "reason": (
                            f"SECURITY: Repo contains custom code files ("
                            f"{', '.join(surviving)}). Only .py files that are "
                            f"byte-identical to {TEACHER_MODEL}'s own .py files "
                            f"are allowed; anything else is custom code and is "
                            f"not permitted."
                        ),
                        "params_b": 0,
                    }
        except Exception as e:
            logger.warning(f"Could not check repo files for {model_repo}: {e}")

        # 0b. SECURITY: weight-file analysis (fake safetensors, hidden .bin
        # weights, size mismatches).
        MIN_MODEL_BYTES = 500_000_000  # 500MB
        MAX_MODEL_BYTES = max_total_params_b * 2.2e9  # ~2.2 bytes/param in bf16

        try:
            total_st_bytes = 0
            total_pt_bytes = 0
            st_files = []
            pt_files = []
            for sibling in (info.siblings or []):
                fname = sibling.rfilename
                fsize = 0
                if hasattr(sibling, 'size') and sibling.size is not None:
                    fsize = sibling.size
                elif hasattr(sibling, 'lfs') and sibling.lfs:
                    fsize = sibling.lfs.get('size', 0)

                if fname.endswith('.safetensors'):
                    total_st_bytes += fsize
                    st_files.append((fname, fsize))
                elif fname.endswith('.bin') and 'pytorch_model' in fname:
                    total_pt_bytes += fsize
                    pt_files.append((fname, fsize))

            # If both .safetensors and .bin are present, the larger is the
            # real weight set; tiny .safetensors next to huge .bin = evasion.
            if st_files and pt_files:
                if total_st_bytes < MIN_MODEL_BYTES and total_pt_bytes > MIN_MODEL_BYTES:
                    return {
                        "pass": False,
                        "reason": f"FRAUD: Tiny safetensors ({total_st_bytes:,}B) alongside large pytorch_model.bin "
                                  f"({total_pt_bytes:,}B). Real model hidden in .bin to bypass safetensors param check.",
                        "params_b": 0,
                    }

            total_weight_bytes = max(total_st_bytes, total_pt_bytes)
            if 0 < total_weight_bytes < MIN_MODEL_BYTES:
                return {
                    "pass": False,
                    "reason": f"FRAUD: Model weights total {total_weight_bytes:,} bytes — impossibly small "
                              f"(min {MIN_MODEL_BYTES:,} for a real model)",
                    "params_b": 0,
                }
            if total_weight_bytes > MAX_MODEL_BYTES:
                return {
                    "pass": False,
                    "reason": f"FRAUD: Model weights total {total_weight_bytes / 1e9:.1f}GB — too large for "
                              f"{max_total_params_b:.1f}B params (max ~{MAX_MODEL_BYTES / 1e9:.1f}GB in bf16)",
                    "params_b": total_weight_bytes / 2e9,  # rough estimate
                }

            # Reject .bin-only repos (modern HF uses safetensors; .bin-only
            # bypasses safetensors-metadata param counting).
            if pt_files and not st_files:
                return {
                    "pass": False,
                    "reason": f"Model uses pytorch_model.bin format only ({len(pt_files)} files, "
                              f"{total_pt_bytes / 1e9:.1f}GB). Safetensors format required — "
                              f"convert with `transformers` save_pretrained().",
                    "params_b": 0,
                }

        except Exception as e:
            logger.warning(f"Could not check weight file sizes for {model_repo}: {e}")

        safetensors_params_b = get_safetensors_param_count(model_repo, revision)
        config_path = hf_hub_download(
            repo_id=model_repo, filename="config.json", revision=revision,
        )
        with open(config_path) as f:
            config = json.load(f)
        moe_info = compute_moe_params(config)
        config_total_b = moe_info["total_params"] / 1e9
        config_active_b = moe_info["active_params"] / 1e9

        vllm_compatible, vllm_reason = assess_vllm_compatibility(config, info)
        # Prefer safetensors-verified param count (more accurate than config estimate).
        total_params_b = safetensors_params_b if safetensors_params_b > 0 else config_total_b

        if total_params_b <= 0:
            return {
                "pass": False,
                "reason": "Cannot determine parameter count — model may be missing safetensors metadata and config",
                "params_b": 0,
            }

        # 4. Total params (not active) -- prevents gaming with huge MoE.
        if total_params_b > max_total_params_b:
            return {
                "pass": False,
                "reason": f"Model too large: {total_params_b:.2f}B > {max_total_params_b:.1f}B max",
                "params_b": total_params_b,
                "active_params_b": config_active_b,
            }

        # 4b. Cross-validate: config param count vs actual file size
        # A real N-billion param model in bf16 should be ~2*N GB on disk.
        # If the config says 3B but files are 70GB, the config is lying.
        try:
            total_weight_bytes = 0
            for sibling in (info.siblings or []):
                fname = sibling.rfilename
                fsize = 0
                if hasattr(sibling, 'size') and sibling.size is not None:
                    fsize = sibling.size
                elif hasattr(sibling, 'lfs') and sibling.lfs:
                    fsize = sibling.lfs.get('size', 0)
                if fname.endswith('.safetensors') or (fname.endswith('.bin') and 'pytorch_model' in fname):
                    total_weight_bytes += fsize

            if total_weight_bytes > 0:
                # Estimate params from file size (bf16 = 2 bytes/param, fp32 = 4 bytes/param)
                estimated_params_from_size = total_weight_bytes / 2e9  # bf16 estimate
                # If file-estimated params are >2x the config-reported params, config is lying
                if estimated_params_from_size > total_params_b * 2.5:
                    return {
                        "pass": False,
                        "reason": f"FRAUD: Config claims {total_params_b:.2f}B params but weight files are "
                                  f"{total_weight_bytes / 1e9:.1f}GB (~{estimated_params_from_size:.1f}B params in bf16). "
                                  f"Config/weights mismatch — possible teacher model disguised as student.",
                        "params_b": estimated_params_from_size,
                    }
        except Exception as e:
            logger.warning(f"Config vs file size cross-validation failed: {e}")

        # 4c. Reject configs that need ``trust_remote_code=True`` to load.
        # The validator runs students with TRC=False; any ``auto_map`` block
        # at the top level OR in a nested sub-config (text_config /
        # vision_config) means Transformers will refuse the load and the
        # pod wastes ~30s + can crash with weird "NoneType is not iterable"
        # paths inside the model code. Examples blocked here today:
        # Godcat252/Besttop979 (text_config.auto_map → kimi_k25),
        # zdsxc/disti-v1, sangerno63/will_be_top, bodenmaurice/distil-new-v3.
        for _scope_label, _scope in (
            ("config", config),
            ("text_config", config.get("text_config") or {}),
            ("vision_config", config.get("vision_config") or {}),
        ):
            am = _scope.get("auto_map") if isinstance(_scope, dict) else None
            if am and isinstance(am, dict):
                return {
                    "pass": False,
                    "reason": (
                        f"{_scope_label}.auto_map is set ({sorted(am.keys())}); "
                        f"this requires trust_remote_code=True at load time, "
                        f"which the subnet does not allow for students. Native "
                        f"loading via model_type={config.get('model_type')} "
                        f"works without auto_map. Fix: delete the auto_map "
                        f"block from {_scope_label} in config.json — no weight "
                        f"or modeling changes are needed."
                    ),
                    "params_b": 0,
                }

        # 5. Reject quantized models (GPTQ, AWQ, GGUF, etc.)
        quant_config = config.get("quantization_config", {})
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown")
            return {
                "pass": False,
                "reason": f"Quantized model detected ({quant_method}) — subnet requires bf16/fp16 architecture distillation",
                "params_b": total_params_b,
            }

        # 6. Check vocab size (may be in text_config for multimodal/nested configs)
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)
        if vocab_size != BASELINE_VOCAB_SIZE:
            return {
                "pass": False,
                "reason": f"Vocab size mismatch: {vocab_size} ≠ {BASELINE_VOCAB_SIZE} (teacher)",
                "params_b": total_params_b,
                "vocab_size": vocab_size,
            }

        # 6b. Cross-check embed_tokens.weight shape/dtype against config
        # (header-only, free at precheck). Catches wrong vocab/hidden_size
        # or quantized integer embed.
        try:
            embed_info = get_embed_weight_shape(model_repo, revision)
            if embed_info is not None:
                embed_vocab, embed_hidden, embed_dtype = embed_info
                if embed_vocab != BASELINE_VOCAB_SIZE:
                    return {
                        "pass": False,
                        "reason": (
                            f"Embed table size {embed_vocab} ≠ {BASELINE_VOCAB_SIZE} "
                            f"(teacher). Config claims vocab_size={vocab_size} but "
                            f"actual weights are an older vocab — model would fail "
                            f"to load on the eval pod."
                        ),
                        "params_b": total_params_b,
                        "vocab_size": vocab_size,
                        "embed_vocab": embed_vocab,
                    }
                cfg_hidden = config.get("hidden_size") or config.get(
                    "text_config", {}
                ).get("hidden_size")
                if cfg_hidden and embed_hidden != int(cfg_hidden):
                    return {
                        "pass": False,
                        "reason": (
                            f"Embed hidden_size {embed_hidden} ≠ {int(cfg_hidden)} "
                            f"(declared in config). Mismatched packed/quantized "
                            f"embed table — would fail to load on the eval pod."
                        ),
                        "params_b": total_params_b,
                        "vocab_size": vocab_size,
                        "embed_hidden": embed_hidden,
                        "config_hidden": int(cfg_hidden),
                    }
                if embed_dtype.upper() not in _FLOAT_DTYPES:
                    return {
                        "pass": False,
                        "reason": (
                            f"Embed table dtype {embed_dtype} is not a valid float "
                            f"format (must be one of {','.join(_FLOAT_DTYPES)}). "
                            f"Looks like a packed/quantized embed without disclosed "
                            f"quantization_config — would fail to load on the eval pod."
                        ),
                        "params_b": total_params_b,
                        "embed_dtype": embed_dtype,
                    }
        except Exception as embed_err:
            logger.warning(f"Embed shape precheck error for {model_repo}: {embed_err}")

        # 6c. Detect quantized weights even when quantization_config was
        # stripped (bnb/GPTQ/AWQ marker tensors in the first shard).
        try:
            quant_info = detect_safetensors_quantization(model_repo, revision)
            if quant_info is not None:
                return {
                    "pass": False,
                    "reason": (
                        f"FRAUD: Quantized weights detected ({quant_info['scheme']} "
                        f"via tensor '{quant_info['marker']}') but config.json "
                        f"has no quantization_config. Subnet requires bf16/fp16 "
                        f"weights — strip quantization before re-uploading."
                    ),
                    "params_b": total_params_b,
                    "quant_scheme": quant_info["scheme"],
                }
        except Exception as quant_err:
            logger.warning(f"Quantization precheck error for {model_repo}: {quant_err}")

        # 7. Verify tokenizer encodings match the teacher.
        try:
            tokenizer_match = verify_tokenizer_match(model_repo, revision)
            if not tokenizer_match["match"]:
                return {
                    "pass": False,
                    "reason": f"Tokenizer encoding mismatch: {tokenizer_match['reason']}",
                    "params_b": total_params_b,
                    "vocab_size": vocab_size,
                }
        except Exception as tok_err:
            if _is_transient_error(tok_err):
                logger.warning(f"Tokenizer encoding check transient error for {model_repo}: {tok_err} (allowing)")
            else:
                return {
                    "pass": False,
                    "reason": f"Tokenizer encoding verification failed (fail-closed): {tok_err}",
                    "params_b": total_params_b,
                    "vocab_size": vocab_size,
                }

        # 8. Verify chat_template matches the teacher's (dynamic hash so a
        # teacher swap doesn't require code changes).
        REFERENCE_TEMPLATE_HASH = _teacher_chat_template_hash()
        try:
            import hashlib
            tok_config_path = hf_hub_download(
                repo_id=model_repo, filename="tokenizer_config.json", revision=revision,
            )
            with open(tok_config_path) as f:
                tok_config = json.load(f)
            student_template = tok_config.get("chat_template", "")
            if isinstance(student_template, list):
                student_template = json.dumps(student_template)

            # Also check standalone chat_template.jinja if tokenizer_config has no template
            if not student_template:
                try:
                    jinja_path = hf_hub_download(
                        repo_id=model_repo, filename="chat_template.jinja", revision=revision,
                    )
                    with open(jinja_path) as f:
                        student_template = f.read()
                except Exception:
                    pass  # No standalone template file

            if student_template:
                # Strip jinja comment watermarks like "{# model by ... #}"
                import re
                cleaned = re.sub(r'^\s*\{#.*?#\}\s*\n?', '', student_template, flags=re.MULTILINE).strip()
                template_hash = hashlib.sha256(cleaned.encode()).hexdigest()

                if template_hash != REFERENCE_TEMPLATE_HASH:
                    raw_hash = hashlib.sha256(student_template.encode()).hexdigest()
                    if raw_hash != REFERENCE_TEMPLATE_HASH:
                        return {
                            "pass": False,
                            "reason": (
                                f"Chat template does not match the teacher's canonical template. "
                                f"Students must use the {TEACHER_MODEL} chat template unmodified. "
                                f"(hash: {template_hash[:16]}... != expected {REFERENCE_TEMPLATE_HASH[:16]}...)"
                            ),
                            "params_b": total_params_b,
                            "vocab_size": vocab_size,
                        }
                    # Raw matches but cleaned doesn't — template has injected comments
                    logger.warning(f"Chat template for {model_repo} has injected comments but base template matches")
        except Exception as tmpl_err:
            logger.warning(f"Chat template check failed for {model_repo}: {tmpl_err} (allowing)")

        if moe_info["is_moe"]:
            logger.info(
                f"  MoE model: {moe_info['num_experts']} experts, "
                f"{moe_info['num_active_experts']} active/token, "
                f"total={total_params_b:.2f}B, active={config_active_b:.2f}B"
            )

        if not vllm_compatible:
            allowed_pairs = ", ".join(
                f"{e.get('model_type','?')}/{e.get('architecture','?')}"
                for e in STUDENT_ARCH_ALLOWLIST
                if isinstance(e, dict)
            ) or "(empty — subnet-config missing studentArchAllowlist)"
            return {
                "pass": False,
                "reason": (
                    f"Model architecture is not in the subnet allowlist. "
                    f"Found: {','.join(config.get('architectures', []))} "
                    f"(model_type={config.get('model_type', 'unknown')}). "
                    f"Allowed pairs (model_type/architecture): {allowed_pairs}. "
                    f"Fix: edit config.json on HuggingFace to use one of the "
                    f"allowed architectures (typically {TEACHER_MODEL}'s "
                    f"text-inner DeepseekV3ForCausalLM). No weight changes "
                    f"needed for a config-only fix."
                ),
                "params_b": total_params_b,
                "vllm_compatible": False,
                "vllm_reason": vllm_reason,
            }

        return {
            "pass": True,
            "reason": "ok",
            "params_b": total_params_b,
            "active_params_b": config_active_b,
            "vocab_size": vocab_size,
            "is_moe": moe_info["is_moe"],
            "vllm_compatible": vllm_compatible,
            "vllm_reason": vllm_reason,
        }

    except Exception as e:
        err_str = str(e).lower()
        is_transient = any(k in err_str for k in [  # do not DQ on transient
            "429", "rate limit", "too many requests",
            "connection", "timeout", "503", "502",
            "temporary", "unavailable",
        ])
        if is_transient:
            return {"pass": True, "reason": f"transient_error:{e}", "transient": True}
        return {"pass": False, "reason": f"check_failed:{e}"}
