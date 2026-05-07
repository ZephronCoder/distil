import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import model_checker


def _safetensors_blob(tensors: dict[str, bytes]) -> bytes:
    offset = 0
    header = {}
    data_parts = []
    for name, data in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [len(data) // 2],
            "data_offsets": [offset, offset + len(data)],
        }
        data_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts)


class _FakeResponse:
    def __init__(self, data: bytes):
        self.raw = SimpleNamespace(read=lambda n=-1: data if n < 0 else data[:n])

    def close(self):
        pass

    def raise_for_status(self):
        pass


class _FakeSession:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None, **_kwargs):
        path = url.removeprefix("hf://").rsplit("/", 1)[0]
        repo = path
        blob = self.blobs[repo]
        range_header = (headers or {}).get("Range")
        if not range_header:
            return _FakeResponse(blob)
        start_s, end_s = range_header.removeprefix("bytes=").split("-", 1)
        start = int(start_s)
        end = int(end_s)
        return _FakeResponse(blob[start : end + 1])


def _patch_hf(monkeypatch, blobs: dict[str, bytes]):
    monkeypatch.setattr(
        model_checker,
        "model_info",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="model.safetensors")]
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_url",
        lambda repo_id, filename, revision=None: f"hf://{repo_id}/{filename}",
    )
    monkeypatch.setattr(
        model_checker._requests,
        "Session",
        lambda: _FakeSession(blobs),
    )


def test_content_hash_includes_attention_tensor(monkeypatch):
    shared = {
        "model.layers.0.input_layernorm.weight": b"ln" * 8,
        "model.layers.0.mlp.down_proj.weight": b"mlp" * 8,
        "model.norm.weight": b"norm" * 8,
    }
    blobs = {
        "base/model": _safetensors_blob(
            {**shared, "model.layers.0.self_attn.q_proj.weight": b"attention-a" * 4}
        ),
        "lora/model": _safetensors_blob(
            {**shared, "model.layers.0.self_attn.q_proj.weight": b"attention-b" * 4}
        ),
    }
    _patch_hf(monkeypatch, blobs)

    base_hash = model_checker.compute_content_hash("base/model")
    lora_hash = model_checker.compute_content_hash("lora/model")

    assert base_hash and base_hash.startswith("v2:")
    assert lora_hash and lora_hash.startswith("v2:")
    assert base_hash != lora_hash


def test_content_hash_skips_when_attention_tensor_missing(monkeypatch):
    blobs = {
        "legacy/model": _safetensors_blob(
            {
                "model.layers.0.input_layernorm.weight": b"ln" * 8,
                "model.layers.0.mlp.down_proj.weight": b"mlp" * 8,
                "model.norm.weight": b"norm" * 8,
            }
        )
    }
    _patch_hf(monkeypatch, blobs)

    assert model_checker.compute_content_hash("legacy/model") is None
