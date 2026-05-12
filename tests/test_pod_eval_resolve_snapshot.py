"""Unit test for ``scripts.pod_eval_vllm.resolve_local_snapshot_path``.

Why this test exists: on 2026-05-11 09:42 UTC the Kimi K2.6 chat template
shipped a new tokenizer file (``tokenization_kimi_fast.py`` in snapshot
``81bcaaa``) whose ``__init__`` does ``os.path.join(model_root,
"tokenizer.json")`` and bails with ``ValueError: Missing tokenizer files
under: moonshotai/Kimi-K2.6`` if the bare HF repo name is passed in.
That crash aborted every eval round for ~12 hours. The fix added a
``resolve_local_snapshot_path`` helper that turns ``"moonshotai/
Kimi-K2.6"`` into the cached snapshot directory before
``AutoTokenizer.from_pretrained`` is called. This test pins that
behaviour (local-first, network fallback, original-name fallback) so it
can never silently regress.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pod_eval_vllm():
    """Import scripts/pod_eval_vllm.py without executing its main().

    The module is huge (~20k lines) and has heavy CUDA imports at top
    level. We swap out ``transformers``/``torch`` etc. with stubs first
    so the test stays fast on CPU-only CI.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False
        @staticmethod
        def current_device() -> int:
            return 0
        @staticmethod
        def device_count() -> int:
            return 0
        @staticmethod
        def empty_cache() -> None:
            return None
        @staticmethod
        def synchronize(_=None) -> None:
            return None
        @staticmethod
        def memory_allocated(_=None) -> int:
            return 0
        @staticmethod
        def memory_reserved(_=None) -> int:
            return 0
        @staticmethod
        def get_device_properties(_=None):
            class _P:
                total_memory = 0
                name = "stub"
            return _P()
    fake_torch.cuda = _NoCuda()
    sys.modules.setdefault("torch", fake_torch)

    spec = importlib.util.spec_from_file_location(
        "pod_eval_vllm_for_test",
        REPO_ROOT / "scripts" / "pod_eval_vllm.py",
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    except Exception:
        pass
    return mod


def test_resolve_returns_local_path_when_input_is_directory(tmp_path):
    """Local directories pass through unchanged so we don't waste a
    network round-trip when the operator already pointed us at a
    pre-staged snapshot."""
    mod = _load_pod_eval_vllm()
    out = mod.resolve_local_snapshot_path(str(tmp_path))
    assert out == str(tmp_path)


def test_resolve_uses_local_files_only_first():
    """Hot path: the eval pod prefetches every model up-front, so the
    cached snapshot is always available locally. The helper must hit
    ``snapshot_download(local_files_only=True)`` first and return its
    result, NEVER reaching for the network when the cache is warm."""
    mod = _load_pod_eval_vllm()
    fake = mock.MagicMock()
    fake.snapshot_download.return_value = "/cache/snap/abc"
    fake_pkg = types.ModuleType("huggingface_hub")
    fake_pkg.snapshot_download = fake.snapshot_download
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_pkg}):
        out = mod.resolve_local_snapshot_path("moonshotai/Kimi-K2.6")
    assert out == "/cache/snap/abc"
    fake.snapshot_download.assert_called_once()
    args, kwargs = fake.snapshot_download.call_args
    assert args == ("moonshotai/Kimi-K2.6",)
    assert kwargs.get("local_files_only") is True


def test_resolve_falls_back_to_network_when_cache_miss():
    """If ``local_files_only=True`` raises, the helper must retry with
    a network download (no ``local_files_only`` flag) so a
    fresh-bootstrap eval pod can hydrate the snapshot the first time."""
    mod = _load_pod_eval_vllm()
    seen = []
    def _fake(name, **kwargs):
        seen.append(("call", name, dict(kwargs)))
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not in cache")
        return "/cache/snap/network-fetched"
    fake_pkg = types.ModuleType("huggingface_hub")
    fake_pkg.snapshot_download = _fake
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_pkg}):
        out = mod.resolve_local_snapshot_path("moonshotai/Kimi-K2.6")
    assert out == "/cache/snap/network-fetched"
    assert len(seen) == 2
    assert seen[0][2].get("local_files_only") is True
    assert "local_files_only" not in seen[1][2]


def test_resolve_returns_original_when_both_paths_fail(capsys):
    """If even the network fetch fails (offline pod, HF outage), fall
    back to the original repo name and let ``from_pretrained`` follow
    its normal path. Logging the failure helps debug a future Kimi
    regression without silently swallowing the error."""
    mod = _load_pod_eval_vllm()
    def _fake(name, **kwargs):
        raise RuntimeError("hf down")
    fake_pkg = types.ModuleType("huggingface_hub")
    fake_pkg.snapshot_download = _fake
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_pkg}):
        out = mod.resolve_local_snapshot_path("moonshotai/Kimi-K2.6")
    assert out == "moonshotai/Kimi-K2.6"
    captured = capsys.readouterr()
    assert "snapshot_download failed" in captured.out


def test_resolve_handles_empty_input():
    mod = _load_pod_eval_vllm()
    assert mod.resolve_local_snapshot_path("") == ""
    assert mod.resolve_local_snapshot_path(None) is None


def test_resolve_passes_revision_when_pinned():
    """A non-main revision must be forwarded so revision-pinned evals
    don't accidentally pull from the moving HEAD."""
    mod = _load_pod_eval_vllm()
    seen = []
    def _fake(name, **kwargs):
        seen.append(dict(kwargs))
        return "/cache/snap/pinned"
    fake_pkg = types.ModuleType("huggingface_hub")
    fake_pkg.snapshot_download = _fake
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_pkg}):
        out = mod.resolve_local_snapshot_path(
            "moonshotai/Kimi-K2.6", revision="abc1234"
        )
    assert out == "/cache/snap/pinned"
    assert seen[0].get("revision") == "abc1234"


def test_resolve_skips_revision_when_main():
    """Default ``main`` is the repo's HEAD already and ``snapshot_download``
    treats absence-of-revision and ``revision="main"`` differently in
    the cache-key path. We pass nothing for ``main`` so the cache key
    matches the default-fetch path."""
    mod = _load_pod_eval_vllm()
    seen = []
    def _fake(name, **kwargs):
        seen.append(dict(kwargs))
        return "/cache/snap/main"
    fake_pkg = types.ModuleType("huggingface_hub")
    fake_pkg.snapshot_download = _fake
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_pkg}):
        mod.resolve_local_snapshot_path("moonshotai/Kimi-K2.6", revision="main")
    assert "revision" not in seen[0]
