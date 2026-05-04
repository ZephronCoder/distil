"""Test isolation helpers.

A handful of unit tests (test_aime_paraphrase, test_math_paraphrase,
test_judge_injection_defense, …) install a stub ``torch`` module into
``sys.modules`` so they can exercise paraphrase / scoring helpers
without dragging in the real PyTorch dependency. The stub never
restored the original torch module, so any test that later tried to
import ``torch.Tensor`` (e.g. test_kl_scoring_edge_cases) failed
non-deterministically depending on collection order.

This conftest eagerly imports the real torch / transformers / bittensor
modules at session start (so they're cached in sys.modules), takes a
snapshot, and reinstalls the snapshot before every test. Tests that
intentionally rely on the stub re-install it within their own setUp /
module-level invocation; tests that need the real torch see the real
torch.
"""
from __future__ import annotations

import sys

import pytest

# Eagerly import the real modules now, BEFORE any test file's
# module-level stub installer runs. Wrapped in try so a missing
# transformers/bittensor in a stripped-down dev env doesn't break the
# test session.
try:
    import torch  # noqa: F401
    import torch.nn  # noqa: F401
    import torch.nn.functional  # noqa: F401
except Exception:
    pass
try:
    import transformers  # noqa: F401
except Exception:
    pass
try:
    import bittensor  # noqa: F401
except Exception:
    pass

_PRESERVED_MODULES = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    "torch.random",
    "transformers",
    "bittensor",
)


def _snapshot() -> dict:
    return {name: sys.modules[name] for name in _PRESERVED_MODULES if name in sys.modules}


_REAL_MODULES = _snapshot()


@pytest.fixture(autouse=True)
def restore_real_modules():
    """Reinstall the real torch/transformers/bittensor modules BEFORE
    each test so per-test stubs (installed at module import time during
    pytest's collection pass) can't bleed into subsequent tests.

    Tests that explicitly need the stub re-install it inside their own
    setUp / module-level invocation; this fixture runs first so the
    stub-using tests still see their stub by the time they execute,
    while real-torch tests get the real module restored.

    We also evict any ``eval.*`` / ``distillation.*`` module from the
    cache so that re-importing them inside a real-torch test gets a
    fresh module bound to the real torch (otherwise the stale cached
    module captured the stub at first import and re-imports return the
    same cached object).
    """
    for name, mod in _REAL_MODULES.items():
        sys.modules[name] = mod
    for name in list(sys.modules):
        if name.startswith("eval.") or name.startswith("distillation.") or name == "eval":
            del sys.modules[name]
    yield
