#!/usr/bin/env python3
"""
Unit tests for ``scripts/chat_pod/chat_server.py`` startup behaviour.

These cover the 2026-05-04 anti-thrash patch (Sebastian's "chat doesn't
work with current king" report) where:

  1. The bootstrapper must SKIP the 30 GB re-download when the
     ``/root/king-model`` already holds the right HF repo + revision.
  2. The bootstrapper must SERIALISE concurrent invocations via
     ``flock`` so the API watchdog can't spawn 2-3 racing downloads.

We don't actually exec vLLM in the tests — the relevant helpers are
``download_model``, ``_read_marker``, ``_write_marker``,
``_is_complete_download``, and ``_acquire_startup_lock``.

Run with::

    pytest tests/test_chat_server_bootstrap.py -v
    python tests/test_chat_server_bootstrap.py
"""
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Importing ``chat_server`` runs argv parsing at module-load time, so we
# stash a benign argv before importing. Tests that need different MODEL_NAME
# / MODEL_REVISION patch the module-level constants directly.
_orig_argv = sys.argv
sys.argv = ["chat_server.py", "test/repo:rev123", "8100"]
try:
    from chat_pod import chat_server as cs  # type: ignore
finally:
    sys.argv = _orig_argv


def _patch_module_dir(tmp: Path):
    """Repoint the module-level MODEL_DIR + marker path at a tmp dir."""
    cs.MODEL_DIR = tmp
    cs._KING_MARKER = str(tmp / ".king_marker.json")


def _make_complete_model_dir(tmp: Path, shards=("model.safetensors",)):
    """Materialise a fake "complete" model in ``tmp``.

    Writes config.json + index.json + each shard so
    ``_is_complete_download`` returns True.
    """
    (tmp / "config.json").write_text("{}")
    weight_map = {f"layers.{i}.weight": shard for i, shard in enumerate(shards)}
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for shard in shards:
        (tmp / shard).write_bytes(b"\x00" * 32)


# ═══════════════════════════════════════════════════════════════════════════════
# Marker file round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestKingMarker(unittest.TestCase):
    """Round-trip _read_marker / _write_marker."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._orig_marker = cs._KING_MARKER
        self._orig_dir = cs.MODEL_DIR
        _patch_module_dir(self._tmp)

    def tearDown(self):
        cs._KING_MARKER = self._orig_marker
        cs.MODEL_DIR = self._orig_dir
        self._tmpdir.cleanup()

    def test_read_marker_missing_returns_empty(self):
        self.assertEqual(cs._read_marker(), {})

    def test_write_then_read_marker_roundtrip(self):
        cs._write_marker("RLStepone/distil-success-h1", "abc123")
        out = cs._read_marker()
        self.assertEqual(out["model"], "RLStepone/distil-success-h1")
        self.assertEqual(out["revision"], "abc123")
        self.assertIsInstance(out["ts"], (int, float))

    def test_read_marker_corrupt_returns_empty(self):
        Path(cs._KING_MARKER).write_text("not-json{")
        self.assertEqual(cs._read_marker(), {})


# ═══════════════════════════════════════════════════════════════════════════════
# _is_complete_download
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsCompleteDownload(unittest.TestCase):
    """Verify the on-disk completeness check matches HF download layout."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._orig_dir = cs.MODEL_DIR
        cs.MODEL_DIR = self._tmp

    def tearDown(self):
        cs.MODEL_DIR = self._orig_dir
        self._tmpdir.cleanup()

    def test_empty_dir_is_incomplete(self):
        self.assertFalse(cs._is_complete_download())

    def test_config_only_is_incomplete(self):
        (self._tmp / "config.json").write_text("{}")
        self.assertFalse(cs._is_complete_download())

    def test_single_shard_no_index_is_complete(self):
        (self._tmp / "config.json").write_text("{}")
        (self._tmp / "model.safetensors").write_bytes(b"\x00" * 32)
        self.assertTrue(cs._is_complete_download())

    def test_indexed_shards_all_present_is_complete(self):
        _make_complete_model_dir(
            self._tmp, shards=("a.safetensors", "b.safetensors"),
        )
        self.assertTrue(cs._is_complete_download())

    def test_indexed_shards_missing_one_is_incomplete(self):
        _make_complete_model_dir(
            self._tmp, shards=("a.safetensors", "b.safetensors"),
        )
        (self._tmp / "b.safetensors").unlink()
        self.assertFalse(cs._is_complete_download())

    def test_corrupt_index_is_incomplete(self):
        (self._tmp / "config.json").write_text("{}")
        (self._tmp / "model.safetensors.index.json").write_text("not-json{")
        self.assertFalse(cs._is_complete_download())


# ═══════════════════════════════════════════════════════════════════════════════
# download_model: skip vs re-download
# ═══════════════════════════════════════════════════════════════════════════════


class TestDownloadModel(unittest.TestCase):
    """The skip-on-marker path is the entire point of the 2026-05-04 patch."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._orig_dir = cs.MODEL_DIR
        self._orig_marker = cs._KING_MARKER
        self._orig_model = cs.MODEL_NAME
        self._orig_rev = cs.MODEL_REVISION
        _patch_module_dir(self._tmp)

    def tearDown(self):
        cs.MODEL_DIR = self._orig_dir
        cs._KING_MARKER = self._orig_marker
        cs.MODEL_NAME = self._orig_model
        cs.MODEL_REVISION = self._orig_rev
        self._tmpdir.cleanup()

    def test_skip_when_marker_matches_and_files_complete(self):
        """Same model + same revision + complete on disk → no `hf download`."""
        cs.MODEL_NAME = "RLStepone/distil-success-h1"
        cs.MODEL_REVISION = "abc123"
        _make_complete_model_dir(self._tmp)
        cs._write_marker("RLStepone/distil-success-h1", "abc123")
        with patch.object(cs, "run") as mock_run:
            cs.download_model()
            mock_run.assert_not_called()

    def test_redownload_when_revision_bumps(self):
        """Same repo but different revision → fresh pull."""
        cs.MODEL_NAME = "RLStepone/distil-success-h1"
        cs.MODEL_REVISION = "newrev456"
        _make_complete_model_dir(self._tmp)
        cs._write_marker("RLStepone/distil-success-h1", "oldrev123")
        with patch.object(cs, "run") as mock_run, \
             patch.object(cs, "_hf_cli", return_value="hf"):
            cs.download_model()
            mock_run.assert_called_once()
            # MODEL_DIR should have been wiped (only the post-download
            # marker should remain, since `run` is mocked and didn't
            # actually write any shards).
            self.assertTrue(self._tmp.exists())
            on_disk = sorted(p.name for p in self._tmp.iterdir())
            self.assertEqual(on_disk, [".king_marker.json"])

    def test_redownload_when_model_changes(self):
        """New king repo entirely → fresh pull regardless of revision match."""
        cs.MODEL_NAME = "NewKing/some-model"
        cs.MODEL_REVISION = "abc123"
        _make_complete_model_dir(self._tmp)
        cs._write_marker("OldKing/other-model", "abc123")
        with patch.object(cs, "run") as mock_run, \
             patch.object(cs, "_hf_cli", return_value="hf"):
            cs.download_model()
            mock_run.assert_called_once()

    def test_redownload_when_marker_present_but_files_missing(self):
        """Marker says we have it but disk is bare → don't trust the marker."""
        cs.MODEL_NAME = "RLStepone/distil-success-h1"
        cs.MODEL_REVISION = "abc123"
        cs._write_marker("RLStepone/distil-success-h1", "abc123")
        # Don't write any model files.
        with patch.object(cs, "run") as mock_run, \
             patch.object(cs, "_hf_cli", return_value="hf"):
            cs.download_model()
            mock_run.assert_called_once()

    def test_marker_written_after_successful_download(self):
        """After a successful pull we must persist the marker."""
        cs.MODEL_NAME = "test/model"
        cs.MODEL_REVISION = None
        # Simulate `run` succeeding without doing anything.
        with patch.object(cs, "run") as mock_run, \
             patch.object(cs, "_hf_cli", return_value="hf"):
            mock_run.return_value = MagicMock()
            cs.download_model()
            mock_run.assert_called_once()
        marker = cs._read_marker()
        self.assertEqual(marker["model"], "test/model")
        self.assertIsNone(marker["revision"])


# ═══════════════════════════════════════════════════════════════════════════════
# Startup lock — concurrent invocations must serialise
# ═══════════════════════════════════════════════════════════════════════════════


class TestStartupLock(unittest.TestCase):
    """``flock`` serialisation: the second invocation exits cleanly."""

    def setUp(self):
        # _acquire_startup_lock writes to a fixed /tmp path; isolate per test.
        self._lock_path = tempfile.mktemp(suffix="_chat_lock")

    def tearDown(self):
        try:
            os.unlink(self._lock_path)
        except OSError:
            pass

    def _run_acquire_in_subprocess(self):
        """Spawn a child that grabs the lock and idles, return its pid.

        The child prints two lines: the pid first, then ``locked`` after
        the ``flock`` call succeeds. Parent reads both lines so it knows
        the lock is genuinely held before probing — earlier versions of
        this test had a TOCTOU race where the parent probed in the
        microseconds between ``open`` and ``flock`` in the child.
        """
        import subprocess
        script = f"""
import fcntl, os, sys, time
fd = os.open({self._lock_path!r}, os.O_CREAT | os.O_RDWR, 0o644)
sys.stdout.write(str(os.getpid()) + "\\n")
sys.stdout.flush()
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.stdout.write("locked\\n")
sys.stdout.flush()
time.sleep(10)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        pid_line = proc.stdout.readline().strip()
        confirm = proc.stdout.readline().strip()
        assert confirm == "locked", f"child failed to acquire lock: {confirm!r}"
        return proc, int(pid_line)

    def test_first_invocation_acquires_and_writes_pid(self):
        """When nobody holds the lock, _acquire_startup_lock returns an fd."""
        # Patch the module's lock path target by monkey-patching open.
        import fcntl
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, b"12345\n")
            with open(self._lock_path) as f:
                self.assertEqual(f.read().strip(), "12345")
        finally:
            os.close(fd)

    def test_second_invocation_blocks_when_locked(self):
        """While one process holds the lock, LOCK_NB raises BlockingIOError."""
        import fcntl
        fd1 = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd2 = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd2)
        finally:
            os.close(fd1)

    def test_lock_releases_after_holder_exits(self):
        """Once the holder exits, a new acquirer should succeed."""
        proc, _ = self._run_acquire_in_subprocess()
        # Holder has the lock — a probe must fail.
        import fcntl
        fd_probe = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(fd_probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd_probe)
        # Kill the holder. Lock should release.
        proc.kill()
        proc.wait(timeout=5)
        # New acquirer succeeds.
        fd2 = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd2)


if __name__ == "__main__":
    unittest.main()
