#!/usr/bin/env python3
"""One-shot cleanup for the activation-copy DQ bug (Apr 18).

What this fixes:
  1. Backfills `commit_block` into `activation_fingerprints.json` entries that
     were stored before the field was added. Uses `model_hashes.json` (which
     records `{uid}_block`) as the source of truth.
  2. Clears activation-copy DQs where the DQ'd UID actually committed EARLIER
     than the claimed "original". These DQs came from the broken fast-path in
     precheck.py that passed commit_block=None to `check_activation_fingerprint`,
     which then defaulted to infinity and always marked the incoming UID as later.

Run once, then restart the validator.

Idempotent: safe to run twice. Makes a .bak file before any write.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
STATE_DIR = Path(_positional[0]) if _positional else Path("/opt/distil/repo/state")

FP_FILE = STATE_DIR / "activation_fingerprints.json"
DQ_FILE = STATE_DIR / "disqualified.json"
MODEL_HASHES_FILE = STATE_DIR / "model_hashes.json"
UID_HOTKEY_FILE = STATE_DIR / "uid_hotkey_map.json"
SCORES_FILE = STATE_DIR / "scores.json"


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _backup_and_save(path: Path, data, tag: str):
    if DRY_RUN:
        print(f"  [dry-run] would save {path}")
        return
    bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path} (backup: {bak.name}) — {tag}")


def backfill_fingerprint_blocks():
    print("[1/2] Backfilling commit_block into activation_fingerprints.json")
    fps = _load(FP_FILE)
    mh = _load(MODEL_HASHES_FILE) or {}
    if not fps:
        print("  no fingerprints file — nothing to do")
        return {}
    changed = 0
    resolved: dict[int, int] = {}
    for uid_str, entry in fps.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if entry.get("commit_block") is not None:
            try:
                resolved[uid] = int(entry["commit_block"])
            except (TypeError, ValueError):
                pass
            continue
        block = mh.get(f"{uid}_block")
        if block is None:
            print(f"  UID {uid}: no commit_block resolvable (not in model_hashes)")
            continue
        entry["commit_block"] = int(block)
        resolved[uid] = int(block)
        changed += 1
        print(f"  UID {uid}: backfilled commit_block={block}")
    if changed:
        _backup_and_save(FP_FILE, fps, f"backfilled {changed} entries")
    else:
        print("  no entries needed backfilling")
    return resolved


_ACT_COPY_RE = re.compile(
    r"copy: activation-space duplicate of UID (\d+) \(([^)]+)\) — cosine similarity"
)


def clear_wrong_copy_dqs(uid_to_block: dict[int, int]):
    print("[2/2] Clearing activation-copy DQs with inverted commit_block ordering")
    dq = _load(DQ_FILE)
    if not dq:
        print("  no disqualified file — nothing to do")
        return
    uid_hk = _load(UID_HOTKEY_FILE) or {}
    hk_to_uid: dict[str, int] = {}
    for uid_s, hk in uid_hk.items():
        try:
            hk_to_uid[hk] = int(uid_s)
        except (TypeError, ValueError):
            continue
    to_remove: list[tuple[str, int, int, int]] = []
    for key, reason in list(dq.items()):
        if not isinstance(reason, str):
            continue
        match = _ACT_COPY_RE.search(reason)
        if not match:
            continue
        orig_uid = int(match.group(1))
        if ":" not in key:
            continue
        hk, block_str = key.rsplit(":", 1)
        try:
            dq_block = int(block_str)
        except (TypeError, ValueError):
            continue
        dq_uid = hk_to_uid.get(hk)
        if dq_uid is None:
            continue
        orig_block = uid_to_block.get(orig_uid)
        if orig_block is None:
            continue
        if dq_block < orig_block:
            to_remove.append((key, dq_uid, dq_block, orig_block))
            print(
                f"  WRONG DQ: UID {dq_uid} (block {dq_block}) was flagged as copy of "
                f"UID {orig_uid} (block {orig_block}). {dq_uid} committed first — clearing."
            )
        else:
            print(
                f"  OK:        UID {dq_uid} (block {dq_block}) copy of "
                f"UID {orig_uid} (block {orig_block}) — keeping."
            )
    if not to_remove:
        print("  nothing to clear")
        return
    scores = _load(SCORES_FILE) or {}
    scores_changed = False
    for key, dq_uid, _, _ in to_remove:
        dq.pop(key, None)
        s = scores.get(str(dq_uid))
        if s is not None and s > 1.0:
            print(f"  UID {dq_uid}: also clearing penalty score {s} (from the wrong DQ)")
            scores.pop(str(dq_uid), None)
            scores_changed = True
    _backup_and_save(DQ_FILE, dq, f"cleared {len(to_remove)} wrong DQs")
    if scores_changed:
        _backup_and_save(SCORES_FILE, scores, "cleared penalty scores tied to wrong DQs")


if __name__ == "__main__":
    print(f"state_dir = {STATE_DIR}  dry_run = {DRY_RUN}")
    resolved = backfill_fingerprint_blocks()
    clear_wrong_copy_dqs(resolved)
    print("done")
