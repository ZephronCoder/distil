"""
Scoring logic: winner-take-all weights.

- Winner-take-all: best KL miner gets ALL the weight (1.0), everyone else gets 0.0
- No EMA — models are permanently committed, scores converge naturally
- Quality floor: KL > threshold gets zero weight
- All state persisted to disk for restart survival
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("distillation.scoring")

STATE_DIR = Path("state")
DEFAULT_MAX_KL = 2.0


def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on missing/corrupt files."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _sanitize_for_json(obj):
    """Replace inf/nan floats with None so JSON serialization never fails."""
    import math
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _save_json(path: Path, data: dict):
    """Save data as JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(data), indent=2))


# ── Scores ────────────────────────────────────────────────────────────────


def load_scores(state_dir: Path = STATE_DIR) -> dict[str, float]:
    """Load KL scores. Keys are string UIDs."""
    return _load_json(state_dir / "scores.json")


def save_scores(scores: dict[str, float], state_dir: Path = STATE_DIR):
    """Persist KL scores to disk."""
    _save_json(state_dir / "scores.json", scores)


# ── Disqualification Tracking ─────────────────────────────────────────────


def load_disqualified(state_dir: Path = STATE_DIR) -> dict[str, str]:
    """Load disqualification reasons. Keys are hotkeys (ss58), values are reason strings.
    Legacy entries keyed by UID are preserved but should be migrated."""
    return _load_json(state_dir / "disqualified.json")


def save_disqualified(dq: dict[str, str], state_dir: Path = STATE_DIR):
    """Persist disqualification reasons to disk."""
    _save_json(state_dir / "disqualified.json", dq)


def disqualify(hotkey: str, reason: str, dq: dict[str, str],
               coldkey: str = None, hf_username: str = None,
               commit_block: int = None):
    """Record a disqualification keyed on the bare hotkey.

    2026-05-04 — DQ scope changed from per-(hotkey, commit_block) to
    per-hotkey. Rationale: a miner whose model definitively misbehaved
    (cheating, gibberish output, prohibited custom-code config, identical-
    to-teacher fraud, etc.) should not be able to bypass the DQ by
    pushing a new on-chain commit on the same hotkey. The new policy:
    once a hotkey is DQ'd, the miner needs a fresh on-chain registration
    (= new hotkey) to be re-evaluated. This burns ~0.5 TAO so it's a
    real cost, but it caps validator pod time wasted on repeat-offender
    hotkeys and pushes the cost of bad submissions onto the submitter.

    The ``commit_block`` parameter is kept on the signature for
    backward-compat with existing callers but is no longer part of the
    storage key — it's accepted and ignored. Existing entries that use
    the legacy ``hotkey:block`` key format are honoured by
    :func:`is_disqualified` via prefix-stripping; we don't migrate them
    on disk to keep the historical record auditable.

    Optionally flags the coldkey and HF username as suspicious.
    These flags don't auto-DQ (to avoid false positives on shared orgs)
    but trigger enhanced scrutiny on future submissions.
    """
    # commit_block intentionally ignored — kept on signature for
    # caller-side backward-compat; the storage key is the bare hotkey.
    del commit_block
    dq[hotkey] = reason
    # NOTE: No coldkey or HF username flags — policy is per-hotkey ONLY.
    # Miners can register a NEW hotkey to retry; the DQ does not propagate
    # across coldkeys. HF username flags removed — anyone can commit any
    # HF account name, so flagging by HF username punishes innocent people.
    # Only hotkey is cryptographically tied to the on-chain committer.


def _legacy_hotkey_dq_keys(hotkey: str, dq: dict[str, str]):
    """Yield any legacy ``hotkey:<commit_block>`` entries in ``dq``.

    Pre-2026-05-04 the DQ store keyed on ``hotkey:block`` so the same
    hotkey could carry multiple entries (one per misbehaving commit).
    The new per-hotkey policy uses the bare hotkey as the key, but we
    must still recognise legacy entries during lookup so historical
    DQ'd hotkeys remain DQ'd. Iterating once over ``dq`` is fine —
    typical ``disqualified.json`` has <500 entries.
    """
    prefix = f"{hotkey}:"
    for k in dq:
        if k.startswith(prefix):
            yield k


def is_disqualified(uid: int, hotkey: str, dq: dict[str, str],
                    commit_block: int = None, **kwargs) -> bool:
    """Check if a hotkey is disqualified.

    2026-05-04 — DQ scope is now per-hotkey (see :func:`disqualify`).
    A new on-chain commit on the same hotkey does NOT clear the DQ;
    the miner must register a new hotkey to be re-evaluated.

    Lookup order:
      1. Bare hotkey (current per-hotkey policy).
      2. Any legacy ``hotkey:<block>`` entry (pre-2026-05-04 stores).
      3. UID string (very-legacy entries from before hotkey migration).

    The ``commit_block`` parameter is kept on the signature for
    backward-compat but no longer affects the lookup result — once a
    hotkey is in ``dq`` for any reason at any commit, it stays DQ'd.
    """
    del commit_block, kwargs
    if hotkey and hotkey in dq:
        return True
    if hotkey and any(True for _ in _legacy_hotkey_dq_keys(hotkey, dq)):
        return True
    if str(uid) in dq:
        return True
    return False


def is_flagged(coldkey: str = None, hf_username: str = None,
               dq: dict[str, str] = None) -> str | None:
    """Check if a coldkey or HF username is flagged as suspicious.
    Returns the flag reason if flagged, None otherwise.
    Flagged miners aren't auto-DQ'd but get logged for scrutiny."""
    if dq is None:
        return None
    if coldkey and f"flag:coldkey:{coldkey}" in dq:
        return dq[f"flag:coldkey:{coldkey}"]
    # HF username flags removed — not cryptographically tied to committer
    return None


def get_dq_reason(uid: int, hotkey: str, dq: dict[str, str],
                  commit_block: int = None, **kwargs) -> str:
    """Resolve the DQ reason for a hotkey.

    2026-05-04 — Mirrors :func:`is_disqualified` lookup order. The
    ``commit_block`` parameter is kept on the signature but ignored.
    Legacy ``hotkey:<block>`` entries are honoured; if multiple legacy
    entries exist for the same hotkey, the most recently inserted one
    wins (Python dict insertion order).
    """
    del commit_block, kwargs
    if hotkey and hotkey in dq:
        return dq[hotkey]
    if hotkey:
        last_legacy = ""
        for k in _legacy_hotkey_dq_keys(hotkey, dq):
            last_legacy = dq[k]
        if last_legacy:
            return last_legacy
    return dq.get(str(uid), "")


# ── Failure Tracking ──────────────────────────────────────────────────────


def load_failures(state_dir: Path = STATE_DIR) -> dict[str, int]:
    """Load failure counts per UID."""
    return _load_json(state_dir / "failures.json")


def save_failures(failures: dict[str, int], state_dir: Path = STATE_DIR):
    """Persist failure counts to disk."""
    _save_json(state_dir / "failures.json", failures)


def record_failure(uid: int, failures: dict[str, int], failure_models: dict[str, str] = None,
                   model_name: str = None) -> int:
    """Increment and return failure count for a UID.
    
    Optionally tracks which model caused the failure so we can reset
    the counter when a miner updates their commitment to a new model.
    """
    uid_str = str(uid)
    failures[uid_str] = failures.get(uid_str, 0) + 1
    if failure_models is not None and model_name:
        failure_models[uid_str] = model_name
    return failures[uid_str]


def reset_failures(uid: int, failures: dict[str, int]):
    """Clear failure count for a UID after a successful eval."""
    failures.pop(str(uid), None)


def is_stale(uid: int, failures: dict[str, int], max_failures: int = 3) -> bool:
    """Check if a UID has exceeded the maximum failure count."""
    return failures.get(str(uid), 0) >= max_failures


# ── Score History ──────────────────────────────────────────────────────────


def load_score_history(state_dir: Path = STATE_DIR) -> list[dict]:
    """Load score history array from disk."""
    path = state_dir / "score_history.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def append_score_history(
    block: int,
    timestamp: float,
    scores: dict[str, float],
    king_uid: int | None,
    state_dir: Path = STATE_DIR,
    max_entries: int = 500,
    uid_to_hotkey: dict[int, str] | None = None,
):
    """Append a score snapshot to history, capping at max_entries.

    Optionally includes uid_to_hotkey mapping so score provenance is
    traceable even after UID recycling."""
    history = load_score_history(state_dir)
    entry = {
        "block": block,
        "timestamp": timestamp,
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "king_uid": king_uid,
    }
    # Include hotkey mapping for score traceability
    if uid_to_hotkey:
        # Only include hotkeys for UIDs that have scores in this entry
        entry["hotkeys"] = {
            k: uid_to_hotkey.get(int(k), "")[:12]
            for k in scores if int(k) in uid_to_hotkey
        }
    history.append(entry)
    if len(history) > max_entries:
        history = history[-max_entries:]
    path = state_dir / "score_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    logger.info(f"Score history: {len(history)} entries (block {block})")
