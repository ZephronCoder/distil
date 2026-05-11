"""Scoring helpers: per-UID KL scores, DQ tracking, failure counters.

Winner-take-all weighting is computed at emit time; this module just
stores the inputs."""
import json
import logging
from pathlib import Path

from eval.state import _load_json, _sanitize_for_json, _save_json

logger = logging.getLogger("distillation.scoring")

STATE_DIR = Path("state")
DEFAULT_MAX_KL = 2.0

__all__ = ["_load_json", "_sanitize_for_json", "_save_json"]


# ── Scores ────────────────────────────────────────────────────────────────


def load_scores(state_dir: Path = STATE_DIR) -> dict[str, float]:
    """Load KL scores. Keys are string UIDs."""
    return _load_json(state_dir / "scores.json")


def save_scores(scores: dict[str, float], state_dir: Path = STATE_DIR):
    """Persist KL scores to disk."""
    _save_json(state_dir / "scores.json", scores)


# ── Disqualification Tracking ─────────────────────────────────────────────


def load_disqualified(state_dir: Path = STATE_DIR) -> dict[str, str]:
    """Load DQ reasons keyed by hotkey (legacy UID-keyed entries preserved)."""
    return _load_json(state_dir / "disqualified.json")


def save_disqualified(dq: dict[str, str], state_dir: Path = STATE_DIR):
    """Persist disqualification reasons to disk."""
    _save_json(state_dir / "disqualified.json", dq)


def disqualify(hotkey: str, reason: str, dq: dict[str, str],
               coldkey: str = None, hf_username: str = None,
               commit_block: int = None):
    """Record a DQ keyed on the hotkey. A new commit on the same hotkey
    does NOT clear the DQ; the miner must register a new hotkey.

    ``coldkey``/``hf_username``/``commit_block`` are accepted for
    backward-compat but ignored: only the hotkey is cryptographically
    tied to the committer."""
    del commit_block
    dq[hotkey] = reason


def _legacy_hotkey_dq_keys(hotkey: str, dq: dict[str, str]):
    """Yield legacy ``hotkey:<commit_block>`` entries in ``dq``."""
    prefix = f"{hotkey}:"
    for k in dq:
        if k.startswith(prefix):
            yield k


def is_disqualified(uid: int, hotkey: str, dq: dict[str, str],
                    commit_block: int = None, **kwargs) -> bool:
    """Check DQ. Lookup: bare hotkey, then legacy hotkey:block, then UID."""
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
    """Return the flag reason for a coldkey, or None. Used for logging only."""
    if dq is None:
        return None
    if coldkey and f"flag:coldkey:{coldkey}" in dq:
        return dq[f"flag:coldkey:{coldkey}"]
    return None


def get_dq_reason(uid: int, hotkey: str, dq: dict[str, str],
                  commit_block: int = None, **kwargs) -> str:
    """Return the DQ reason; same lookup order as :func:`is_disqualified`."""
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
    """Increment and return failure count; optionally tracks model name."""
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
    """Append a score snapshot (capped at ``max_entries``); records hotkeys
    for traceability across UID recycling."""
    history = load_score_history(state_dir)
    entry = {
        "block": block,
        "timestamp": timestamp,
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "king_uid": king_uid,
    }
    if uid_to_hotkey:
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
