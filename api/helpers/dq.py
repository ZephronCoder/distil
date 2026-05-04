"""Disqualification-reason lookup shared by miners + evaluation routes.

The lookup keying is canonically per-hotkey (2026-05-04). The
``commitment`` arg is no longer consulted but kept on the signature so
existing callers don't have to plumb it out yet.
"""


def _dq_reason_for_commitment(uid: int, hotkey: str | None, commitment: dict | None, dq: dict):
    """Resolve the DQ reason shown to API consumers for a UID's commitment.

    DQ scope is per-hotkey: once a hotkey is DQ'd, a new on-chain commit
    on the same hotkey does NOT clear the DQ — the miner needs a fresh
    registration to be re-evaluated.

    Lookup order:
      1. Bare hotkey (current policy).
      2. Any legacy ``hotkey:<block>`` entry (pre-2026-05-04 stores).
      3. UID string (very-legacy pre-hotkey-migration entries).
    """
    del commitment
    uid_str = str(uid)
    if hotkey and hotkey in dq:
        return dq.get(hotkey)
    if hotkey:
        prefix = f"{hotkey}:"
        legacy_reason = None
        for k, v in dq.items():
            if k.startswith(prefix):
                legacy_reason = v
        if legacy_reason is not None:
            return legacy_reason
    if uid_str in dq:
        return dq.get(uid_str)
    return None
