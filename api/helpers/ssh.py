"""SSH execution for chat pod communication."""

import subprocess

from ..config import CHAT_POD_HOST, CHAT_POD_SSH_PORT, CHAT_POD_SSH_KEY


def _ssh_exec(cmd: str, timeout: int = 30) -> str:
    """Execute command on chat pod via SSH. Returns stdout."""
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
        f"root@{CHAT_POD_HOST}", cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout
