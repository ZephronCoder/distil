import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.pod import PodManager


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeChannel:
    def settimeout(self, _timeout):
        pass

    def recv_ready(self):
        return False

    def recv_stderr_ready(self):
        return False

    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return 0


class _FakeStream:
    def __init__(self):
        self.channel = _FakeChannel()


class _FakeClient:
    def __init__(self):
        self.commands = []
        self.stdin = _FakeStdin()

    def get_transport(self):
        return None

    def exec_command(self, command):
        self.commands.append(command)
        return self.stdin, _FakeStream(), _FakeStream()


class _FakeConnection:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *args):
        return False


class _FakeLium:
    def __init__(self, client):
        self.client = client

    def ssh_connection(self, _pod, timeout=30):
        return _FakeConnection(self.client)


def test_exec_feeds_env_over_stdin_not_command_line():
    client = _FakeClient()
    pod = PodManager(_FakeLium(client), pod_name="test")
    pod.pod = object()

    result = pod.exec("echo ok", env={"SENSITIVE_ENV": "sensitive-value"}, timeout=5)

    assert result["success"]
    assert client.commands == ["bash -s"]
    script = "".join(client.stdin.writes)
    assert "SENSITIVE_ENV=sensitive-value" in script
    assert "echo ok" in script
    assert "sensitive-value" not in client.commands[0]


def test_exec_without_env_uses_command_directly():
    client = _FakeClient()
    pod = PodManager(_FakeLium(client), pod_name="test")
    pod.pod = object()

    result = pod.exec("echo ok", timeout=5)

    assert result["success"]
    assert client.commands == ["echo ok"]
    assert client.stdin.writes == []
