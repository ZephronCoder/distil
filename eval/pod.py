"""Pod lifecycle management: connect, install deps, transfer files,
exec commands, and clean up disk on a Lium GPU pod."""
import json
import os
import re
import shlex
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("distillation.pod")

# Patterns for sanitizing GPU logs before public exposure
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SECRET_PATTERNS = re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}')
_SENSITIVE_KEYWORDS = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret")


def sanitize_gpu_log(raw: str) -> str:
    """Strip ANSI codes, secrets, and SSH noise from GPU pod logs before writing to disk."""
    lines = []
    for line in raw.splitlines():
        cleaned = _ANSI_RE.sub('', line).strip()
        if not cleaned:
            continue
        if any(kw in cleaned for kw in _SENSITIVE_KEYWORDS):
            continue
        if any(noise in cleaned for noise in (
            "sftp", "Authentication", "Connected (version", "chan ",
            "Opened sftp", "sftp session closed",
        )):
            continue
        cleaned = _SECRET_PATTERNS.sub('[REDACTED]', cleaned)
        lines.append(cleaned)
    return '\n'.join(lines)


def _retry(fn, max_attempts: int = 3, delay: float = 5.0, label: str = "operation"):
    """Retry ``fn`` up to ``max_attempts`` times with linear backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{label} failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


class PodManager:
    """Lium GPU pod handle: discovery, deps, file IO, exec, cleanup."""

    def __init__(self, lium, pod_name: str = "distil-validator"):
        self.lium = lium
        self.pod_name = pod_name
        self.pod = None
        # Serialise SFTP/SSH; concurrent sessions can drop long eval channels.
        self._io_lock = threading.Lock()

    def connect(self):
        """Find and attach to the named Lium pod (raises if missing)."""
        pods = self.lium.ps()
        for p in pods:
            if self.pod_name in p.name:
                self.pod = p
                logger.info(f"Connected to pod: {p.name} ({p.id[:12]})")
                return
        available = [p.name for p in pods]
        raise RuntimeError(f"Pod '{self.pod_name}' not found. Available: {available}")

    def reconnect(self):
        """Re-discover the pod. Use after network failures or pod restarts."""
        logger.info("Reconnecting to pod...")
        self.pod = None
        self.connect()

    def upload(self, local: str, remote: str, max_attempts: int = 5):
        """Upload a file to the pod with retries."""
        def _do():
            with self._io_lock:
                self.lium.upload(self.pod, local=local, remote=remote)
        _retry(_do, max_attempts=max_attempts, delay=10, label=f"Upload {local}")
        logger.info(f"Uploaded {local} → {remote}")

    def download(self, remote: str, local: str, max_attempts: int = 3):
        """Download a file from the pod with retries."""
        def _do():
            with self._io_lock:
                self.lium.download(self.pod, remote=remote, local=local)
        _retry(_do, max_attempts=max_attempts, delay=5, label=f"Download {remote}")

    def _prep_command(self, command: str, env: dict | None = None) -> str:
        """Prefix ``command`` with env exports; fed via stdin so secrets
        don't appear in remote ``ps`` output."""
        if not env:
            return command
        exports = " && ".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in env.items()
        )
        return f"{exports} && {command}"

    def exec(self, command: str, env: dict = None, timeout: int = None):
        """Run a command on the pod; returns result dict, raises on timeout."""
        full_command = self._prep_command(command, env)
        ssh_command = "bash -s" if env else full_command
        started = time.time()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        # Drain stdout/stderr incrementally so long jobs don't stall the session.
        with self._io_lock:
            with self.lium.ssh_connection(self.pod, timeout=30) as client:
                transport = client.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)

                stdin, stdout, stderr = client.exec_command(ssh_command)
                if env:
                    stdin.write(full_command)
                    if not full_command.endswith("\n"):
                        stdin.write("\n")
                    stdin.flush()
                stdin.close()
                channel = stdout.channel
                channel.settimeout(0.1)

                while True:
                    while channel.recv_ready():
                        stdout_chunks.append(channel.recv(4096).decode("utf-8", errors="replace"))
                    while channel.recv_stderr_ready():
                        stderr_chunks.append(channel.recv_stderr(4096).decode("utf-8", errors="replace"))

                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        break

                    if timeout is not None and (time.time() - started) > timeout:
                        logger.error(f"Pod exec timed out after {timeout}s: {command[:80]}")
                        try:
                            channel.close()
                        except Exception:
                            pass
                        raise TimeoutError(f"Pod exec timed out after {timeout}s")

                    time.sleep(0.1)

                exit_code = channel.recv_exit_status()

        return {
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def is_alive(self, timeout: int = 15) -> bool:
        """Quick liveness check — returns True if the pod responds."""
        try:
            result = self.exec("echo alive", timeout=timeout)
            return "alive" in result.get("stdout", "")
        except Exception:
            return False

    def ensure_dependencies(self, teacher_model: str = "moonshotai/Kimi-K2.6"):
        """Install pod deps and apply B200 grouped_mm + Kimi K2.6 patches
        (idempotent, guarded by sentinels)."""
        try:
            logger.info("Ensuring pod dependencies...")
            # ``datasets`` is needed by verify_round / dataset.py / evalscope.
            dep_result = self.exec(
                "pip install --break-system-packages 'vllm>=0.19' accelerate -q 2>&1 | tail -1 && "
                "pip install --break-system-packages 'transformers>=5.0' -q 2>&1 | tail -1 && "
                "pip install --break-system-packages 'datasets>=2.20' -q 2>&1 | tail -1 && "
                "python3 -c 'import torch; import transformers; import vllm; import datasets; "
                "print(f\"torch={torch.__version__} transformers={transformers.__version__} "
                "vllm={vllm.__version__} datasets={datasets.__version__} cuda={torch.cuda.is_available()}\")'"
            )
            logger.info(f"Pod deps: {dep_result.get('stdout', '').strip()}")

            # B200 sm_100 grouped_mm fallback patch.
            self.exec(
                'python3 -c "import torch; cap=torch.cuda.get_device_capability(0); '
                'print(f\\"GPU compute capability: {cap}\\")" && '
                'grep -q PATCHED /usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py 2>/dev/null || '
                'sed -i \'s/return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/'
                '# PATCHED: force fallback on sm_100 (B200)\n'
                '    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)\n'
                '    if cap[0] >= 10:\n'
                '        return hasattr(torch.nn.functional, "grouped_mm")\n'
                '    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/\' '
                '/usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py'
            )
            logger.info("Applied grouped_mm B200 patch")

            # Kimi K2.6 transformers-5.x compat patches (no-op if absent).
            kimi_patch = r"""
set -e

# Patch 1: Kimi modeling_deepseek.py — stub is_torch_fx_available
for f in $(find /root/.cache/huggingface -path '*moonshotai*Kimi*K2*6*/modeling_deepseek.py' 2>/dev/null); do
    [ -f "$f" ] || continue
    if grep -q '_distil_is_torch_fx_stub' "$f"; then continue; fi
    cp -n "$f" "$f.orig"
    python3 -c "
import sys
path=sys.argv[1]
src=open(path).read()
old='from transformers.utils.import_utils import is_torch_fx_available'
new=('def is_torch_fx_available():  # _distil_is_torch_fx_stub\n'
     '    return False')
if old in src:
    open(path,'w').write(src.replace(old,new,1))
    print('Patched',path)
" "$f"
done

# Patch 2a: vLLM kimi_k25 model_executor — tolerate missing media_tokens_calculator
V=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py
if [ -f "$V" ] && ! grep -q '_distil_mm_tolerant' "$V"; then
    cp -n "$V" "$V.orig"
    python3 -c "
import sys
path=sys.argv[1]
src=open(path).read()
old='        self.media_tokens_calculator = image_processor.media_tokens_calculator'
new=('        # _distil_mm_tolerant\n'
     '        self.media_tokens_calculator = getattr(\n'
     '            image_processor, \\'media_tokens_calculator\\', lambda *a, **kw: 0,\n'
     '        )')
if old in src:
    open(path,'w').write(src.replace(old,new,1))
    print('Patched',path)
" "$V"
fi

# Patch 2b: vLLM kimi_k25 model_executor — short-circuit get_dummy_mm_items,
# get_dummy_text, and _call_hf_processor when the image processor is a
# transformers>=5.0 Qwen2VLImageProcessor fallback that lacks Kimi-specific
# attrs (num_frames_per_chunk etc.). Encoder-budget profiling triggers all
# three even with --skip-mm-profiling / --limit-mm-per-prompt; returning empty
# / bypassing the multimodal path is safe because pod_eval feeds text-only
# inputs at runtime.
if [ -f "$V" ] && ! grep -q '_distil_dummy_mm_tolerant' "$V"; then
    cp -n "$V" "$V.orig.v2"
fi
python3 - <<'PY'
import sys
path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py"
try:
    src = open(path).read()
except Exception:
    print("vLLM kimi_k25.py not found; skipping")
    raise SystemExit(0)
changed = False

# (a) get_dummy_mm_items returns [] when image_processor is wrong fallback
items_old = "    def get_dummy_mm_items(self):"
items_new = (
    "    def get_dummy_mm_items(self):  # _distil_dummy_mm_tolerant\n"
    "        ip = getattr(self.info, 'image_processor', None)\n"
    "        if ip is None or not hasattr(ip, 'num_frames_per_chunk'):\n"
    "            return []\n"
    "        return self._real_get_dummy_mm_items()\n"
    "\n"
    "    def _real_get_dummy_mm_items(self):"
)
if items_old in src and "_distil_dummy_mm_tolerant" not in src:
    src = src.replace(items_old, items_new, 1)
    changed = True

# (b) get_dummy_text returns " " (single space) when image_processor is wrong fallback.
# Empty string would tokenize to 0 tokens and break the upstream
# `(prompt_ids,) = input_ids` unpack in vLLM's processor.py:1165; a single
# space produces 1+ tokens which is enough for the encoder budget profiling.
text_old = "    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:"
text_new = (
    "    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:  # _distil_dummy_text_bypass\n"
    "        ip = getattr(self.info, 'image_processor', None)\n"
    "        if ip is None or not hasattr(ip, 'num_frames_per_chunk'):\n"
    "            return \" \"\n"
    "        return self._real_get_dummy_text(mm_counts)\n"
    "\n"
    "    def _real_get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:"
)
if text_old in src and "_distil_dummy_text_bypass" not in src:
    src = src.replace(text_old, text_new, 1)
    changed = True

if changed:
    open(path, "w").write(src)
    print("vLLM kimi_k25.py patched")
PY

echo KIMI_PATCHES_OK
"""
            patch_result = self.exec(kimi_patch)
            patch_out = patch_result.get('stdout', '') if isinstance(patch_result, dict) else patch_result
            logger.info(f"Kimi compat patches: {str(patch_out).strip()[-200:]}")
        except Exception as e:
            logger.warning(f"Pod dep check failed (non-fatal): {e}")

    def disk_cleanup(self, teacher_name: str, threshold: int = 85):
        """Drop student caches + stale /tmp; keep the teacher cached."""
        try:
            disk_check = self.exec("df --output=pcent / | tail -1 | tr -d ' %'")
            disk_pct_str = disk_check.get('stdout', disk_check) if isinstance(disk_check, dict) else disk_check
            disk_pct = int(str(disk_pct_str).strip())
            logger.info(f"Pod disk: {disk_pct}% used")

            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_teacher_only.json 2>/dev/null; "
                "df -h / | tail -1"
            )
            clean_result = self.exec(clean_cmd)
            clean_info = clean_result.get('stdout', clean_result) if isinstance(clean_result, dict) else clean_result
            logger.info(f"Cleanup done: {str(clean_info).strip()}")
            return disk_pct
        except Exception as e:
            logger.warning(f"Disk cleanup failed (non-fatal): {e}")
            return 0

    def clear_gpu(self):
        """Kill background GPU processes to free VRAM for eval."""
        try:
            self.exec("for s in distil train; do tmux kill-session -t $s 2>/dev/null; done; sleep 2; echo 'GPU cleared'")
            logger.info("Cleared GPU for eval")
        except Exception:
            pass

    def resume_background_tasks(self):
        """Restart any background tasks that were cleared for eval."""
        try:
            self.exec("test -f /home/autostart.sh && bash /home/autostart.sh; echo 'Background tasks resumed'")
            logger.info("Resumed background tasks on pod")
        except Exception:
            pass

    def post_eval_cleanup(self, teacher_name: str):
        """Clean up after eval: remove student caches, stale teacher cache, /tmp."""
        try:
            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "rm -f /home/teacher_cache.pt 2>/dev/null; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "df -h / | tail -1"
            )
            result = self.exec(clean_cmd)
            disk_info = result.get('stdout', result) if isinstance(result, dict) else result
            logger.info(f"Post-eval cleanup: {str(disk_info).strip()}")
        except Exception as e:
            logger.warning(f"Post-eval cleanup failed (non-fatal): {e}")
