"""Lium pod connection, initialization, and lifecycle management."""
import logging
import time

from eval.pod import PodManager

logger = logging.getLogger("distillation.remote_validator")

# How long to wait between retries when the pod is missing or unreachable
# at startup. Validator stays alive across pod outages instead of crash-
# looping under systemd, which thrashes the journal and burns CPU on retries.
_POD_WAIT_INTERVAL_S = 60
# Cap the total wait so a permanently-missing pod still surfaces an alert.
_POD_WAIT_TIMEOUT_S = 6 * 3600


def init_pod(lium, pod_name: str, teacher_model: str) -> PodManager:
    """Connect to the Lium GPU pod, clear stale artifacts, ensure deps.

    If the named pod is missing at startup (Lium tear-down, hardware swap,
    operator pause) we poll every minute until it appears, up to
    ``_POD_WAIT_TIMEOUT_S``. This keeps the validator warm across pod
    outages instead of forcing systemd into a tight crash loop.
    """
    print("[validator] Initializing Lium client...", flush=True)
    pod = PodManager(lium, pod_name=pod_name)

    deadline = time.time() + _POD_WAIT_TIMEOUT_S
    attempt = 0
    while True:
        attempt += 1
        print(f"[validator] Connecting to pod '{pod_name}' (attempt {attempt})...", flush=True)
        try:
            pod.connect()
            break
        except RuntimeError as exc:
            if time.time() >= deadline:
                raise
            wait_min = round((deadline - time.time()) / 60)
            logger.warning(
                f"Pod '{pod_name}' not ready yet ({exc}); retrying in {_POD_WAIT_INTERVAL_S}s "
                f"(remaining wait budget: {wait_min} min)"
            )
            print(
                f"[validator] Pod '{pod_name}' not ready; retrying in {_POD_WAIT_INTERVAL_S}s "
                f"(remaining wait budget: {wait_min} min)",
                flush=True,
            )
            time.sleep(_POD_WAIT_INTERVAL_S)
    print(f"[validator] Connected to pod: {pod.pod.name if pod.pod else '?'}", flush=True)

    print("[validator] Cleaning stale /home/pod_eval.py...", flush=True)
    try:
        pod.exec(
            "rm -f /home/pod_eval.py /home/pod_eval_vllm.py /home/eval_output.log "
            "/home/eval_progress.json /home/eval_results.json 2>/dev/null"
        )
    except Exception:
        pass
    print("[validator] Pod init complete (eval script uploaded per-round)", flush=True)

    print("[validator] Ensuring pod dependencies...", flush=True)
    pod.ensure_dependencies(teacher_model)
    print("[validator] Pod dependencies ready", flush=True)

    return pod
