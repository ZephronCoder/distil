"""Cached responses for eval-data JSON files."""

import json
import os
import threading
import time

from fastapi.responses import JSONResponse, Response


class EvalDataCache:
    def __init__(self, *, list_ttl_s: float = 30.0, body_max_entries: int = 8):
        self.list_ttl_s = list_ttl_s
        self.body_max_entries = body_max_entries
        self.body_cache = {}
        self.body_lock = threading.Lock()
        self.list_cache = {"data": None, "ts": 0.0}

    def list_payload(self, data_dir: str):
        now = time.time()
        cached = self.list_cache
        if cached["data"] is not None and now - cached["ts"] < self.list_ttl_s:
            return cached["data"]
        if not os.path.exists(data_dir):
            payload = {"files": []}
        else:
            files = sorted(
                [name for name in os.listdir(data_dir) if name.endswith(".json")],
                reverse=True,
            )
            payload = {"files": files, "count": len(files)}
        cached["data"] = payload
        cached["ts"] = now
        return payload

    def response_for_file(self, path: str, read_json_file) -> Response:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return JSONResponse(content={"error": "No eval data available"}, status_code=404)
        key = (path, mtime)
        cached = self.body_cache.get(key)
        if cached is None:
            with self.body_lock:
                cached = self.body_cache.get(key)
                if cached is None:
                    try:
                        payload = read_json_file(path, {})
                    except Exception as exc:
                        return JSONResponse(content={"error": str(exc)}, status_code=500)
                    cached = {"body": json.dumps(payload).encode()}
                    if len(self.body_cache) >= self.body_max_entries:
                        keep = max(0, self.body_max_entries - 1)
                        for stale_key in list(self.body_cache)[: -keep or None]:
                            self.body_cache.pop(stale_key, None)
                    self.body_cache[key] = cached
        return Response(
            content=cached["body"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
        )
