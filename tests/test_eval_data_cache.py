import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "api"
for path in (str(API_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_data_cache import EvalDataCache  # noqa: E402


def test_list_payload_caches_sorted_json_files(monkeypatch, tmp_path):
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "skip.txt").write_text("x")
    cache = EvalDataCache(list_ttl_s=30)
    monkeypatch.setattr("eval_data_cache.time.time", lambda: 100.0)

    payload = cache.list_payload(str(tmp_path))

    assert payload == {"files": ["b.json", "a.json"], "count": 2}
    (tmp_path / "c.json").write_text("{}")
    assert cache.list_payload(str(tmp_path)) == payload


def test_response_for_file_reuses_cached_body(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"round": 1}))
    calls = {"n": 0}

    def read_json_file(file_path, default):
        calls["n"] += 1
        return json.loads(Path(file_path).read_text())

    cache = EvalDataCache()
    first = cache.response_for_file(str(path), read_json_file)
    second = cache.response_for_file(str(path), read_json_file)

    assert first.body == b'{"round": 1}'
    assert second.body == b'{"round": 1}'
    assert calls["n"] == 1


def test_response_for_file_missing_returns_404(tmp_path):
    cache = EvalDataCache()

    response = cache.response_for_file(str(tmp_path / "missing.json"), lambda _path, default: default)

    assert response.status_code == 404
