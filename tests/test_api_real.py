import pytest
from fastapi.testclient import TestClient

from main import app

pytest.importorskip("torch")
client = TestClient(app)


def test_tasks_lists_real_tasks_with_neutral_ids():
    ids = [t["id"] for t in client.get("/tasks").json()["real_tasks"]]
    assert ids == [f"task_{i}" for i in range(1, 6)]


def test_real_mode_roundtrip_and_legacy_switch_back():
    h = {"X-Session-Id": "real-test"}
    obs = client.post("/reset", json={"mode": "real", "task_id": "task_1"}, headers=h).json()
    assert "train.py" in obs["files"] and obs["metric_summary"]
    r = client.post("/step", json={"action_type": "read_file", "path": "train.py"}, headers=h).json()
    assert "common" in r["observation"]["content"]
    assert client.get("/state", headers=h).json()["steps_used"] == 1
    bad = client.post("/step", json={"action_type": "read_log", "target": "x"}, headers=h)
    assert bad.status_code == 422  # legacy action rejected in real mode
    legacy = client.post("/reset", json={"task_id": "task_1"}, headers=h).json()
    assert "visible_logs" in legacy
