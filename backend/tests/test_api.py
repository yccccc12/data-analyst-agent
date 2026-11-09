"""Tests for the FastAPI app in main.py. The agent itself is faked, so there are no LLM calls."""
import json
import pytest
from fastapi.testclient import TestClient

from app import agent, config, main


@pytest.fixture
def client():
    main.SESSIONS.clear()
    main._LAST_USED.clear()
    return TestClient(main.app)


def test_idle_sessions_expire(monkeypatch):
    main.SESSIONS.clear(); main._LAST_USED.clear()
    monkeypatch.setattr(config, "SESSION_TTL_S", 60)
    now = 1_000_000.0
    monkeypatch.setattr(main.time, "time", lambda: now)
    main.get_history("old").append("x")
    monkeypatch.setattr(main.time, "time", lambda: now + 120)
    main.get_history("new")
    assert list(main.SESSIONS) == ["new"] and list(main._LAST_USED) == ["new"]


def test_least_recently_used_session_is_evicted_over_the_cap(monkeypatch):
    main.SESSIONS.clear(); main._LAST_USED.clear()
    monkeypatch.setattr(config, "MAX_SESSIONS", 2)
    main.get_history("a"); main.get_history("b")
    main.get_history("a")          # a is now the most recently used
    main.get_history("c")          # over the cap: b goes
    assert list(main.SESSIONS) == ["a", "c"]


def test_old_charts_are_deleted_and_new_ones_kept():
    import os, time
    from app import charts
    old, new = config.CHARTS_DIR / "_old_test.png", config.CHARTS_DIR / "_new_test.png"
    old.write_bytes(b"x"); new.write_bytes(b"x")
    try:
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        assert charts.cleanup_old_charts(3600) >= 1
        assert not old.exists() and new.exists()
    finally:
        new.unlink(missing_ok=True); old.unlink(missing_ok=True)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_answer_and_chart_urls(client, monkeypatch):
    monkeypatch.setattr(agent, "run", lambda history, msg: {"answer": f"echo: {msg}", "chart_urls": ["/charts/a.png"]})
    r = client.post("/chat", json={"session_id": "s1", "message": "hello"})
    assert r.status_code == 200
    assert r.json() == {"answer": "echo: hello", "chart_urls": ["/charts/a.png"]}


def test_chat_validates_the_request_body(client):
    assert client.post("/chat", json={"message": "no session"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s1"}).status_code == 422


def test_each_session_has_its_own_history(client, monkeypatch):
    def fake_run(history, msg):
        history.append(msg)
        return {"answer": str(len(history)), "chart_urls": []}

    monkeypatch.setattr(agent, "run", fake_run)
    ask = lambda sid: client.post("/chat", json={"session_id": sid, "message": "m"}).json()["answer"]
    assert [ask("a"), ask("a"), ask("b")] == ["1", "2", "1"]


def test_delete_resets_a_session(client, monkeypatch):
    def fake_run(history, msg):
        history.append(msg)
        return {"answer": str(len(history)), "chart_urls": []}

    monkeypatch.setattr(agent, "run", fake_run)
    ask = lambda: client.post("/chat", json={"session_id": "a", "message": "m"}).json()["answer"]
    ask(); ask()
    assert client.delete("/chat/a").json() == {"status": "reset"}
    assert ask() == "1"
    assert client.delete("/chat/never-existed").status_code == 200   # idempotent


def test_missing_llm_settings_become_a_500_with_a_message(client, monkeypatch):
    def fake_run(history, msg):
        raise RuntimeError("Missing required settings in .env: LLM_MODEL")

    monkeypatch.setattr(agent, "run", fake_run)
    r = client.post("/chat", json={"session_id": "s", "message": "hi"})
    assert r.status_code == 500 and "LLM_MODEL" in r.json()["detail"]


def test_unimplemented_agent_is_a_501(client, monkeypatch):
    def fake_run(history, msg):
        raise NotImplementedError("not yet")

    monkeypatch.setattr(agent, "run", fake_run)
    assert client.post("/chat", json={"session_id": "s", "message": "hi"}).status_code == 501


def test_charts_are_served_as_png(client):
    (config.CHARTS_DIR / "_test_chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        r = client.get("/charts/_test_chart.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    finally:
        (config.CHARTS_DIR / "_test_chart.png").unlink()
    assert client.get("/charts/does-not-exist.png").status_code == 404


def _events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_stream_sends_events_as_sse(client, monkeypatch):
    def fake_stream(history, msg):
        yield {"type": "tool_call", "id": "1", "name": "list_tables", "arguments": {}}
        yield {"type": "answer", "text": f"echo: {msg}", "chart_urls": []}

    monkeypatch.setattr(agent, "run_stream", fake_stream, raising=False)
    r = client.post("/chat/stream", json={"session_id": "s1", "message": "hi"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in _events(r)] == ["tool_call", "answer"]


def test_stream_reports_a_failure_as_an_error_event(client, monkeypatch):
    def broken(history, msg):
        yield {"type": "thinking", "text": "x"}
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "run_stream", broken, raising=False)
    events = _events(client.post("/chat/stream", json={"session_id": "s1", "message": "hi"}))
    assert events[-1] == {"type": "error", "message": "boom"}


def test_stream_falls_back_to_the_stub_until_run_stream_exists(client, monkeypatch):
    monkeypatch.delattr(agent, "run_stream", raising=False)
    events = _events(client.post("/chat/stream", json={"session_id": "s1", "message": "hi"}))
    assert events[-1]["type"] == "answer"
