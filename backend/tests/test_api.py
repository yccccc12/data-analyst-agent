"""Tests for the FastAPI app in main.py. The agent itself is faked, so there are no LLM calls."""
import json
import pytest
from fastapi.testclient import TestClient

from app import agent, config, main


@pytest.fixture
def client():
    main.SESSIONS.clear()
    return TestClient(main.app)


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
