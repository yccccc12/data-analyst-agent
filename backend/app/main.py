"""HTTP API.

EVENT CONTRACT for POST /chat/stream (Server-Sent Events, one JSON object per event):
    {"type": "thinking",    "text": str}
    {"type": "tool_call",   "id": str, "name": str, "arguments": dict}
    {"type": "tool_result", "id": str, "ok": bool, "output": str, "chart_url": str | None}
    {"type": "answer",      "text": str, "chart_urls": list[str]}   # last event on success
    {"type": "error",       "message": str}                          # last event on failure
The event source is agent.run_stream(history, message) -> Iterator[dict]; until it
exists, app.stub_stream.run_stream is used.
"""
import json
import logging
import time
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, charts, config, stub_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Data Analyst Agent")

# Charts rendered by make_chart are served at /charts/<file>.png
app.mount("/charts", StaticFiles(directory=config.CHARTS_DIR), name="charts")

# Conversation memory: session_id -> list of Responses-API input items.
# In-memory only, so it is lost on restart (fine for learning). Ordered least to most
# recently used; idle sessions expire after SESSION_TTL_S and at most MAX_SESSIONS are kept.
SESSIONS: OrderedDict[str, list] = OrderedDict()
_LAST_USED: dict[str, float] = {}
_last_chart_cleanup = 0.0


def _evict_sessions(now: float) -> None:
    for sid in [s for s in SESSIONS if now - _LAST_USED.get(s, 0) > config.SESSION_TTL_S]:
        SESSIONS.pop(sid)
    while len(SESSIONS) > config.MAX_SESSIONS:
        SESSIONS.popitem(last=False)  # least recently used
    for sid in set(_LAST_USED) - set(SESSIONS):
        del _LAST_USED[sid]


def _housekeeping(now: float) -> None:
    """Throttled: delete old chart PNGs."""
    global _last_chart_cleanup
    if now - _last_chart_cleanup >= config.CLEANUP_INTERVAL_S:
        _last_chart_cleanup = now
        removed = charts.cleanup_old_charts(config.CHART_TTL_S, now)
        if removed:
            log.info("removed %d old chart(s)", removed)


def get_history(session_id: str) -> list:
    """Return the session's history (creating it), mark it recently used, evict stale sessions."""
    now = time.time()
    history = SESSIONS.setdefault(session_id, [])
    SESSIONS.move_to_end(session_id)
    _LAST_USED[session_id] = now
    _evict_sessions(now)
    _housekeeping(now)
    return history


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    chart_urls: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = get_history(req.session_id)
    try:
        return agent.run(history, req.message)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except RuntimeError as e:  # e.g. missing LLM settings
        raise HTTPException(status_code=500, detail=str(e))


def _event_source():
    return getattr(agent, "run_stream", stub_stream.run_stream)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history = get_history(req.session_id)

    def generate():
        start, calls = time.time(), 0
        log.info("stream start session=%s", req.session_id)
        try:
            for event in _event_source()(history, req.message):
                calls += event["type"] == "tool_call"
                yield _sse(event)
        except Exception as e:  # the stream is already open, so report it as an event
            log.exception("stream failed session=%s", req.session_id)
            yield _sse({"type": "error", "message": str(e) or type(e).__name__})
        finally:
            log.info("stream end session=%s tool_calls=%d seconds=%.1f",
                     req.session_id, calls, time.time() - start)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.delete("/chat/{session_id}")
def reset(session_id: str):
    SESSIONS.pop(session_id, None)
    _LAST_USED.pop(session_id, None)
    return {"status": "reset"}
