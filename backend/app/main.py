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

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, config, stub_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Data Analyst Agent")

# Charts rendered by make_chart are served at /charts/<file>.png
app.mount("/charts", StaticFiles(directory=config.CHARTS_DIR), name="charts")

# Conversation memory: session_id -> list of Responses-API input items.
# In-memory only, so it is lost on restart (fine for learning).
SESSIONS: dict[str, list] = {}


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
    history = SESSIONS.setdefault(req.session_id, [])
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
    history = SESSIONS.setdefault(req.session_id, [])

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
    return {"status": "reset"}
