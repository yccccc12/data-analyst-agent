"""Streamlit chat UI for the data-analyst agent.  Run:  streamlit run frontend/app.py
Needs the API running (cd backend; uvicorn app.main:app --reload)."""
import json
import os
import time
import uuid

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
STARTERS = ["Total sales by region", "Monthly profit in 2016 as a line chart", "Top 5 products by profit"]
THINKING_MAX_CHARS = 400   # the model's scratch work can run long; the UI shows only the start
TABLE_MAX_HEIGHT = 320     # px; taller results scroll inside the table instead of stretching the page

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="centered")
st.markdown(
    "<style>"
    ".block-container{max-width:760px;padding-top:2rem}"
    "[data-testid=stExpander] blockquote, [data-testid=stStatusWidget] blockquote{"
    "font-size:.9rem;opacity:.7;margin:.25rem 0}"
    "</style>", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
    # {"role", "content", "steps", "chart_urls", "label", "error"}; "label" and "error" are assistant-only
    st.session_state.messages = []


def ask(text: str) -> None:
    """Button callback: queue a question; it is picked up below like a typed one."""
    st.session_state.pending = text


def starter_buttons(where: str) -> None:
    for i, text in enumerate(STARTERS):
        st.button(text, key=f"{where}-{i}", on_click=ask, args=(text,), width="stretch")


with st.sidebar:
    st.title("📊 Data Analyst")
    st.caption("Ask questions about the Superstore sales data (2014–2017).")
    if st.button("New conversation", width="stretch", help=f"Session {st.session_state.session_id}"):
        try:
            requests.delete(f"{API}/chat/{st.session_state.session_id}", timeout=5)
        except requests.RequestException:
            pass
        st.session_state.clear()
        st.rerun()
    if st.session_state.messages:  # before the first question the examples are in the main area
        st.markdown("**Try:**")
        starter_buttons("side")


@st.cache_data(show_spinner=False, max_entries=64)
def fetch_chart(url: str) -> bytes:
    """Fetch the PNG server-side, so the browser never needs to reach the API address.
    Cached: Streamlit reruns the whole script on every message, and chart files never change."""
    r = requests.get(f"{API}{url}", timeout=10)
    r.raise_for_status()
    return r.content


def show_chart(url: str) -> None:
    try:
        data = fetch_chart(url)
    except requests.RequestException:
        st.caption(f"(could not load chart {url})")
        return
    st.image(data, width="stretch")
    st.download_button("Download chart", data, file_name=url.rsplit("/", 1)[-1], mime="image/png",
                       key=f"download-{url}", on_click="ignore", type="tertiary",
                       icon=":material/download:")


def show_table(df: pd.DataFrame) -> None:
    """A dataframe with a capped height and thousands separators on float columns."""
    floats = df.select_dtypes("float").columns
    st.dataframe(df, hide_index=True, height=min(35 * (len(df) + 1) + 3, TABLE_MAX_HEIGHT),
                 column_config={c: st.column_config.NumberColumn(format="localized") for c in floats})


def plural(n: int) -> str:
    return f"{n} tool call{'s' if n != 1 else ''}"


def step_label(steps: list[dict], seconds: float, failed: bool = False) -> str:
    return f"{'Failed after' if failed else 'Done in'} {seconds:.0f}s · {plural(sum(s['type'] == 'tool_call' for s in steps))}"


def render_thinking(text: str) -> None:
    """Muted blockquote. (Wrapping in _..._ breaks as soon as the text has a blank line.)"""
    text = text.strip()
    if len(text) > THINKING_MAX_CHARS:
        text = text[:THINKING_MAX_CHARS].rstrip() + "…"
    st.markdown("\n".join(f"> {line}" for line in text.splitlines()))


def render_tool_call(step: dict) -> None:
    args = step["arguments"]
    st.markdown(f"🔧 **`{step['name']}`**")
    if "query" in args:
        st.code(args["query"], language="sql", wrap_lines=True)
    elif args:
        st.caption("  ·  ".join(f"**{k}** = `{v}`" for k, v in args.items()))


def render_tool_result(step: dict) -> None:
    out = step["output"]
    if not step["ok"]:
        st.error(out, icon="❌")
        return
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        st.caption(f"✅ {out[:300]}{'…' if len(out) > 300 else ''}")
    elif "columns" in data and "rows" in data and "row_count" in data:  # run_sql
        shown = len(data["rows"])
        more = f", showing the first {shown}" if data["row_count"] > shown else ""
        cut = " (hit the row limit)" if data.get("truncated") else ""
        st.caption(f"✅ {data['row_count']} rows{more}{cut}  ·  `{data['result_id']}`")
        show_table(pd.DataFrame(data["rows"], columns=data["columns"]))
    elif "sample_rows" in data:  # describe_table: routine, so one line instead of a table
        names = ", ".join(f"`{c['name']}`" for c in data["columns"])
        st.caption(f"✅ Table `{data['table']}`: {len(data['columns'])} columns — {names}")
    elif "tables" in data:  # list_tables
        st.caption("✅ Tables: " + ", ".join(f"`{t}`" for t in data["tables"]))
    elif "chart_url" in data:  # make_chart
        st.caption("✅ Chart rendered")
    else:
        st.caption(f"✅ {out[:300]}{'…' if len(out) > 300 else ''}")


def render_step(step: dict) -> None:
    {"thinking": lambda s: render_thinking(s["text"]),
     "tool_call": render_tool_call,
     "tool_result": render_tool_result}[step["type"]](step)


def render_outcome(msg: dict) -> None:
    """What the user came for: charts first (the answer text can run long), then the text or the error."""
    for url in msg["chart_urls"]:
        show_chart(url)
    if msg.get("error"):
        st.error(msg["error"])
        st.caption("This turn was not completed. You can send the question again.")
    else:
        st.markdown(msg["content"])


def render_assistant(msg: dict) -> None:
    if msg["steps"]:
        with st.expander(msg["label"]):
            for s in msg["steps"]:
                render_step(s)
    render_outcome(msg)


def stream_events(message: str):
    """Yield event dicts from the SSE endpoint."""
    with requests.post(f"{API}/chat/stream",
                       json={"session_id": st.session_state.session_id, "message": message},
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


# st.chat_input is pinned to the bottom wherever it is called; calling it first lets the
# empty-state block below disappear as soon as a question (typed or from a button) arrives.
prompt = st.chat_input("Ask about the data…") or st.session_state.pop("pending", None)

if not st.session_state.messages and not prompt:
    st.markdown("### What would you like to know?")
    st.caption("Superstore sales, 2014–2017 · 9,994 order lines. Pick an example or type your own question.")
    starter_buttons("main")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        steps, answer, charts, failed = [], "", [], None
        started = time.monotonic()
        status = st.status("Working…", expanded=True)
        try:
            with status:
                for ev in stream_events(prompt):
                    if ev["type"] in ("thinking", "tool_call", "tool_result"):
                        steps.append(ev)
                        render_step(ev)
                    elif ev["type"] == "answer":
                        answer, charts = ev["text"], ev["chart_urls"]
                    elif ev["type"] == "error":
                        failed = ev["message"]
        except requests.RequestException as e:
            failed = f"Cannot reach the API at {API}: {e}"

        label = step_label(steps, time.monotonic() - started, failed=bool(failed))
        status.update(label=label, state="error" if failed else "complete", expanded=False)
        msg = {"role": "assistant", "content": answer, "steps": steps, "chart_urls": charts,
               "label": label, "error": failed}
        render_outcome(msg)
        # Kept even when it failed, so the error survives the next rerun instead of vanishing.
        st.session_state.messages.append(msg)
