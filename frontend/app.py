"""Streamlit chat UI for the data-analyst agent.  Run:  streamlit run frontend/app.py
Needs the API running (cd backend; uvicorn app.main:app --reload)."""
import json
import os
import uuid

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="centered")
st.markdown(
    "<style>"
    ".block-container{max-width:760px;padding-top:2rem}"
    "[data-testid=stExpander] blockquote, [data-testid=stStatusWidget] blockquote{"
    "font-size:.9rem;opacity:.7;margin:.25rem 0}"
    "</style>", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []  # {"role", "content", "steps", "chart_urls"}

with st.sidebar:
    st.title("📊 Data Analyst")
    st.caption("Ask questions about the Superstore sales data (2014–2017).")
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("New conversation", use_container_width=True):
        try:
            requests.delete(f"{API}/chat/{st.session_state.session_id}", timeout=5)
        except requests.RequestException:
            pass
        st.session_state.clear()
        st.rerun()
    st.markdown("**Try:**\n- Total sales by region\n- Monthly profit in 2016 as a line chart\n- Top 5 products by profit")


def show_chart(url: str) -> None:
    """Fetch the PNG server-side, so the browser never needs to reach the API address."""
    try:
        st.image(requests.get(f"{API}{url}", timeout=10).content)
    except requests.RequestException:
        st.caption(f"(could not load chart {url})")


def plural(n: int) -> str:
    return f"{n} tool call{'s' if n != 1 else ''}"


def render_thinking(text: str) -> None:
    """Muted blockquote. (Wrapping in _..._ breaks as soon as the text has a blank line.)"""
    st.markdown("\n".join(f"> {line}" for line in text.strip().splitlines()))


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
        st.dataframe(pd.DataFrame(data["rows"], columns=data["columns"]), hide_index=True)
    elif "sample_rows" in data:  # describe_table
        st.caption(f"✅ Table `{data['table']}`: {len(data['columns'])} columns")
        st.dataframe(pd.DataFrame(data["columns"]), hide_index=True)
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


def render_assistant(msg: dict) -> None:
    if msg["steps"]:
        calls = sum(s["type"] == "tool_call" for s in msg["steps"])
        with st.expander(f"Thinking and tools ({plural(calls)})"):
            for s in msg["steps"]:
                render_step(s)
    st.markdown(msg["content"])
    for url in msg["chart_urls"]:
        show_chart(url)


def stream_events(message: str):
    """Yield event dicts from the SSE endpoint."""
    with requests.post(f"{API}/chat/stream",
                       json={"session_id": st.session_state.session_id, "message": message},
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

if prompt := st.chat_input("Ask about the data…"):
    st.session_state.messages.append({"role": "user", "content": prompt, "steps": [], "chart_urls": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        steps, answer, charts, failed = [], "", [], None
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

        calls = sum(s["type"] == "tool_call" for s in steps)
        if failed:
            status.update(label="Failed", state="error", expanded=False)
            st.error(failed)
        else:
            status.update(label=f"Done ({plural(calls)})", state="complete", expanded=False)
            st.markdown(answer)
            for url in charts:
                show_chart(url)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "steps": steps, "chart_urls": charts})
