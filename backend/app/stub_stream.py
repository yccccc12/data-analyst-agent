"""Stand-in event generator, used by /chat/stream until agent.run_stream exists.

Yields the same event dicts the real loop will (see EVENT CONTRACT in main.py),
so the API and the Streamlit UI can be built and tested without an LLM.
"""
import time


def run_stream(history: list, user_message: str):
    history.append({"role": "user", "content": user_message})
    yield {"type": "thinking", "text": "I should look at the table first."}
    time.sleep(0.4)
    yield {"type": "tool_call", "id": "c1", "name": "describe_table", "arguments": {"table": "sales"}}
    time.sleep(0.4)
    yield {"type": "tool_result", "id": "c1", "ok": True, "output": "columns: order_date, sales, profit, ..."}
    yield {"type": "tool_call", "id": "c2", "name": "run_sql", "arguments": {"query": "SELECT 1"}}
    time.sleep(0.4)
    yield {"type": "tool_result", "id": "c2", "ok": False, "output": "Error: stub tool failure"}
    yield {"type": "answer", "text": f"(stub) You asked: {user_message}", "chart_urls": []}
