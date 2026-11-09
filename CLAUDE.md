# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

A data-analyst agent that answers questions about a SQLite database and can render charts. The backend is FastAPI; the frontend is a Streamlit chat UI (`frontend/app.py`) that shows the agent's thinking steps and tool calls live; both run in Docker. The agent core is implemented and works end to end (`agent.py`, `tools.py`, and the SQL guard in `db.py`; the user wrote the first regex-based `validate_select`, which was later replaced by a SQLite authorizer at their request). Tests are in `backend/tests/` (SQL guard, tools, agent loop and its event stream with a scripted fake model from `conftest.py`, charts/results cache, API incl. SSE) and make no network calls; there is no linter. The branch is `main`, pushed to GitHub (`origin`). `README.md` is the user-facing overview; keep it in step with this file.

## Commands

Windows / PowerShell. The virtualenv is `backend/.venv` (gitignored); dependencies are in the root `requirements.txt`.

```powershell
backend\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend/app/scripts/load_csv.py          # (re)build backend/sample.db from data/*.csv
cd backend; uvicorn app.main:app --reload       # run the API; run from backend/ so `app` imports
cd backend; python -m app.agent ["question"]    # chat with the agent in the terminal (live LLM calls)
cd backend; python -m pytest                    # tests (pip install -r requirements-dev.txt); a single test: pytest -k name
cd backend; python -m evals.run [--repeat N] [--only ID...] [--tag T] [--quiet]   # eval set against the LIVE agent (LLM calls, slow); results in backend/evals/results/
streamlit run frontend/app.py                   # the UI (from the repo root); needs the API running; API_URL env var, default http://127.0.0.1:8000
docker compose up --build                       # API on :8000 + UI on :8501; reads .env, builds sample.db into the image
```

Endpoints: `GET /health`, `POST /chat {session_id, message}` → `{answer, chart_urls}`, `POST /chat/stream` (same body; Server-Sent Events, see Architecture), `DELETE /chat/{session_id}`, static charts at `/charts/<file>.png`.

Settings live in a root-level `.env` (gitignored): `DASHSCOPE_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` are required (checked by `config.require_llm_config()` on first LLM call, not at import). Optional: `DB_PATH` (relative paths resolve against `backend/`), `LLM_ENABLE_THINKING`, `LLM_TIMEOUT_S`, `SESSION_TTL_S` (default 3600), `MAX_SESSIONS` (100), `CHART_TTL_S` (86400).

## Evals

`backend/evals/` is separate from `tests/` because it makes live LLM calls. `cases.json` is the eval set (format documented in its `_about` / `_check_types` keys): each case has `turns` and `checks`, and checks carry their own gold SQL, run against `sample.db`, so no expected values are hard-coded. `checks.py` has the grading functions (`number` parsing handles `$`, commas, `81.8K`, and ignores sign), `run.py` runs each case from an empty history via `agent.run_stream`, records tool calls, and reports pass counts per case/tag. Cases with `"db": "poisoned"` run against a copy of the database whose text columns carry injected instructions (`poison.py`, numbers unchanged, `sample.db` never written); every run also hashes the DB file before and after (`db_unchanged` check). Tags: `injection` (direct + indirect), `scope` (off-topic handling). `tests/test_evals.py` covers the harness itself with the fake model (no network).

## Rules

- **No agent frameworks.** Do not add LangChain, LlamaIndex, CrewAI, AutoGen, OpenAI Agents SDK, or similar. Build the agent loop by hand on top of the plain `openai` SDK.
- **Database access is read-only.** The agent must never write to, alter, or delete from any database. Use read-only credentials/connections and reject anything other than `SELECT`-style queries before execution.
- **Print every tool call.** Each tool invocation the agent makes must be printed to the console (tool name and arguments, plus the result or error) so runs can be followed and audited.

## Data

The agent queries a SQLite DB (`backend/sample.db`, gitignored) built from `data/e commerce dataset (2).csv` (Superstore sales, 2014–2017, 9,994 rows).

- Single flat table `sales`, one row per order line (an `order_id` spans several rows). There are no other tables, so no JOINs are needed.
- The loader converts dates to ISO `YYYY-MM-DD`, renames columns to snake_case, and stores `postal_code` as 5-char text. The CSV is Windows-1252, not UTF-8. It builds into a temp file and swaps it in with `os.replace`.
- Known quirks, left in on purpose: `profit` can be negative, and 32 `product_id`s map to more than one `product_name`.

## Architecture

Request path: `main.py` (`POST /chat` and `POST /chat/stream`, per-session history in an in-memory `SESSIONS` dict) → `agent.run(history, message)` / `agent.run_stream(history, message)` → loop of `llm.call_model` ↔ `tools.execute_tool` → `tool_impl.*` → `db` / `charts`.

- **LLM:** the OpenAI Python SDK against Alibaba Cloud DashScope (Qwen), not OpenAI. `llm.py` builds the client lazily and uses the **Responses API** (`client.responses.create`), not Chat Completions. Qwen options go through `extra_body` (`enable_thinking`). Verified live: `response.output` items are `reasoning`, `function_call` (`call_id`, `name`, JSON-string `arguments`) and `message`; tool results go back as `{"type": "function_call_output", "call_id", "output"}`. The full shapes are in the `agent.py` docstring.
- **Two-layer read-only DB access (`db.py`):** layer 1 is a SQLite **authorizer** (`con.set_authorizer`) installed on the very connection that runs the query: SQLite asks before every operation, and only `SELECT`/`READ`/`FUNCTION`/`RECURSIVE` are allowed, so INSERT/UPDATE/DELETE/DDL/PRAGMA/ATTACH/BEGIN are refused however the SQL is written (comments, quoting, `WITH ... DELETE`). Denials become a `ValueError` with a message for the model (schema changes are reported as "Changing the schema", since SQLite does them as writes to `sqlite_master`); multiple statements and empty/comment-only input are also `ValueError`; ordinary SQL errors (`no such column`, syntax) stay `sqlite3.OperationalError`. Layer 2 is `get_connection()`: `mode=ro` + `PRAGMA query_only=ON`. `run_query()` is for model-written SQL (`_execute(sql, untrusted=True)`); `_execute(sql)` without the flag skips the authorizer and is only for our own queries in `tool_impl` (it needs `PRAGMA table_info`). Both enforce `MAX_ROWS` and a `QUERY_TIMEOUT_S` via a progress handler.
- **Tool bodies vs. wiring:** `tool_impl.py` holds the plain functions (`list_tables`, `describe_table`, `run_sql`, `make_chart`). `tools.py` holds the JSON schemas, the dispatch dict and `execute_tool`, which turns every failure into an `Error: ...` string so the loop never crashes and the model can self-correct.
- **Agent loop (`agent.py`):** the system prompt goes in as the first `{"role": "system"}` history item (verified to work on DashScope). `run_stream()` is the loop itself: a generator that yields events as it goes, and `run()` is a thin wrapper that consumes it and returns `{answer, chart_urls}`. It replays every `response.output` item into the history, answers each `function_call` (parallel calls supported) via `print_tool_call` + `execute_tool`, stops at `MAX_ITERATIONS`, and rolls the history back if a turn raises (or the stream is closed early, via `GeneratorExit`), since a dangling `function_call` would break later calls. The final `answer` event is yielded outside the `try`, and `run()` must read the stream to the end, so a finished turn is never rolled back. At the start of each turn `compact_history()` drops old `reasoning` items and truncates old tool outputs to `KEEP_OUTPUT_CHARS` to keep long sessions inside the context window. A non-`completed` status or empty text raises instead of returning an empty answer.
- **Result cache:** `run_sql` stores the full result in `results.py` and returns only a `PREVIEW_ROWS` preview plus a `result_id`; `make_chart` takes that `result_id`, so large data never enters the model's context. The cache is process-wide and in memory.
- **Charts:** `charts.py` renders with matplotlib's `Agg` backend into `backend/charts/` (gitignored, created by `config.py`) and `main.py` serves it at `/charts`.
- **Event stream (`POST /chat/stream`):** SSE, one JSON object per `data:` line. Events: `thinking {text}` (non-empty `reasoning` summaries; Qwen returns them), `tool_call {id, name, arguments}`, `tool_result {id, ok, output, chart_url}`, then `answer {text, chart_urls}` last, or `error {message}` if the generator raised (`main.py` converts exceptions to this event and logs the request with the `logging` module). The contract is in the `main.py` docstring. `main.py` uses `agent.run_stream` and falls back to `stub_stream.run_stream` (fake events, no LLM) only if it is missing.
- **Frontend (`frontend/app.py`):** Streamlit; reads the SSE stream with `requests`, renders thinking/tool calls inside a live `st.status` that collapses to an expander per message, and fetches chart PNGs server-side (`fetch_chart`, cached with `st.cache_data`) so the browser never needs to resolve the API's address (in Docker it is `http://backend:8000`). Charts render above the answer text; failed turns are kept in the history as assistant messages with an `error`, so they survive reruns; the empty state has clickable example questions (queued via `session_state.pending`); `describe_table` results are one line and long thinking text is truncated. UI config (minimal toolbar, accent colour) is in `.streamlit/config.toml`, read from the directory `streamlit run` starts in (the repo root; the Dockerfile copies it).
- **Docker:** one `Dockerfile` (python:3.12-slim) used by both services in `docker-compose.yml`; `sample.db` is built at image build time from `data/` (the `.db` is gitignored); the API key comes from `.env` via `env_file` and is never baked in; charts live in a named volume.
- **Scope assumption:** single user. Sessions, the results cache and chart rendering are not made thread-safe or session-scoped, the results cache is never trimmed. `SESSIONS` is an `OrderedDict` with idle-TTL and LRU eviction (`main.get_history`), and `charts.cleanup_old_charts` deletes PNGs older than `CHART_TTL_S` (run at most every `CLEANUP_INTERVAL_S` from chat requests). Revisit before serving multiple users.
