"""Run the eval set against the real agent (live LLM calls: slow, and it costs money).

    cd backend
    python -m evals.run                         # every case once
    python -m evals.run --repeat 3              # 3 runs per case: shows how unreliable each one is
    python -m evals.run --only profit-2016 unique-products
    python -m evals.run --tag trap              # only cases with this tag
    python -m evals.run --quiet                 # keep the agent's tool-call prints out of the console

Each run starts from an empty conversation. Results are also written to
evals/results/<timestamp>.json (answers, tool calls and the reason for every check).
Exit code: 0 if every run passed, 1 otherwise.
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app import agent, config

from . import checks, poison

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    problems = [p for c in cases for p in checks.validate_case(c)]
    ids = [c.get("id") for c in cases]
    problems += [f"duplicate id {i!r}" for i in {i for i in ids if ids.count(i) > 1}]
    if problems:
        sys.exit("cases.json has problems:\n  " + "\n  ".join(problems))
    return cases


def select(cases: list[dict], only: list[str] | None, tag: str | None) -> list[dict]:
    if only:
        unknown = set(only) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in only]
    if tag:
        cases = [c for c in cases if tag in c.get("tags", [])]
    if not cases:
        sys.exit("No cases selected.")
    return cases


_poisoned_path: Path | None = None


def poisoned_db() -> Path:
    """The poisoned copy of the database, built once per process (see poison.py)."""
    global _poisoned_path
    if _poisoned_path is None:
        _poisoned_path = poison.build(RESULTS_DIR / "poisoned.db")
    return _poisoned_path


@contextlib.contextmanager
def database_for(case: dict):
    """Point the agent (and the gold-SQL checks) at the database this case needs: the real one,
    or the poisoned copy when the case says "db": "poisoned"."""
    original = config.DB_PATH
    if case.get("db") == "poisoned":
        config.DB_PATH = poisoned_db()
    checks.gold_rows.cache_clear()  # cached gold results belong to one database
    try:
        yield config.DB_PATH
    finally:
        config.DB_PATH = original
        checks.gold_rows.cache_clear()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_turns(turns: list[str]) -> checks.Outcome:
    """Send the turns in order in ONE conversation; return what happened in the last turn."""
    history: list = []
    outcome = checks.Outcome(answer="")
    for message in turns:
        outcome = checks.Outcome(answer="")
        for event in agent.run_stream(history, message):
            if event["type"] == "tool_call":
                outcome.tool_calls.append(event["name"])
                outcome.tool_args.append(json.dumps(event["arguments"], ensure_ascii=False))
            elif event["type"] == "answer":
                outcome.answer, outcome.chart_urls = event["text"], event["chart_urls"]
    return outcome


def run_once(case: dict, quiet: bool) -> dict:
    """One run of one case -> a record with the outcome and every check's verdict."""
    log = io.StringIO()
    started = time.monotonic()
    error = None
    outcome = checks.Outcome(answer="")
    with database_for(case) as db_path:
        before = file_hash(db_path)
        try:
            with contextlib.redirect_stdout(log) if quiet else contextlib.nullcontext():
                outcome = run_turns(case["turns"])
        except Exception as e:  # noqa: BLE001 - an agent crash is a failed run, not a crashed eval
            error = f"{type(e).__name__}: {e}"
        outcome.db_changed = file_hash(db_path) != before
        seconds = time.monotonic() - started

        results = []
        for spec in case["checks"]:
            if error:
                results.append({"type": spec["type"], "passed": False, "reason": f"agent raised {error}"})
            else:
                ok, reason = checks.run_check(spec, outcome)
                results.append({"type": spec["type"], "passed": ok, "reason": reason})
    return {
        "case": case["id"], "tags": case.get("tags", []), "db": case.get("db", "main"),
        "passed": all(r["passed"] for r in results), "checks": results,
        "answer": outcome.answer, "tool_calls": outcome.tool_calls, "tool_args": outcome.tool_args,
        "chart_urls": outcome.chart_urls, "db_changed": outcome.db_changed,
        "seconds": round(seconds, 1), "error": error,
        "tool_log": log.getvalue() if quiet else None,
    }


def passes(records: list[dict]) -> str:
    return f"{sum(r['passed'] for r in records)}/{len(records)}"


def print_report(records: list[dict], cases: list[dict]) -> None:
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case"]].append(r)

    width = max(len(c["id"]) for c in cases)
    print(f"\n{'case'.ljust(width)}  passed  avg s  avg tools")
    print("-" * (width + 28))
    for c in cases:
        runs = by_case.get(c["id"], [])
        if not runs:
            continue
        avg_s = sum(r["seconds"] for r in runs) / len(runs)
        avg_t = sum(len(r["tool_calls"]) for r in runs) / len(runs)
        flag = "" if all(r["passed"] for r in runs) else "   <-- " + ("FAIL" if not any(r["passed"] for r in runs) else "FLAKY")
        print(f"{c['id'].ljust(width)}  {passes(runs):>6}  {avg_s:5.1f}  {avg_t:9.1f}{flag}")

    by_tag = defaultdict(list)
    for r in records:
        for t in r["tags"] or ["(untagged)"]:
            by_tag[t].append(r)
    print("\nby tag:  " + "   ".join(f"{t} {passes(rs)}" for t, rs in sorted(by_tag.items())))
    print(f"overall: {passes(records)} runs passed")

    failures = [r for r in records if not r["passed"]]
    if failures:
        print("\n--- failures ---")
    for r in failures:
        print(f"\n[{r['case']}]  {r['seconds']}s, tools: {r['tool_calls']}")
        for chk in r["checks"]:
            if not chk["passed"]:
                print(f"  x {chk['type']}: {chk['reason']}")
        print(f"  answer: {' '.join(r['answer'].split())[:400]!r}")


def save(records: list[dict], args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    payload = {"model": config.MODEL, "repeat": args.repeat, "passed": sum(r["passed"] for r in records),
               "runs": len(records), "records": records}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the data-analyst agent eval set.")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (default 1)")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these case ids")
    parser.add_argument("--tag", help="run just the cases with this tag")
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json", help="path to the cases file")
    parser.add_argument("--quiet", action="store_true", help="capture the agent's tool-call prints into the results file")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers contain characters the Windows console can't print
    config.require_llm_config()
    cases = select(load_cases(args.cases), args.only, args.tag)
    total = len(cases) * args.repeat
    print(f"Model: {config.MODEL}   cases: {len(cases)}   repeat: {args.repeat}   runs: {total}")

    records, n = [], 0
    try:
        for case in cases:
            for i in range(1, args.repeat + 1):
                n += 1
                record = run_once(case, args.quiet)
                records.append(record)
                print(f"[{n}/{total}] {case['id']} (run {i}): {'PASS' if record['passed'] else 'FAIL'}  {record['seconds']}s", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; reporting the runs that finished.")

    if not records:
        return 1
    print_report(records, cases)
    print(f"\nSaved: {save(records, args)}")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
