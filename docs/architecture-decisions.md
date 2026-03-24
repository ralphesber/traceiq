# Architecture Decisions & Invariants

These are decisions that were made deliberately. Do not change them without understanding why.

---

## 1. Score computation must use all 100 rows, not the sample

**Rule:** `fetch_experiment_rows` always fetches all 100 runs to compute `aggregated_scores`. The sample (capped at 50) is only for row-level pattern analysis.

**Why:** LangSmith computes averages across all runs. If we average only a sample, scores diverge from what LangSmith shows — e.g. `marks_exact_match` shows 49% on TraceIQ vs 39% on LangSmith. This breaks user trust immediately.

**What this means in code:** In `agent/experiment_tools.py`, `fetch_experiment_rows` always queries `limit=100`, computes `aggregated_scores` from all runs, then returns `sample_rows[:50]` for the agent to read. Never change the aggregation to use the sample.

---

## 2. Worker must never import server.py

**Rule:** `worker.py` imports DB helpers from `db.py` only. It must never `import server` or `from server import ...`.

**Why:** `server.py` has `gevent.monkey.patch_all()` at the top (required for gunicorn gevent workers). If the worker imports server.py, the monkey patch fires and completely breaks Python's `asyncio` — causing `ainvoke()` and `asyncio.wait_for()` to hang silently forever. This works fine locally (no gevent installed) but breaks on Railway.

**What this means in code:** All DB helpers live in `db.py`. Both `server.py` and `worker.py` import from `db.py`. Never move DB helpers back into `server.py` only.

---

## 3. get_failing_rows is capped at 10 rows with 1000-char truncation

**Rule:** `get_failing_rows` returns at most 10 failing rows, with `inputs` and `outputs` truncated to 1000 chars each.

**Why:** Without this cap, a single tool call returns 932KB of data (61 full rows with long educational content). Passing 932KB into Claude's context causes multi-minute hangs or API timeouts. 10 rows with truncation is enough to identify failure patterns.

---

## 4. LangGraph agent uses asyncio.timeout, not asyncio.wait_for

**Rule:** The agent timeout uses `async with asyncio.timeout(AGENT_TIMEOUT)` wrapping `astream()`.

**Why:** `asyncio.wait_for()` does not reliably cancel LangGraph's `ainvoke()`. `asyncio.timeout()` (Python 3.11+) is the correct primitive for cancelling async generators.

---

## 5. Streaming via astream(stream_mode="updates"), not ainvoke

**Rule:** The agent streams via `astream(stream_mode="updates")` so tool calls and agent thinking are logged to the DB as steps in real time.

**Why:** `ainvoke()` blocks until complete with no intermediate visibility. With `astream`, the UI shows the agent working step by step, which is the core UX of the feature.
