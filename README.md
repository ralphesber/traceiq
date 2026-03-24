# TraceIQ

**LangSmith tells you what happened. TraceIQ tells you why — and what to change.**

TraceIQ is a hosted web tool for AI product builders who use LangSmith. Connect to a LangSmith experiment, ask a question, and an AI agent investigates your eval results — identifying failure patterns and recommending specific prompt changes.

---

## What it does

### Experiments — Prompt Improvement

Connect to a LangSmith dataset + experiment. Ask a question. An AI agent reads your eval results, identifies failure patterns, and recommends specific prompt changes.

> *"Why is marks_exact_match low?"*
> → Agent fetches failing rows, reads actual inputs and outputs, identifies root causes, proposes concrete prompt edits.

The agent plans its own investigation based on your question — it doesn't follow a fixed script:
- *"Why is X low?"* → goes straight to `get_failing_rows`
- *"What should I improve?"* → starts with `fetch_experiment_rows` to find the weakest metrics, then drills in
- *"Compare the last two experiments"* → uses `compare_experiments`

---

## Architecture

```
Browser (index.html)
    ↕ REST (polling /jobs/<id> every 2s)
Flask server (server.py) — gunicorn + gevent
    ↓ Postgres job queue
Worker (worker.py) — separate Railway service
    └── agent/
        ├── prompt_advisor.py     — LangGraph ReAct agent (experiments)
        ├── hypothesis_agent.py   — deepagents-based agent (traces, legacy)
        ├── experiment_tools.py   — 5 LangSmith tools for experiment analysis
        └── tools.py              — tools for trace analysis
db.py                             — shared DB helpers (no Flask, no gevent)
```

### Job queue flow
1. User submits question → `POST /experiments/analyze/start` → job inserted as `queued`
2. Worker polls Postgres every 2s, claims job (`FOR UPDATE SKIP LOCKED`)
3. LangGraph agent runs, writes step updates to DB as it goes
4. Browser polls `/jobs/<id>` every 2s, displays steps in real time
5. Job marked `done` or `failed`, result written to DB

---

## Getting started (local)

```bash
cd ~/Documents/traceiq
cp .env.example .env   # add your API keys
python3 server.py      # web server on :5050
python3 worker.py      # job worker (separate terminal)
# → open http://localhost:5050
```

**Required env vars:**
```
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://...   # optional locally, uses file fallback
```

---

## Deployed (Railway)

Live at: `https://web-production-65478.up.railway.app`

Three Railway services:
- **web** — Flask + gunicorn (gevent workers)
- **worker** — `python3 worker.py` (job processor)
- **Postgres** — shared DB for job queue + history

---

## UI flow

1. **Connect** — enter LangSmith API key + project name
2. **Overview** — 7-day sparkline, error rate, week-on-week comparison
3. **Experiments** — select dataset → select experiment → ask a question
4. **Analysis** — agent runs, steps appear in real time as it works
5. **Results** — overall scores, failure patterns, ranked recommendations
6. **History** — all past analyses saved to Postgres, viewable anytime

---

## What's built

### Core
- ✅ Experiment analysis with LangGraph ReAct agent
- ✅ Question-driven agent — plans tool calls based on what you ask
- ✅ Async job queue via Postgres (web enqueues, worker executes)
- ✅ Real-time step streaming — UI polls and shows agent progress
- ✅ History — save, view, delete past analyses (Postgres + file fallback)

### Agent tools (`experiment_tools.py`)
- `list_datasets` — list all LangSmith datasets
- `list_experiments` — list experiments for a dataset
- `fetch_experiment_rows` — aggregated scores (all 100 rows) + sample for analysis
- `get_failing_rows` — inputs/outputs for rows below a metric threshold (capped at 10)
- `compare_experiments` — side-by-side comparison of two experiments

### Data accuracy
- ✅ `fetch_experiment_rows` always computes aggregated scores across all 100 rows — matches LangSmith UI exactly
- ✅ `get_failing_rows` capped at 10 rows with 1000-char truncation — prevents context overflow
- ✅ `eq(is_root, true)` filter — fetches top-level traces only

### UX
- ✅ Project overview screen — sparkline chart, week-on-week deltas
- ✅ localStorage persistence — API key and project auto-fill on return
- ✅ Experiments-only flow — Traces tab removed

---

## Key design principles

1. **Read-only** — TraceIQ reads what LangSmith already computed. Never reruns experiments.
2. **Question-driven** — the agent adapts its investigation to your question.
3. **Evidence-based** — every finding cites real metric values and real examples.
4. **Agentic, not scripted** — Claude decides which tools to call and in what order.

---

## Tech stack

- **Backend**: Python 3.11+, Flask, gunicorn (gevent)
- **Agent framework**: LangGraph (`create_react_agent`) + LangChain Anthropic
- **LLM**: Anthropic Claude Sonnet
- **Observability**: LangSmith API
- **Database**: Postgres (Railway) with file fallback for local dev
- **Frontend**: Vanilla JS, SVG charts, polling-based job status
- **Hosting**: Railway (web + worker + Postgres)

---

## Architecture decisions

See [`docs/architecture-decisions.md`](docs/architecture-decisions.md) for invariants that must not be changed, including:

- Score computation must use all 100 rows (not the sample)
- Worker must never import `server.py` (gevent monkey patch breaks asyncio)
- `get_failing_rows` capped at 10 rows
- Streaming via `astream(stream_mode="updates")`

---

## Roadmap

- [ ] Multiple worker replicas for concurrent users (Railway: scale worker to N)
- [ ] Streaming intermediate results to UI (currently polls on completion)
- [ ] Compare two experiments side-by-side in UI
- [ ] Braintrust support
- [ ] Auth / multi-tenant
- [ ] Landing page
