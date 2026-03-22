# TraceIQ

**LangSmith tells you what happened. TraceIQ tells you why — and what to change.**

TraceIQ is a local web tool for AI product builders who use LangSmith. It sits on top of your existing traces and experiments and gives you agentic analysis: ask a question, get a real investigation with evidence and actionable recommendations.

---

## What it does

### Traces — Hypothesis Testing
State a belief about your agent's behavior. TraceIQ fetches your production traces and tells you if the data supports it.

> *"I think the agent is penalizing students for typos"*
> → Not supported — high confidence. Feedback is consistently rubric-focused, no evidence of grammar penalization.

**Analysis modes:**
- **Prompt change** — splits traces before/after a detected prompt change
- **Time split** — compares first half vs second half of the date range
- **No split** — scans all traces as one group
- **🔍 Deep analysis** — agentic mode: uses 5 tools (query, sample, classify, compute stats, compare groups) to investigate iteratively. Finds patterns humans miss.

### Experiments — Prompt Improvement
Connect to a LangSmith dataset + experiment. Ask a question. An AI agent reads your eval results, identifies failure patterns, and recommends specific prompt changes.

> *"Why is marks_exact_match low?"*
> → Agent fetches failing rows, reads actual student answers and grader outputs, identifies 3 root causes, proposes concrete prompt edits.

The agent plans its own investigation based on your question — it doesn't follow a fixed script. "Why is X low?" goes straight to failing rows. "Compare the last two experiments" uses the comparison tool.

---

## Architecture

```
browser (index.html)
    ↕ Server-Sent Events (streaming)
Flask server (server.py)
    ↕ subprocess.Popen (streams stderr live)
    ├── traceiq.py          — traces analysis (hypothesis testing)
    └── agent/
        ├── hypothesis_agent.py   — deep analysis agent (traces)
        ├── prompt_advisor.py     — prompt improvement agent (experiments)
        ├── run_advisor.py        — CLI entrypoint for prompt advisor
        ├── tools.py              — 5 agent tools for trace analysis
        └── experiment_tools.py  — 5 agent tools for experiment analysis
```

Both analysis flows stream live progress to the browser via Server-Sent Events. You see real log lines as the agent works — not a fake spinner.

---

## Getting started

```bash
cd ~/Documents/traceiq
python3 server.py
# → open http://localhost:5050
```

**You need:**
- LangSmith API key (`lsv2_pt_...`)
- Anthropic API key (for deep analysis + experiment advisor) — add to `.env`

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## UI flow

1. **Connect** — enter LangSmith API key + project name
2. **Overview** — see a 7-day sparkline of trace volume + error rate, week-over-week comparison
3. **Traces or Experiments** — switch modes from the overview screen
4. **Analyse** — enter a hypothesis/question, pick a mode, click Analyse
5. **Results** — verdict, evidence, recommendations
6. **History** — all past analyses saved locally, viewable anytime

---

## What's been built (as of 2026-03-22)

### Core
- ✅ Hypothesis testing over production traces (4 split modes)
- ✅ Deep analysis agent mode (LangChain deepagents SDK, 5 tools)
- ✅ Experiment analysis with prompt improvement advisor
- ✅ Question-driven agent — agent plans tool calls based on what you ask
- ✅ History — save, view, delete past analyses

### UX
- ✅ Project overview screen — sparkline chart, week-on-week deltas, key metrics
- ✅ Guided question suggestions — contextual chips generated from your live data
- ✅ localStorage persistence — API key and project auto-fill on return visits
- ✅ Mode tabs appear after login (not before)

### Streaming
- ✅ Server-Sent Events for both traces and experiments analysis
- ✅ Real `[TraceIQ]` log lines streamed live to the browser
- ✅ Tool-level progress: "Fetching failing rows for marks_exact_match...", "Found 8 failing rows out of 20 checked"

### Data
- ✅ `eq(is_root, true)` filter — fetches top-level traces only (matches LangSmith UI count)
- ✅ `get_experiment_results()` SDK call for accurate aggregated scores (matches LangSmith UI)
- ✅ Read-only — never re-runs experiments, never recomputes what LangSmith already has

---

## Key design principles

1. **Read-only** — TraceIQ is an analysis layer. It reads what LangSmith already computed. It never re-runs experiments or recomputes aggregates.
2. **Question-driven** — the agent adapts its investigation to what you're asking. Specific question → targeted tool calls. Open question → broad scan first.
3. **Evidence-based** — every finding cites actual data: real metric values, real input/output excerpts, real counts.
4. **Fast by default** — standard analysis is a single LLM call. Deep/agent analysis is opt-in.

---

## Tech stack

- **Backend**: Python 3, Flask
- **Agent framework**: LangChain deepagents SDK (built on LangGraph)
- **LLM**: Anthropic Claude (sonnet for analysis, haiku for classification)
- **Observability**: LangSmith API
- **Frontend**: Vanilla JS, SVG charts, Server-Sent Events
- **Tests**: pytest

---

## Roadmap

- [ ] Hosting (Railway/Render) — currently local only
- [ ] Auth — API keys stored in browser only, no accounts
- [ ] Streaming intermediate results (partial JSON as agent works)
- [ ] Compare two experiments side-by-side in UI
- [ ] Braintrust support
- [ ] Landing page
