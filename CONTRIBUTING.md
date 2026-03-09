# Contributing to TraceIQ

## Code Principles

These apply to all code in this repo — engine and dashboard.

### General
1. **Tests for core logic** — unit tests for analysis functions (percentile calc, regression detection, health score derivation, JSON schema transformation). No need to test third-party rendering.
2. **Loud failures for config/auth, graceful degradation for data** — if the API key is wrong, crash loudly. If a trace is missing a field, log a warning and continue.
3. **Minimal dependencies** — stdlib + `requests` for the engine; CDN-only (Chart.js) for the dashboard. No new deps without discussion.
4. **Type hints everywhere** — not just public functions. Full typing on all functions in the engine. `int | None` is not the same as `int`.
5. **Docstrings on public functions** — one-liner minimum; full Args/Returns for anything complex.
6. **No dead code** — if it's commented out or unused, delete it. Git has history.
7. **No silent failures** — if the JSON schema is malformed, the dashboard must surface an error. Never render blank charts with no explanation.
8. **Single source of truth** — health score is derived once (dashboard layer), not recalculated in multiple places.

### Frontend
9. **No inline styles in JS** — CSS classes only. Chart.js color configs are the exception (Chart.js requires them), but keep them in a `CHART_THEME` const at the top of the file.
10. **Schema is the contract** — any change to the JSON schema requires sign-off from both engine owner (Jarvis) and dashboard owner (Jax). This is the API boundary.

### Repo
11. **README-driven** — if someone clones this repo and can't get it running in under 2 minutes following the README, that's a bug.
12. **Commit messages** — `type: description` format. Types: `feat`, `fix`, `docs`, `refactor`, `test`. Keep descriptions under 72 chars.

---

## JSON Schema Contract (v1)

The engine outputs this schema via `--output json`. The dashboard consumes it. Neither side changes this without bilateral sign-off.

```json
{
  "meta": {
    "project": "string",
    "window_days": "int",
    "generated_at": "ISO8601"
  },
  "alerts": [
    {
      "severity": "critical | warning | info",
      "type": "latency_regression | error_rate_increase | model_change | prompt_change",
      "message": "string"
    }
  ],
  "latency": {
    "p50": "float (seconds)",
    "p95": "float (seconds)",
    "p99": "float (seconds)",
    "prev_p50": "float (seconds)",
    "prev_p95": "float (seconds)",
    "prev_p99": "float (seconds)",
    "p95_change_pct": "float"
  },
  "cost": {
    "total_tokens": "int",
    "estimated_usd": "float",
    "per_run_usd": "float",
    "primary_model": "string"
  },
  "volume": {
    "total": "int",
    "errors": "int",
    "error_rate_pct": "float",
    "prev_error_rate_pct": "float"
  },
  "changes": {
    "prompts": [{ "date": "YYYY-MM-DD", "from_hash": "string", "to_hash": "string" }],
    "models": [{ "date": "YYYY-MM-DD", "from_model": "string", "to_model": "string" }]
  },
  "errors": {
    "top_patterns": [{ "message": "string", "count": "int", "pct": "float" }]
  }
}
```

Note: `health_score` is **not** in the schema. It is derived by the dashboard from `alerts`, `volume.error_rate_pct`, and `latency.p95`. Formula: start at 100, -15 per critical alert, -8 per warning, -10 if `error_rate_pct > 5`, -5 if `p95 > 2.0s`. Floor at 0. Show component breakdown on hover.

---

## Running Locally

### Engine
```bash
pip install -r requirements.txt
python3 traceiq.py --demo --project my-agent       # demo mode, no API key needed
python3 traceiq.py --project my-agent --output json > output.json  # real data
```

### Dashboard
Open `dashboard/index.html` in a browser. Use the file upload button to load `output.json`.
