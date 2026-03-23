# Postgres History — Design Spec
Date: 2026-03-23

## Problem
History is stored as JSON files in `history/` on local disk. On Railway (and any cloud host), the filesystem is ephemeral — every deploy wipes it. Scaling to multiple instances splits history across workers. Not acceptable for a team tool.

## Solution: Postgres via DATABASE_URL

Replace file-based history with a Postgres table. Railway provides a free Postgres add-on with a `DATABASE_URL` env var. The UI stays completely unchanged — only `server.py` changes.

## Schema

```sql
CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    result_type TEXT,
    hypothesis TEXT,
    question TEXT,
    verdict TEXT,
    confidence TEXT,
    project TEXT,
    experiment_name TEXT,
    traces_analyzed INTEGER,
    generated_at TIMESTAMP WITH TIME ZONE,
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS history_generated_at_idx ON history (generated_at DESC);
```

`filename` stays as the public identifier (e.g. `20260323T182500Z_why-is-marks-low.json`) so the UI/API surface doesn't change.
`data` stores the full result JSON for retrieval.

## Fallback behavior

If `DATABASE_URL` is not set (local dev), fall back to file-based history. This means:
- Local dev works exactly as before
- Production (Railway) uses Postgres automatically
- No breaking change

## Functions to replace in server.py

| Current | New |
|---|---|
| `_save_to_history(result)` | INSERT into history, fall back to file |
| `_save_experiment_to_history(result)` | INSERT into history, fall back to file |
| `GET /history` | SELECT from history ORDER BY generated_at DESC |
| `GET /history/<filename>` | SELECT WHERE filename = ? |
| `DELETE /history/<filename>` | DELETE WHERE filename = ? |

## DB connection

Use `psycopg2-binary`. Get connection string from `os.environ.get("DATABASE_URL")`.

Auto-create the table on startup (`CREATE TABLE IF NOT EXISTS`).

Use a simple connection helper — no ORM needed:

```python
def _get_db_conn():
    import psycopg2
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    return psycopg2.connect(url)
```

## What does NOT change
- UI (index.html) — zero changes
- History endpoint paths and response format
- `filename` as identifier
- Local dev behavior (falls back to files)
