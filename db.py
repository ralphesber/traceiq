"""
TraceIQ DB helpers — shared between server.py and worker.py.

Intentionally has NO Flask imports and NO gevent monkey patching.
worker.py imports from here, not from server.py, to avoid the gevent
monkey patch that server.py triggers at import time.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)


def _get_db_conn():
    """Return a psycopg2 connection or None if DATABASE_URL not set."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    try:
        import psycopg2
        return psycopg2.connect(url, sslmode="require")
    except Exception as e:
        print(f"[db] connect failed: {e}", flush=True)
        return None


def _init_db():
    """Create history and jobs tables if they don't exist."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
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
                        generated_at TIMESTAMPTZ,
                        data JSONB NOT NULL
                    )
                """)
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS history_generated_at_idx ON history (generated_at DESC)")
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        status TEXT NOT NULL DEFAULT 'queued',
                        job_type TEXT NOT NULL,
                        input JSONB NOT NULL,
                        steps TEXT[],
                        result JSONB,
                        error TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status)")
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at DESC)")
        print("[db] schema ready (history + jobs)", flush=True)
    except Exception as e:
        print(f"[db] init failed: {e}", flush=True)
    finally:
        conn.close()


def _create_job(job_type: str, input_data: dict) -> str | None:
    import psycopg2.extras
    conn = _get_db_conn()
    if not conn:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO jobs (job_type, input, steps) VALUES (%s, %s, %s) RETURNING id::text",
                    (job_type, psycopg2.extras.Json(input_data), [])
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"[db] _create_job failed: {e}", flush=True)
        return None
    finally:
        conn.close()


def _append_job_step(job_id: str, step_text: str) -> None:
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET steps = array_append(COALESCE(steps, '{}'), %s), updated_at = NOW() WHERE id = %s",
                    (step_text, job_id)
                )
    except Exception as e:
        print(f"[db] _append_job_step failed: {e}", flush=True)
    finally:
        conn.close()


def _complete_job(job_id: str, result: dict) -> None:
    import psycopg2.extras
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = 'done', result = %s, updated_at = NOW() WHERE id = %s",
                    (psycopg2.extras.Json(result), job_id)
                )
    except Exception as e:
        print(f"[db] _complete_job failed: {e}", flush=True)
    finally:
        conn.close()


def _fail_job(job_id: str, error: str) -> None:
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = 'failed', error = %s, updated_at = NOW() WHERE id = %s",
                    (error, job_id)
                )
    except Exception as e:
        print(f"[db] _fail_job failed: {e}", flush=True)
    finally:
        conn.close()


def _get_job(job_id: str) -> dict | None:
    conn = _get_db_conn()
    if not conn:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text, status, job_type, steps, result, error, created_at, updated_at FROM jobs WHERE id = %s",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0], "status": row[1], "job_type": row[2],
                    "steps": row[3] or [], "result": row[4], "error": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                    "updated_at": row[7].isoformat() if row[7] else None,
                }
    except Exception as e:
        print(f"[db] _get_job failed: {e}", flush=True)
        return None
    finally:
        conn.close()


def _make_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text[:40])
    return slug.strip("-").lower() or "item"


def _save_to_history(result: dict) -> str:
    hypothesis = result.get("hypothesis", "")
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_make_slug(hypothesis)}.json"
    conn = _get_db_conn()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO history (filename, result_type, hypothesis, verdict, confidence, project, traces_analyzed, generated_at, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (filename) DO NOTHING
                    """, (filename, result.get("result_type", "hypothesis"), hypothesis,
                          result.get("verdict", ""), result.get("confidence", ""),
                          result.get("project", ""), result.get("traces_analyzed"),
                          result.get("generated_at"), json.dumps(result, default=str)))
            return filename
        except Exception as e:
            print(f"[db] save_to_history failed: {e}", flush=True)
        finally:
            conn.close()
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return filename


def _save_experiment_to_history(result: dict) -> str:
    question = result.get("question", "")
    slug = _make_slug(question or result.get("experiment_name", "experiment"))
    filename = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    conn = _get_db_conn()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO history (filename, result_type, hypothesis, question, verdict, confidence, project, experiment_name, traces_analyzed, generated_at, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (filename) DO NOTHING
                    """, (filename, "experiment_analysis", None, question,
                          result.get("verdict", ""), result.get("confidence", ""),
                          result.get("project", result.get("dataset_name", "")),
                          result.get("experiment_name", ""), result.get("traces_analyzed"),
                          result.get("generated_at"), json.dumps(result, default=str)))
            return filename
        except Exception as e:
            print(f"[db] save_experiment_to_history failed: {e}", flush=True)
        finally:
            conn.close()
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return filename
