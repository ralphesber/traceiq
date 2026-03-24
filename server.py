#!/usr/bin/env python3
# Gevent monkey patch — must be first, before any other imports
# subprocess=False: gevent's subprocess patch breaks Popen on Linux
try:
    from gevent import monkey
    monkey.patch_all(subprocess=False)
except ImportError:
    pass  # local dev without gevent

"""
TraceIQ server — minimal Flask server for the hypothesis testing UI.

Serves:
  GET  /                    → index.html
  GET  /static/*            → static files
  POST /analyze             → runs traceiq.py with hypothesis args, returns JSON result
  GET  /history             → list saved hypothesis results
  GET  /history/<filename>  → get a single saved result
  DELETE /history/<filename> → delete a saved result
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

_sessions: dict = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600  # 1 hour

def _create_session(api_key: str) -> str:
    if not api_key:
        raise ValueError("api_key cannot be empty")
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = {"api_key": api_key, "created_at": time.time()}
    return session_id

def _get_session_key(session_id: str):
    with _sessions_lock:
        s = _sessions.get(session_id)
        if not s:
            return None
        if time.time() - s["created_at"] > SESSION_TTL:
            del _sessions[session_id]
            return None
        return s["api_key"]

def _resolve_api_key(request_args=None, request_json=None):
    """Resolve api_key from session_id (preferred) or direct api_key (fallback for dev)."""
    # Try session_id first
    session_id = (request_args or {}).get("session_id", "")
    if session_id:
        key = _get_session_key(session_id)
        if key:
            return key
        return None  # invalid session
    # Fallback: direct api_key (local dev / curl)
    return (request_args or {}).get("api_key", "")


# ── Postgres history (falls back to files when DATABASE_URL not set) ──────

def _get_db_conn():
    """Return a psycopg2 connection or None if DATABASE_URL not set."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    try:
        import psycopg2
        # Railway appends ?sslmode=require — psycopg2 handles it
        return psycopg2.connect(url, sslmode="require")
    except Exception as e:
        print(f"[server] DB connect failed: {e}", flush=True)
        return None


def _init_db():
    """Create history and jobs tables if they don't exist. Called at startup."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        # Create history table + index
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
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS history_generated_at_idx
                        ON history (generated_at DESC)
                """)
        # Create jobs table + indexes
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
        print("[server] DB: schema ready (history + jobs)", flush=True)
    except Exception as e:
        print(f"[server] DB init failed: {e}", flush=True)
    finally:
        conn.close()


# ── Job queue helpers ─────────────────────────────────────────────────────

def _create_job(job_type: str, input_data: dict) -> str | None:
    """Insert a new queued job; returns job_id (UUID string) or None on failure."""
    import psycopg2.extras
    conn = _get_db_conn()
    if not conn:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (job_type, input, steps)
                    VALUES (%s, %s, %s)
                    RETURNING id::text
                    """,
                    (job_type, psycopg2.extras.Json(input_data), [])
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"[server] _create_job failed: {e}", flush=True)
        return None
    finally:
        conn.close()


def _update_job_status(job_id: str, status: str) -> None:
    """Update job status (queued → running → done/failed)."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = %s, updated_at = NOW() WHERE id = %s",
                    (status, job_id)
                )
    except Exception as e:
        print(f"[server] _update_job_status failed: {e}", flush=True)
    finally:
        conn.close()


def _append_job_step(job_id: str, step_text: str) -> None:
    """Append a progress step string to the job's steps array."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET steps = array_append(COALESCE(steps, '{}'), %s),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (step_text, job_id)
                )
    except Exception as e:
        print(f"[server] _append_job_step failed: {e}", flush=True)
    finally:
        conn.close()


def _complete_job(job_id: str, result: dict) -> None:
    """Mark job as done and write the final result JSON."""
    import psycopg2.extras
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'done', result = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (psycopg2.extras.Json(result), job_id)
                )
    except Exception as e:
        print(f"[server] _complete_job failed: {e}", flush=True)
    finally:
        conn.close()


def _fail_job(job_id: str, error: str) -> None:
    """Mark job as failed and store the error message."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', error = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error, job_id)
                )
    except Exception as e:
        print(f"[server] _fail_job failed: {e}", flush=True)
    finally:
        conn.close()


def _get_job(job_id: str) -> dict | None:
    """Return job row as dict {id, status, job_type, steps, result, error, created_at, updated_at} or None."""
    conn = _get_db_conn()
    if not conn:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, status, job_type, steps, result, error,
                           created_at, updated_at
                    FROM jobs WHERE id = %s
                    """,
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "status": row[1],
                    "job_type": row[2],
                    "steps": row[3] or [],
                    "result": row[4],
                    "error": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                    "updated_at": row[7].isoformat() if row[7] else None,
                }
    except Exception as e:
        print(f"[server] _get_job failed: {e}", flush=True)
        return None
    finally:
        conn.close()


app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
_init_db()


# ── Session endpoint ──────────────────────────────────────────────────────

@app.route("/api/session", methods=["POST"])
def create_session():
    data = request.get_json(force=True) or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    session_id = _create_session(api_key)
    return jsonify({"session_id": session_id})


# ── Static routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    # Don't let this catch /history/* routes
    if filename.startswith("history"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(BASE_DIR), filename)


# ── SSE helpers ───────────────────────────────────────────────────────────

def _parse_sse_line(line: str):
    """Parse a stderr line from traceiq.py into an SSE event dict, or None to skip."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("[TraceIQ]"):
        text = line[len("[TraceIQ]"):].strip()
        return {"type": "step", "text": text}
    return None


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ── Analysis ──────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    hypothesis = data.get("hypothesis", "").strip()
    api_key = data.get("api_key", "").strip()
    project = data.get("project", "").strip()
    days = int(data.get("days", 30))
    split_mode = data.get("split_mode", "prompt_change").strip()

    if split_mode not in ("prompt_change", "time_split", "none", "agent"):
        split_mode = "prompt_change"

    if not hypothesis:
        return jsonify({"error": "hypothesis is required"}), 400
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    if not project:
        return jsonify({"error": "project is required"}), 400

    cmd = [
        sys.executable,
        str(BASE_DIR / "traceiq.py"),
        "--project", project,
        "--api-key", api_key,
        "--hypothesis", hypothesis,
        "--days", str(days),
        "--split-mode", split_mode,
    ]

    env = os.environ.copy()
    # Pass through LLM keys from environment
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if key not in env and data.get(key):
            env[key] = data[key]

    print(f"[server] Running analysis: split_mode={split_mode} project={project}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Analysis timed out after 5 minutes"}), 504
    except Exception as e:
        return jsonify({"error": f"Failed to run analysis: {e}"}), 500

    if result.returncode != 0:
        stderr = result.stderr[-2000:] if result.stderr else ""
        return jsonify({"error": f"Analysis failed: {stderr}"}), 500

    # Try to read hypothesis_output.json (authoritative output)
    output_path = BASE_DIR / "hypothesis_output.json"
    if output_path.exists():
        try:
            with open(output_path) as f:
                output = json.load(f)

            # Save to history if successful (not an error result)
            if not output.get("error") and output.get("verdict") not in (None, "error"):
                try:
                    _save_to_history(output)
                except Exception as e:
                    print(f"[server] Warning: could not save to history: {e}", flush=True)

            return jsonify(output)
        except Exception as e:
            return jsonify({"error": f"Could not read output: {e}"}), 500

    # Fall back to stdout
    try:
        output = json.loads(result.stdout)
        return jsonify(output)
    except json.JSONDecodeError:
        return jsonify({"error": "Analysis produced invalid output", "raw": result.stdout[-1000:]}), 500


@app.route("/analyze/start", methods=["POST"])
def analyze_start():
    """
    POST /analyze/start
    Body: {api_key OR session_id, hypothesis, project, days, split_mode}
    Returns: {job_id}
    Enqueues a hypothesis analysis job; worker.py picks it up asynchronously.
    """
    data = request.get_json(force=True) or {}

    # Resolve API key
    api_key = (_resolve_api_key(
        request_args=data,
        request_json=data,
    ) or data.get("api_key", "")).strip()

    hypothesis = data.get("hypothesis", "").strip()
    project = data.get("project", "").strip()
    days = int(data.get("days", 30))
    split_mode = data.get("split_mode", "agent").strip()

    if not hypothesis:
        return jsonify({"error": "hypothesis is required"}), 400
    if not api_key:
        return jsonify({"error": "api_key is required (or valid session_id)"}), 400
    if not project:
        return jsonify({"error": "project is required"}), 400
    if split_mode not in ("prompt_change", "time_split", "none", "agent"):
        split_mode = "agent"

    job_input = {
        "api_key": api_key,
        "hypothesis": hypothesis,
        "project": project,
        "days": days,
        "split_mode": split_mode,
    }

    job_id = _create_job("hypothesis", job_input)
    if not job_id:
        return jsonify({"error": "Failed to create job — DATABASE_URL may not be set"}), 500

    print(f"[server] queued hypothesis job {job_id} project={project}", flush=True)
    return jsonify({"job_id": job_id})


@app.route("/analyze/stream", methods=["GET"])
def analyze_stream():
    """
    GET /analyze/stream?hypothesis=...&project=...&api_key=...&days=...&split_mode=...
    Streams real-time [TraceIQ] log lines as SSE step events.
    Final result is emitted as a 'result' event.
    """
    hypothesis = request.args.get("hypothesis", "").strip()
    api_key = (_resolve_api_key(request.args) or "").strip()
    project = request.args.get("project", "").strip()
    days = int(request.args.get("days", 30))
    split_mode = request.args.get("split_mode", "prompt_change").strip()

    if not hypothesis:
        return jsonify({"error": "hypothesis is required"}), 400
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    if not project:
        return jsonify({"error": "project is required"}), 400
    if split_mode not in ("prompt_change", "time_split", "none", "agent"):
        split_mode = "prompt_change"

    cmd = [
        sys.executable, str(BASE_DIR / "traceiq.py"),
        "--project", project,
        "--hypothesis", hypothesis,
        "--days", str(days),
        "--split-mode", split_mode,
        "--api-key", api_key,
    ]

    env = os.environ.copy()

    def generate():
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(BASE_DIR),
            )

            # Stream stderr with keepalive to prevent proxy timeouts
            import queue as _queue
            import threading as _threading

            line_q = _queue.Queue()

            def _reader(pipe, q):
                for line in pipe:
                    q.put(line)
                q.put(None)

            _threading.Thread(target=_reader, args=(proc.stderr, line_q), daemon=True).start()

            while True:
                try:
                    line = line_q.get(timeout=25)
                except _queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if line is None:
                    break
                event = _parse_sse_line(line)
                if event:
                    yield _sse_event(event)

            proc.wait(timeout=300)

            if proc.returncode != 0:
                yield _sse_event({"type": "error", "message": f"Analysis failed (exit {proc.returncode})"})
                return

            # Read final result
            output_path = BASE_DIR / "hypothesis_output.json"
            if output_path.exists():
                try:
                    with open(output_path) as f:
                        output = json.load(f)
                    # Save to history
                    if not output.get("error") and output.get("verdict") not in (None, "error"):
                        try:
                            _save_to_history(output)
                        except Exception:
                            pass
                    yield _sse_event({"type": "result", "data": output})
                except Exception as e:
                    yield _sse_event({"type": "error", "message": f"Could not read output: {e}"})
            else:
                yield _sse_event({"type": "error", "message": "No output produced"})

        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
            yield _sse_event({"type": "error", "message": "Analysis timed out after 5 minutes"})
        except GeneratorExit:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

    from flask import Response
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
    )


# ── History ───────────────────────────────────────────────────────────────

def _make_slug(hypothesis: str) -> str:
    """Turn hypothesis into a safe filename slug (first 40 chars)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", hypothesis[:40])
    return slug.strip("-").lower() or "hypothesis"


def _save_to_history(result: dict) -> str:
    """Save a hypothesis result. Uses Postgres if available, else local file."""
    hypothesis = result.get("hypothesis", "")
    slug = _make_slug(hypothesis)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{slug}.json"

    conn = _get_db_conn()
    if conn:
        try:
            generated_at = result.get("generated_at")
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO history
                            (filename, result_type, hypothesis, verdict, confidence,
                             project, traces_analyzed, generated_at, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (filename) DO NOTHING
                    """, (
                        filename,
                        result.get("result_type", "hypothesis"),
                        hypothesis,
                        result.get("verdict", ""),
                        result.get("confidence", ""),
                        result.get("project", ""),
                        result.get("traces_analyzed"),
                        generated_at,
                        json.dumps(result, default=str),
                    ))
            print(f"[server] Saved to DB history: {filename}", flush=True)
            return filename
        except Exception as e:
            print(f"[server] DB save failed, falling back to file: {e}", flush=True)
        finally:
            conn.close()

    # File fallback
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[server] Saved to file history: {filename}", flush=True)
    return filename


def _safe_filename(filename: str) -> bool:
    """Validate that a filename is safe (no path traversal etc.)."""
    return bool(re.match(r"^[\w\-\.]+\.json$", filename)) and ".." not in filename


def _require_session():
    """Return None if request has a valid session, else a 401 response."""
    session_id = request.args.get("session_id", "") or (request.get_json(silent=True) or {}).get("session_id", "")
    if not session_id or not _get_session_key(session_id):
        return jsonify({"error": "Authentication required — please connect with your LangSmith API key first"}), 401
    return None


@app.route("/history", methods=["GET"])
def list_history():
    err = _require_session()
    if err:
        return err
    conn = _get_db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT filename, result_type, hypothesis, question, verdict,
                           confidence, generated_at, project, experiment_name, traces_analyzed
                    FROM history
                    ORDER BY generated_at DESC
                    LIMIT 200
                """)
                rows = cur.fetchall()
            entries = []
            for row in rows:
                entries.append({
                    "filename": row[0],
                    "result_type": row[1] or "hypothesis",
                    "hypothesis": row[2] or "",
                    "question": row[3] or "",
                    "verdict": row[4] or "",
                    "confidence": row[5] or "",
                    "generated_at": row[6].isoformat() if row[6] else "",
                    "project": row[7] or "",
                    "experiment_name": row[8] or "",
                    "traces_analyzed": row[9] or 0,
                })
            return jsonify(entries)
        except Exception as e:
            print(f"[server] DB list failed, falling back to files: {e}", flush=True)
        finally:
            conn.close()

    # File fallback (existing logic)
    entries = []
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            entries.append({
                "filename": f.name,
                "result_type": data.get("result_type", "hypothesis"),
                "hypothesis": data.get("hypothesis", ""),
                "question": data.get("question", ""),
                "verdict": data.get("verdict", ""),
                "confidence": data.get("confidence", ""),
                "generated_at": data.get("generated_at", ""),
                "project": data.get("project", ""),
                "experiment_name": data.get("experiment_name", ""),
                "traces_analyzed": data.get("traces_analyzed", 0),
            })
        except Exception:
            continue
    return jsonify(entries)


@app.route("/history/<filename>", methods=["GET"])
def get_history_entry(filename):
    err = _require_session()
    if err:
        return err
    if not _safe_filename(filename):
        return jsonify({"error": "Invalid filename"}), 400

    conn = _get_db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM history WHERE filename = %s", (filename,))
                row = cur.fetchone()
            if row:
                return jsonify(json.loads(row[0]))
            return jsonify({"error": "Not found"}), 404
        except Exception as e:
            print(f"[server] DB get failed, falling back to file: {e}", flush=True)
        finally:
            conn.close()

    # File fallback
    path = HISTORY_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/history/<filename>", methods=["DELETE"])
def delete_history_entry(filename):
    err = _require_session()
    if err:
        return err
    if not _safe_filename(filename):
        return jsonify({"error": "Invalid filename"}), 400

    conn = _get_db_conn()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM history WHERE filename = %s", (filename,))
            print(f"[server] Deleted history entry from DB: {filename}", flush=True)
            return jsonify({"ok": True})
        except Exception as e:
            print(f"[server] DB delete failed, falling back to file: {e}", flush=True)
        finally:
            conn.close()

    # File fallback
    path = HISTORY_DIR / filename
    if path.exists():
        path.unlink()
        print(f"[server] Deleted history entry from file: {filename}", flush=True)
    return jsonify({"ok": True})


# ── Experiments ───────────────────────────────────────────────────────────

def _ls_ssl_context():
    """Return an SSL context that works on macOS."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    ctx = ssl.create_default_context()
    try:
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    except Exception:
        pass
    return ctx


def _ls_get(api_key: str, path: str, params: dict = None):
    """Make a GET request to the LangSmith API."""
    import urllib.request
    import urllib.parse
    url = "https://api.smith.langchain.com/api/v1" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=30, context=_ls_ssl_context()) as resp:
        return json.loads(resp.read().decode())


@app.route("/experiments/datasets", methods=["GET"])
def experiments_datasets():
    api_key = (_resolve_api_key(request.args) or "").strip()
    if not api_key:
        return jsonify({"error": "api_key query parameter is required"}), 400
    try:
        data = _ls_get(api_key, "/datasets")
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("datasets", data.get("results", []))
        else:
            rows = []
        result = []
        for d in rows:
            result.append({
                "id": d.get("id", ""),
                "name": d.get("name", ""),
                "example_count": d.get("example_count", 0),
                "session_count": d.get("session_count", 0),
                "last_session_start_time": d.get("last_session_start_time"),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/experiments/list", methods=["GET"])
def experiments_list():
    api_key = (_resolve_api_key(request.args) or "").strip()
    dataset_id = request.args.get("dataset_id", "").strip()
    if not api_key:
        return jsonify({"error": "api_key query parameter is required"}), 400
    if not dataset_id:
        return jsonify({"error": "dataset_id query parameter is required"}), 400
    try:
        data = _ls_get(api_key, "/sessions", params={"reference_dataset_id": dataset_id})
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("sessions", data.get("results", []))
        else:
            rows = []
        result = []
        for s in rows:
            result.append({
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "created_at": s.get("start_time") or s.get("created_at"),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/experiments/analyze", methods=["POST"])
def experiments_analyze():
    data = request.get_json(force=True)

    api_key = data.get("api_key", "").strip()
    dataset_id = data.get("dataset_id", "").strip()
    experiment_id = data.get("experiment_id", "").strip()
    question = data.get("question", "").strip()

    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not experiment_id:
        return jsonify({"error": "experiment_id is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    env = os.environ.copy()
    env["LANGSMITH_API_KEY"] = api_key

    print(f"[server] Running experiment analysis: dataset={dataset_id} experiment={experiment_id}", flush=True)

    # Run in a subprocess to avoid blocking and get clean env
    import subprocess
    script = f"""
import sys, os, json
sys.path.insert(0, {repr(str(BASE_DIR))})
os.environ['ANTHROPIC_API_KEY'] = {repr(env.get('ANTHROPIC_API_KEY', ''))}
from agent.prompt_advisor import run_prompt_advisor
result = run_prompt_advisor(
    api_key={repr(api_key)},
    dataset_id={repr(dataset_id)},
    experiment_id={repr(experiment_id)},
    question={repr(question)},
)
print(json.dumps(result, default=str))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Analysis timed out after 10 minutes"}), 504
    except Exception as e:
        return jsonify({"error": f"Failed to run analysis: {e}"}), 500

    if proc.returncode != 0:
        stderr = proc.stderr[-2000:] if proc.stderr else ""
        return jsonify({"error": f"Analysis failed: {stderr}"}), 500

    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return jsonify({"error": "Analysis produced invalid output", "raw": proc.stdout[-1000:]}), 500

    # Save to history with exp_ prefix
    if not output.get("error"):
        try:
            _save_experiment_to_history(output)
        except Exception as e:
            print(f"[server] Warning: could not save experiment to history: {e}", flush=True)

    return jsonify(output)


@app.route("/experiments/analyze/start", methods=["POST"])
def experiments_analyze_start():
    """
    POST /experiments/analyze/start
    Body: {api_key OR session_id, dataset_id, experiment_id, question}
    Returns: {job_id}
    Enqueues an experiment analysis job; worker.py picks it up asynchronously.
    """
    data = request.get_json(force=True) or {}

    api_key = (_resolve_api_key(
        request_args=data,
        request_json=data,
    ) or data.get("api_key", "")).strip()

    dataset_id = data.get("dataset_id", "").strip()
    experiment_id = data.get("experiment_id", "").strip()
    question = data.get("question", "").strip()

    if not api_key:
        return jsonify({"error": "api_key is required (or valid session_id)"}), 400
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not experiment_id:
        return jsonify({"error": "experiment_id is required"}), 400
    if not question:
        question = "Analyze this experiment and recommend prompt improvements"

    job_input = {
        "api_key": api_key,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "question": question,
    }

    job_id = _create_job("experiment", job_input)
    if not job_id:
        return jsonify({"error": "Failed to create job — DATABASE_URL may not be set"}), 500

    print(f"[server] queued experiment job {job_id} dataset={dataset_id}", flush=True)
    return jsonify({"job_id": job_id})


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    GET /jobs/<job_id>
    Returns: {id, status, job_type, steps, result, error, created_at, updated_at}
    Polled by the UI every 2s to track job progress.
    """
    # Validate UUID format to prevent injection
    import re as _re
    if not _re.match(r'^[0-9a-f-]{36}$', job_id):
        return jsonify({"error": "Invalid job ID"}), 400
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/experiments/analyze/stream", methods=["GET"])
def experiments_analyze_stream():
    """
    GET /experiments/analyze/stream?api_key=...&dataset_id=...&experiment_id=...&question=...
    Streams real-time [TraceIQ] log lines as SSE step events.
    Final result emitted as a 'result' event.
    """
    api_key = (_resolve_api_key(request.args) or "").strip()
    dataset_id = request.args.get("dataset_id", "").strip()
    experiment_id = request.args.get("experiment_id", "").strip()
    question = request.args.get("question", "").strip()

    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not experiment_id:
        return jsonify({"error": "experiment_id is required"}), 400

    env = os.environ.copy()
    env["LANGSMITH_API_KEY"] = api_key

    cmd = [
        sys.executable, str(BASE_DIR / "agent" / "run_advisor.py"),
        "--api-key", api_key,
        "--dataset-id", dataset_id,
        "--experiment-id", experiment_id,
        "--question", question or "Analyze this experiment and recommend prompt improvements",
    ]

    def generate():
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(BASE_DIR),
            )

            import queue as _queue
            import threading as _threading

            line_q = _queue.Queue()

            def _reader(pipe, q):
                for line in pipe:
                    q.put(line)
                q.put(None)  # sentinel

            _threading.Thread(target=_reader, args=(proc.stderr, line_q), daemon=True).start()

            # Read stderr lines; send keepalive comment every 25s if agent is quiet
            while True:
                try:
                    line = line_q.get(timeout=25)
                except _queue.Empty:
                    yield ": keepalive\n\n"  # SSE comment — keeps connection alive
                    continue
                if line is None:
                    break
                event = _parse_sse_line(line)
                if event:
                    yield _sse_event(event)

            proc.wait(timeout=600)

            if proc.returncode != 0:
                yield _sse_event({"type": "error", "message": f"Analysis failed (exit {proc.returncode})"})
                return

            try:
                raw = proc.stdout.read()
                output = json.loads(raw)
            except Exception as e:
                yield _sse_event({"type": "error", "message": f"Could not parse output: {e}"})
                return

            if not output.get("error"):
                try:
                    _save_experiment_to_history(output)
                except Exception as e:
                    print(f"[server] Warning: could not save experiment to history: {e}", flush=True)

            yield _sse_event({"type": "result", "data": output})

        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
            yield _sse_event({"type": "error", "message": "Analysis timed out after 10 minutes"})
        except GeneratorExit:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

    from flask import Response
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


def _save_experiment_to_history(result: dict) -> str:
    """Save an experiment result. Uses Postgres if available, else local file."""
    question = result.get("question", "")
    slug = _make_slug(question or result.get("experiment_name", "experiment"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"exp_{timestamp}_{slug}.json"

    conn = _get_db_conn()
    if conn:
        try:
            generated_at = result.get("generated_at")
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO history
                            (filename, result_type, hypothesis, question, verdict, confidence,
                             project, experiment_name, traces_analyzed, generated_at, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (filename) DO NOTHING
                    """, (
                        filename,
                        "experiment_analysis",
                        None,
                        question,
                        result.get("verdict", ""),
                        result.get("confidence", ""),
                        result.get("project", result.get("dataset_name", "")),
                        result.get("experiment_name", ""),
                        result.get("traces_analyzed"),
                        generated_at,
                        json.dumps(result, default=str),
                    ))
            print(f"[server] Saved to DB history: {filename}", flush=True)
            return filename
        except Exception as e:
            print(f"[server] DB save failed, falling back to file: {e}", flush=True)
        finally:
            conn.close()

    # File fallback
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[server] Saved to file history: {filename}", flush=True)
    return filename



# ── Overview ─────────────────────────────────────────────────────────────────────────

def _ls_post(api_key: str, path: str, payload: dict):
    """POST to LangSmith API."""
    import urllib.request
    url = "https://api.smith.langchain.com/api/v1" + path
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25, context=_ls_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _ls_resolve_session(api_key: str, project_name: str) -> str:
    """Resolve a LangSmith project name to a tracing session ID."""
    # Use tenant_id-based endpoint that traceiq.py uses
    data = _ls_get(api_key, "/sessions", params={"name": project_name})
    sessions = data if isinstance(data, list) else data.get("results", data.get("sessions", []))
    # Filter to non-experiment sessions (no reference_dataset_id)
    tracing_sessions = [s for s in sessions if not s.get("reference_dataset_id")]
    if tracing_sessions:
        return tracing_sessions[0]["id"]
    if sessions:
        return sessions[0]["id"]
    raise ValueError(f"No project found with name '{project_name}'")


def _ls_fetch_runs_window(api_key: str, session_id: str, start: datetime, end: datetime, limit: int = 100) -> list:
    """Fetch root runs from LangSmith for a specific time window."""
    payload = {
        "session": [session_id],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "filter": "eq(is_root, true)",
        "limit": limit,
    }
    data = _ls_post(api_key, "/runs/query", payload)
    return data.get("runs", data) if isinstance(data, dict) else data


@app.route("/overview", methods=["GET"])
def overview():
    """
    GET /overview?project=<name>&api_key=<key>

    Returns daily_stats + summary + comparison + latest_experiment for the sparkline UI.
    """
    api_key = (_resolve_api_key(request.args) or "").strip()
    project = request.args.get("project", "").strip()
    if not api_key or not project:
        return jsonify({"error": "api_key and project are required"}), 400

    from datetime import timedelta, datetime as _dt

    now = datetime.now(timezone.utc)
    start_14d = (now - timedelta(days=14)).isoformat()

    # Fetch up to 500 root runs for last 14 days
    try:
        # Try resolving session first (some projects need session ID)
        try:
            session_id = _ls_resolve_session(api_key, project)
            runs_data = _ls_post(api_key, "/runs/query", {
                "session": [session_id],
                "filter": "eq(is_root, true)",
                "limit": 100,
                "start_time": start_14d,
            })
        except Exception:
            runs_data = {"runs": []}
        runs = runs_data if isinstance(runs_data, list) else runs_data.get("runs", [])
    except Exception as e:
        return jsonify({"error": f"Failed to fetch traces: {e}"}), 500

    # Group by day
    day_buckets: dict = {}
    for run in runs:
        ts = run.get("start_time") or run.get("created_at") or ""
        if not ts:
            continue
        try:
            if isinstance(ts, str):
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                d = _dt.fromisoformat(ts)
            else:
                d = ts
            day_key = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        if day_key not in day_buckets:
            day_buckets[day_key] = {"count": 0, "errors": 0}
        day_buckets[day_key]["count"] += 1
        status = (run.get("status") or "").lower()
        error = run.get("error")
        if status == "error" or (error and error not in (None, "", "null")):
            day_buckets[day_key]["errors"] += 1

    # Build daily_stats for last 7 days (fill gaps with zeros)
    daily_stats = []
    for i in range(6, -1, -1):
        dk = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        b = day_buckets.get(dk, {"count": 0, "errors": 0})
        daily_stats.append({"date": dk, "count": b["count"], "errors": b["errors"]})

    # Summary (last 7 days)
    total_7d = sum(d["count"] for d in daily_stats)
    total_errors_7d = sum(d["errors"] for d in daily_stats)
    error_rate = round(total_errors_7d / total_7d * 100, 1) if total_7d > 0 else 0.0

    # Week-on-week comparison
    prev_total = sum(
        day_buckets.get((now - timedelta(days=i)).strftime("%Y-%m-%d"), {"count": 0})["count"]
        for i in range(14, 7, -1)
    )
    prev_errors_total = sum(
        day_buckets.get((now - timedelta(days=i)).strftime("%Y-%m-%d"), {"errors": 0})["errors"]
        for i in range(14, 7, -1)
    )
    count_delta = round((total_7d - prev_total) / prev_total * 100, 1) if prev_total > 0 else None
    prev_err_rate = round(prev_errors_total / prev_total * 100, 1) if prev_total > 0 else None
    curr_err_rate = round(total_errors_7d / total_7d * 100, 1) if total_7d > 0 else None
    error_rate_delta = round(curr_err_rate - prev_err_rate, 1) if (
        prev_err_rate is not None and curr_err_rate is not None
    ) else None

    comparison = {
        "prev_total": prev_total,
        "curr_total": total_7d,
        "count_delta": count_delta,
        "prev_error_rate": prev_err_rate,
        "curr_error_rate": curr_err_rate,
        "error_rate_delta": error_rate_delta,
    }

    # Latest experiment avg score (best-effort)
    latest_experiment = None
    try:
        exp_data = _ls_get(api_key, "/sessions", params={"project_name": project, "limit": 10})
        sessions = exp_data if isinstance(exp_data, list) else exp_data.get("sessions", [])
        for s in sessions:
            fb = s.get("feedback_stats") or {}
            if fb:
                scores = {}
                for k, v in fb.items():
                    avg = v.get("avg") if isinstance(v, dict) else None
                    if avg is not None:
                        scores[k] = round(float(avg), 3)
                if scores:
                    latest_experiment = {"name": s.get("name", ""), "scores": scores}
                    break
    except Exception:
        pass  # best-effort

    print(f"[server] Overview: project={project} runs={len(runs)} 7d={total_7d} err={error_rate}%", flush=True)
    return jsonify({
        "daily_stats": daily_stats,
        "summary": {"total": total_7d, "error_rate": error_rate},
        "comparison": comparison,
        "latest_experiment": latest_experiment,
    })


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[TraceIQ] Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
