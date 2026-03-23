# Postgres History Implementation Plan
Date: 2026-03-23

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace file-based history with Postgres so history survives deploys and scales across instances. Falls back to files when DATABASE_URL is not set.

**Architecture:** psycopg2 connection from DATABASE_URL env var. Auto-creates table on startup. All 5 history operations replaced with SQL equivalents. File fallback for local dev.

**Tech Stack:** psycopg2-binary, Postgres (Railway add-on), existing Flask server

---

## File Map
- Modify: `server.py` — replace history functions + endpoints
- Modify: `requirements.txt` — add psycopg2-binary
- Create: `tests/test_history_db.py` — unit tests for DB helpers

---

### Task 1: DB helpers + schema init in server.py

**Files:**
- Modify: `server.py`
- Modify: `requirements.txt`
- Create: `tests/test_history_db.py`

- [ ] **Install psycopg2-binary**
```bash
pip3 install psycopg2-binary
```

- [ ] **Write failing tests** in `tests/test_history_db.py`:
```python
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_get_db_conn_returns_none_without_env(monkeypatch):
    """Returns None when DATABASE_URL not set"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from server import _get_db_conn
    assert _get_db_conn() is None

def test_history_uses_file_fallback_without_db(monkeypatch, tmp_path):
    """Falls back to file when no DATABASE_URL"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import server
    # Patch HISTORY_DIR to tmp_path
    original = server.HISTORY_DIR
    server.HISTORY_DIR = tmp_path
    try:
        filename = server._save_to_history({
            "hypothesis": "test hypothesis",
            "verdict": "supported",
            "confidence": "high",
            "project": "test",
            "traces_analyzed": 10,
            "generated_at": "2026-03-23T00:00:00Z",
        })
        assert filename is not None
        assert (tmp_path / filename).exists()
    finally:
        server.HISTORY_DIR = original
```

- [ ] **Run tests — confirm FAIL**
```bash
cd ~/Documents/traceiq && python3 -m pytest tests/test_history_db.py -v
```
Expected: ImportError or AttributeError

- [ ] **Add DB helpers to server.py** (after the session store functions, before route definitions):

```python
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
    """Create history table if it doesn't exist. Called at startup."""
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
                    );
                    CREATE INDEX IF NOT EXISTS history_generated_at_idx
                        ON history (generated_at DESC);
                """)
        print("[server] DB: history table ready", flush=True)
    except Exception as e:
        print(f"[server] DB init failed: {e}", flush=True)
    finally:
        conn.close()
```

- [ ] **Call `_init_db()` at startup** — add after `HISTORY_DIR.mkdir(exist_ok=True)`:
```python
_init_db()
```

- [ ] **Run tests — confirm PASS**
```bash
cd ~/Documents/traceiq && python3 -m pytest tests/test_history_db.py tests/test_streaming.py -v
```
Expected: all pass

- [ ] **Add psycopg2-binary to requirements.txt**
```bash
pip3 install psycopg2-binary
pip3 freeze | grep psycopg2 >> ~/Documents/traceiq/requirements.txt
```

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add server.py requirements.txt tests/test_history_db.py
git commit -m "feat: add Postgres DB helpers and schema init with file fallback"
```

---

### Task 2: Replace _save_to_history and _save_experiment_to_history

**Files:**
- Modify: `server.py`

Find the existing `_save_to_history()` function and replace its body to try DB first, fall back to file:

- [ ] **Replace `_save_to_history()`**

```python
def _save_to_history(result: dict) -> str:
    """Save a hypothesis result. Uses Postgres if available, else local file."""
    hypothesis = result.get("hypothesis", "")
    slug = _make_slug(hypothesis)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{slug}.json"

    conn = _get_db_conn()
    if conn:
        try:
            import psycopg2.extras
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
```

- [ ] **Replace `_save_experiment_to_history()`** similarly:

```python
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
```

- [ ] **Verify syntax**
```bash
cd ~/Documents/traceiq && python3 -c "import server; print('OK')"
```

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add server.py
git commit -m "feat: _save_to_history and _save_experiment_to_history use Postgres with file fallback"
```

---

### Task 3: Replace GET /history, GET /history/<filename>, DELETE /history/<filename>

**Files:**
- Modify: `server.py`

Replace the 3 history read/delete endpoints to try DB first, fall back to files.

- [ ] **Replace `list_history()` (GET /history)**:

```python
@app.route("/history", methods=["GET"])
def list_history():
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
```

- [ ] **Replace `get_history_entry()` (GET /history/<filename>)**:

```python
@app.route("/history/<filename>", methods=["GET"])
def get_history_entry(filename):
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
```

- [ ] **Replace `delete_history_entry()` (DELETE /history/<filename>)**:

```python
@app.route("/history/<filename>", methods=["DELETE"])
def delete_history_entry(filename):
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
```

- [ ] **Verify syntax + all tests pass**
```bash
cd ~/Documents/traceiq && python3 -c "import server; print('OK')"
python3 -m pytest tests/ -v 2>&1 | tail -15
```

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add server.py
git commit -m "feat: history endpoints use Postgres with file fallback for local dev"
git push
```
