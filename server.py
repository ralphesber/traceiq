#!/usr/bin/env python3
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
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


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


@app.route("/analyze/stream", methods=["GET"])
def analyze_stream():
    """
    GET /analyze/stream?hypothesis=...&project=...&api_key=...&days=...&split_mode=...
    Streams real-time [TraceIQ] log lines as SSE step events.
    Final result is emitted as a 'result' event.
    """
    hypothesis = request.args.get("hypothesis", "").strip()
    api_key = request.args.get("api_key", "").strip()
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

            # Stream stderr line by line
            for line in proc.stderr:
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
        }
    )


# ── History ───────────────────────────────────────────────────────────────

def _make_slug(hypothesis: str) -> str:
    """Turn hypothesis into a safe filename slug (first 40 chars)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", hypothesis[:40])
    return slug.strip("-").lower() or "hypothesis"


def _save_to_history(result: dict) -> str:
    """Save a result dict to the history directory. Returns the filename."""
    hypothesis = result.get("hypothesis", "")
    slug = _make_slug(hypothesis)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{slug}.json"
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[server] Saved to history: {filename}", flush=True)
    return filename


def _safe_filename(filename: str) -> bool:
    """Validate that a filename is safe (no path traversal etc.)."""
    return bool(re.match(r"^[\w\-\.]+\.json$", filename)) and ".." not in filename


@app.route("/history", methods=["GET"])
def list_history():
    entries = []
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            result_type = data.get("result_type", "hypothesis")
            entries.append({
                "filename": f.name,
                "hypothesis": data.get("hypothesis", ""),
                "verdict": data.get("verdict", ""),
                "confidence": data.get("confidence", ""),
                "generated_at": data.get("generated_at", ""),
                "project": data.get("project", ""),
                "traces_analyzed": data.get("traces_analyzed", 0),
                "split_mode": data.get("split_mode", ""),
                "result_type": result_type,
                # Experiment-specific fields
                "experiment_name": data.get("experiment_name", ""),
                "dataset_name": data.get("dataset_name", ""),
                "question": data.get("question", ""),
            })
        except Exception:
            continue
    return jsonify(entries)


@app.route("/history/<filename>", methods=["GET"])
def get_history_entry(filename):
    if not _safe_filename(filename):
        return jsonify({"error": "Invalid filename"}), 400
    path = HISTORY_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/history/<filename>", methods=["DELETE"])
def delete_history_entry(filename):
    if not _safe_filename(filename):
        return jsonify({"error": "Invalid filename"}), 400
    path = HISTORY_DIR / filename
    if path.exists():
        path.unlink()
        print(f"[server] Deleted history entry: {filename}", flush=True)
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
    api_key = request.args.get("api_key", "").strip()
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
    api_key = request.args.get("api_key", "").strip()
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


def _save_experiment_to_history(result: dict) -> str:
    """Save an experiment result to the history directory. Returns the filename."""
    exp_name = result.get("experiment_name", "experiment")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", exp_name[:40]).strip("-").lower() or "experiment"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"exp_{timestamp}_{slug}.json"
    path = HISTORY_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[server] Saved experiment to history: {filename}", flush=True)
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
    api_key = request.args.get("api_key", "").strip()
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
