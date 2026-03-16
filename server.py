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


# ── Analysis ──────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    hypothesis = data.get("hypothesis", "").strip()
    api_key = data.get("api_key", "").strip()
    project = data.get("project", "").strip()
    days = int(data.get("days", 30))
    split_mode = data.get("split_mode", "prompt_change").strip()

    if split_mode not in ("prompt_change", "time_split", "none"):
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
            entries.append({
                "filename": f.name,
                "hypothesis": data.get("hypothesis", ""),
                "verdict": data.get("verdict", ""),
                "confidence": data.get("confidence", ""),
                "generated_at": data.get("generated_at", ""),
                "project": data.get("project", ""),
                "traces_analyzed": data.get("traces_analyzed", 0),
                "split_mode": data.get("split_mode", ""),
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


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[TraceIQ] Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
