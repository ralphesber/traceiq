#!/usr/bin/env python3
"""
TraceIQ server — minimal Flask server for the hypothesis testing UI.

Serves:
  GET  /          → index.html
  GET  /static/*  → static files
  POST /analyze   → runs traceiq.py with hypothesis args, returns JSON result
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).parent.resolve()
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(BASE_DIR), filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    hypothesis = data.get("hypothesis", "").strip()
    api_key = data.get("api_key", "").strip()
    project = data.get("project", "").strip()
    days = int(data.get("days", 30))

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
    ]

    env = os.environ.copy()
    # Pass through LLM keys from environment
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if key in env:
            pass  # already in env
        elif data.get(key):
            env[key] = data[key]

    print(f"[server] Running: {' '.join(cmd[:4])} ...", flush=True)

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
            return jsonify(output)
        except Exception as e:
            return jsonify({"error": f"Could not read output: {e}"}), 500

    # Fall back to stdout
    try:
        output = json.loads(result.stdout)
        return jsonify(output)
    except json.JSONDecodeError:
        return jsonify({"error": "Analysis produced invalid output", "raw": result.stdout[-1000:]}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[TraceIQ] Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
