# Streaming Experiments Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Each task is independently executable. Read the design spec at `docs/superpowers/specs/2026-03-22-streaming-experiments-design.md` before starting. Run verification commands after each task to confirm correctness before moving on.

**Goal:** Replace the blocking `POST /experiments/analyze` endpoint with a streaming `GET /experiments/analyze/stream` SSE endpoint so users see live progress during the agent investigation.

**Architecture:** Extract the inline `-c` subprocess script to a standalone `agent/run_advisor.py` CLI file; add `[TraceIQ]`-prefixed progress prints to `prompt_advisor.py`; add a new Flask SSE endpoint that uses `subprocess.Popen` to stream stderr line-by-line (same pattern as `GET /analyze/stream`); update the UI to consume the stream via `EventSource`.

**Tech Stack:** Python 3 / Flask (`stream_with_context`, `Response`), `subprocess.Popen`, Server-Sent Events, JavaScript `EventSource` API.

---

### Task 1: Extract inline script + add `[TraceIQ]` progress prints

**Files:**
- `agent/run_advisor.py` — **create**
- `agent/prompt_advisor.py` — **modify** (update progress print prefixes)

- [ ] Create `agent/run_advisor.py` as a standalone CLI entrypoint:

  ```python
  #!/usr/bin/env python3
  """CLI entrypoint for prompt advisor — used by GET /experiments/analyze/stream."""
  import sys, os, json, argparse
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).parent.parent))

  from agent.prompt_advisor import run_prompt_advisor

  parser = argparse.ArgumentParser(description="Run TraceIQ prompt advisor")
  parser.add_argument("--api-key", required=True, help="LangSmith API key")
  parser.add_argument("--dataset-id", required=True, help="LangSmith dataset ID")
  parser.add_argument("--experiment-id", required=True, help="LangSmith experiment ID")
  parser.add_argument("--question", default="", help="User question/focus")
  args = parser.parse_args()

  result = run_prompt_advisor(
      api_key=args.api_key,
      dataset_id=args.dataset_id,
      experiment_id=args.experiment_id,
      question=args.question,
  )
  print(json.dumps(result, default=str))
  ```

- [ ] In `agent/prompt_advisor.py`, update the three existing progress prints to use `[TraceIQ]` prefix (not `[TraceIQ/advisor]`) so they match `_parse_sse_line()` in `server.py`:

  ```python
  # Line ~132 — change from:
  print(f"[TraceIQ/advisor] Analyzing experiment '{experiment_name}' on dataset '{dataset_name}'...", file=sys.stderr)
  # To:
  print(f"[TraceIQ] Analyzing experiment '{experiment_name}' on dataset '{dataset_name}'...", file=sys.stderr, flush=True)

  # Line ~170 — change from:
  print(f"[TraceIQ/advisor] Starting agent investigation...", file=sys.stderr)
  # To:
  print(f"[TraceIQ] Starting agent investigation — this may take a few minutes...", file=sys.stderr, flush=True)

  # Line ~208 — change from:
  print(f"[TraceIQ/advisor] Agent finished. Extracting recommendations...", file=sys.stderr)
  # To:
  print(f"[TraceIQ] Agent finished. Extracting recommendations...", file=sys.stderr, flush=True)
  ```

- [ ] Verify the script runs end-to-end (dry-run with `--help`):

  ```bash
  python agent/run_advisor.py --help
  # Expected: prints usage with --api-key, --dataset-id, --experiment-id, --question
  ```

- [ ] Verify `[TraceIQ]` lines appear in stderr during a real run:

  ```bash
  python agent/run_advisor.py \
    --api-key $LANGSMITH_API_KEY \
    --dataset-id <test_dataset_id> \
    --experiment-id <test_experiment_id> \
    --question "test" 2>&1 | grep "\[TraceIQ\]"
  # Expected: 3+ lines starting with [TraceIQ]
  ```

---

### Task 2: Add `GET /experiments/analyze/stream` SSE endpoint

**Files:**
- `server.py` — **modify** (add new route after the existing `experiments_analyze` function)

- [ ] Add the new streaming route to `server.py` after the `experiments_analyze()` function (~line 480). Insert:

  ```python
  @app.route("/experiments/analyze/stream", methods=["GET"])
  def experiments_analyze_stream():
      """
      GET /experiments/analyze/stream?api_key=...&dataset_id=...&experiment_id=...&question=...
      Streams real-time [TraceIQ] log lines as SSE step events.
      Final result is emitted as a 'result' event.
      """
      api_key = request.args.get("api_key", "").strip()
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
          "--question", question,
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

              # Stream stderr line by line
              for line in proc.stderr:
                  event = _parse_sse_line(line)
                  if event:
                      yield _sse_event(event)

              proc.wait(timeout=600)

              if proc.returncode != 0:
                  stderr_tail = ""
                  if proc.stderr:
                      try:
                          stderr_tail = proc.stderr.read()[-500:]
                      except Exception:
                          pass
                  yield _sse_event({"type": "error", "message": f"Analysis failed (exit {proc.returncode}): {stderr_tail}"})
                  return

              # Read final JSON result from stdout
              try:
                  raw = proc.stdout.read()
                  output = json.loads(raw)
              except Exception as e:
                  yield _sse_event({"type": "error", "message": f"Could not parse output: {e}"})
                  return

              # Save to history
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
          except Exception as e:
              yield _sse_event({"type": "error", "message": f"Unexpected error: {e}"})

      return Response(stream_with_context(generate()), mimetype="text/event-stream")
  ```

- [ ] Confirm `stream_with_context` is imported (it already is if the traces endpoint works — check with):

  ```bash
  grep "stream_with_context" server.py
  # Expected: at least one import line
  ```

- [ ] Test the endpoint with curl (requires a real LangSmith key and IDs):

  ```bash
  curl -N "http://localhost:5001/experiments/analyze/stream?api_key=$LANGSMITH_API_KEY&dataset_id=<id>&experiment_id=<id>&question=test"
  # Expected: SSE stream with data: {"type":"step",...} lines followed by data: {"type":"result",...}
  ```

---

### Task 3: Update UI to use EventSource for experiments flow

**Files:**
- `static/` or `templates/` — locate the experiments JS/HTML (check `grep -r "experiments/analyze" static/ templates/`)
- Modify whichever file calls `POST /experiments/analyze`

- [ ] Find the current experiments fetch call:

  ```bash
  grep -rn "experiments/analyze\|experimentAnalyze\|runExperiment" ~/Documents/traceiq/static/ ~/Documents/traceiq/templates/ 2>/dev/null
  ```

- [ ] Replace the blocking `fetch` / `POST` call with an `EventSource` stream. Pattern:

  ```javascript
  function runExperimentsAnalysis(apiKey, datasetId, experimentId, question) {
    // Show progress container, clear previous results
    showProgressContainer();
    clearProgressLog();

    const params = new URLSearchParams({
      api_key: apiKey,
      dataset_id: datasetId,
      experiment_id: experimentId,
      question: question,
    });

    const source = new EventSource(`/experiments/analyze/stream?${params}`);

    source.onmessage = (e) => {
      const event = JSON.parse(e.data);

      if (event.type === "step") {
        appendProgressLine(event.text);         // live progress log
      } else if (event.type === "result") {
        source.close();
        hideProgressContainer();
        renderExperimentResult(event.data);     // existing render function
      } else if (event.type === "error") {
        source.close();
        hideProgressContainer();
        showError(event.message);
      }
    };

    source.onerror = () => {
      source.close();
      showError("Stream connection lost. Please try again.");
    };
  }
  ```

- [ ] Add or reuse a progress log UI element (share the component used in the traces tab if possible):

  ```html
  <!-- In the experiments section of index.html -->
  <div id="exp-progress-container" style="display:none">
    <div class="progress-header">🔍 Analyzing...</div>
    <div id="exp-progress-log" class="progress-log"></div>
  </div>
  ```

- [ ] Verify in browser: submit an experiments analysis → progress lines appear live → result renders on completion.

---

### Task 4: Integration test + push

**Files:** none modified — verification only, then git push.

- [ ] Run a full end-to-end test:

  ```bash
  # 1. Start the server
  cd ~/Documents/traceiq && python server.py &

  # 2. Curl the stream endpoint
  curl -N "http://localhost:5001/experiments/analyze/stream?api_key=$LANGSMITH_API_KEY&dataset_id=<real_id>&experiment_id=<real_id>&question=Why+are+scores+low"
  # Expected: 3+ step events, then a result event with non-empty recommendations

  # 3. Kill server
  kill %1
  ```

- [ ] Confirm the old `POST /experiments/analyze` still works (backwards compat):

  ```bash
  curl -s -X POST http://localhost:5001/experiments/analyze \
    -H "Content-Type: application/json" \
    -d '{"api_key":"'$LANGSMITH_API_KEY'","dataset_id":"<id>","experiment_id":"<id>","question":"test"}' \
    | python -m json.tool | head -20
  # Expected: valid JSON result (blocking, same as before)
  ```

- [ ] Check no Python syntax errors in modified files:

  ```bash
  python -m py_compile server.py && echo "server.py OK"
  python -m py_compile agent/run_advisor.py && echo "run_advisor.py OK"
  python -m py_compile agent/prompt_advisor.py && echo "prompt_advisor.py OK"
  ```

- [ ] Commit and push:

  ```bash
  cd ~/Documents/traceiq
  git add agent/run_advisor.py agent/prompt_advisor.py server.py static/ templates/
  git commit -m "feat: stream experiments analysis via SSE (GET /experiments/analyze/stream)"
  git push
  ```
