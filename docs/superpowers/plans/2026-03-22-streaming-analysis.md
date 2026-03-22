# Streaming Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking POST /analyze with a streaming GET /analyze/stream endpoint that pushes real-time [TraceIQ] log lines to the browser via Server-Sent Events (SSE), so users see live progress instead of a fake spinner.

**Architecture:** New Flask SSE endpoint uses subprocess.Popen to read stderr line-by-line from the existing traceiq.py process, emitting step events as they arrive and a final result event when done. The UI replaces the static loading card with live event-driven updates using EventSource.

**Tech Stack:** Python Flask (SSE via streaming response), EventSource (browser native), existing traceiq.py subprocess

---

## File Map

- Modify: `server.py` — add `/analyze/stream` SSE endpoint
- Modify: `index.html` — replace fetch POST with EventSource in runAnalysis(), update loading card
- Modify: `agent/tools.py` — add granular batch progress prints to classify_traces
- Create: `tests/test_streaming.py` — unit tests for SSE event parsing logic

---

### Task 1: SSE endpoint in server.py

**Files:**
- Modify: `server.py` (add after existing `/analyze` route, around line 130)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming.py
import pytest
import json

def test_parse_traceiq_log_line_step():
    """[TraceIQ] lines become step events"""
    line = "[TraceIQ] Fetched 50 total runs"
    from server import _parse_sse_line
    event = _parse_sse_line(line)
    assert event == {"type": "step", "text": "Fetched 50 total runs"}

def test_parse_non_traceiq_line_returns_none():
    """Non-[TraceIQ] lines are ignored"""
    from server import _parse_sse_line
    assert _parse_sse_line("some random output") is None

def test_parse_empty_line_returns_none():
    from server import _parse_sse_line
    assert _parse_sse_line("") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Documents/traceiq && python -m pytest tests/test_streaming.py -v
```
Expected: FAIL with ImportError (_parse_sse_line not found)

- [ ] **Step 3: Add `_parse_sse_line` helper to server.py**

Add before the `/analyze` route:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Documents/traceiq && python -m pytest tests/test_streaming.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Add `/analyze/stream` endpoint**

Add after the existing `/analyze` route in server.py:
```python
@app.route("/analyze/stream", methods=["GET"])
def analyze_stream():
    """
    GET /analyze/stream?hypothesis=...&project=...&api_key=...&days=...&split_mode=...

    Streams real-time progress via Server-Sent Events. Each [TraceIQ] stderr
    line becomes a 'step' event. Final result is a 'result' event.
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
            proc.kill()
            yield _sse_event({"type": "error", "message": "Analysis timed out after 5 minutes"})
        except GeneratorExit:
            # Client disconnected
            try:
                proc.kill()
            except Exception:
                pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
```

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/traceiq && git add server.py tests/test_streaming.py
git commit -m "feat: add SSE streaming endpoint /analyze/stream"
```

---

### Task 2: Granular progress in classify_traces tool

**Files:**
- Modify: `agent/tools.py` (inside classify_traces function, around batch loop)

- [ ] **Step 1: Find the batch loop in classify_traces**

```bash
grep -n "batch\|Classif\|haiku" ~/Documents/traceiq/agent/tools.py | head -20
```

- [ ] **Step 2: Add batch progress prints**

Inside the batch loop in `classify_traces`, after each batch completes, add:
```python
print(f"[TraceIQ] Classifying traces: {min(i+batch_size, total)}/{total} labeled", file=sys.stderr, flush=True)
```

Where `i` is the batch index, `batch_size` is 20, `total` is `len(all_runs)`.

- [ ] **Step 3: Verify output manually**

```bash
cd ~/Documents/traceiq && python -c "
import sys
sys.path.insert(0, '.')
# just check the file looks right
import ast
with open('agent/tools.py') as f:
    ast.parse(f.read())
print('syntax OK')
"
```

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/traceiq && git add agent/tools.py
git commit -m "feat: emit batch progress from classify_traces to stderr"
```

---

### Task 3: UI — EventSource in runAnalysis()

**Files:**
- Modify: `index.html` (runAnalysis function, around line 820)

- [ ] **Step 1: Locate runAnalysis fetch call**

```bash
grep -n "fetch('/analyze'\|EventSource\|runAnalysis" ~/Documents/traceiq/index.html | head -10
```

- [ ] **Step 2: Replace fetch POST with EventSource**

Find the section in `runAnalysis()` that does:
```javascript
const resp = await fetch('/analyze', { method: 'POST', ... });
```

Replace the entire fetch block with:
```javascript
  // Build SSE URL
  const params = new URLSearchParams({
    hypothesis: state.hypothesis,
    api_key: state.apiKey,
    project: state.project,
    days: String(days),
    split_mode: state.splitMode,
  });
  const url = `/analyze/stream?${params}`;

  const es = new EventSource(url);
  let resolved = false;

  es.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === 'step') {
      // Update the active loading step with real text
      const activeStep = document.querySelector('.loading-steps li.active');
      if (activeStep) activeStep.textContent = msg.text;
    } else if (msg.type === 'result') {
      resolved = true;
      es.close();
      setStep('step-done');
      document.getElementById('loading-card').style.display = 'none';
      renderResults(msg.data);
      showScreen('screen-results');
    } else if (msg.type === 'error') {
      resolved = true;
      es.close();
      document.getElementById('loading-card').style.display = 'none';
      renderError('Analysis failed: ' + msg.message);
      showScreen('screen-results');
    }
  };

  es.onerror = () => {
    if (!resolved) {
      es.close();
      document.getElementById('loading-card').style.display = 'none';
      renderError('Connection lost. Please try again.');
      showScreen('screen-results');
    }
  };
  // Early return — result handled by EventSource callbacks
  return;
```

- [ ] **Step 3: Verify the function still has correct structure**

Open browser to http://localhost:5050, connect, and click Analyse with a hypothesis. Confirm the loading card appears and real log lines appear instead of fake steps.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/traceiq && git add index.html
git commit -m "feat: stream analysis progress via EventSource, show real [TraceIQ] log lines"
```

---

### Task 4: Integration test + server restart

- [ ] **Step 1: Restart server**

```bash
cd ~/Documents/traceiq && pkill -f "python3 server.py" 2>/dev/null; sleep 1; nohup python3 server.py > server.log 2>&1 &
sleep 2 && cat server.log
```

- [ ] **Step 2: Smoke test the SSE endpoint directly**

```bash
curl -N "http://localhost:5050/analyze/stream?hypothesis=test+hypothesis&project=autograding_production&api_key=REDACTED_LANGSMITH_KEY&days=7&split_mode=time_split" 2>/dev/null | head -20
```
Expected: lines starting with `data: {"type":"step",...}` then eventually `data: {"type":"result",...}`

- [ ] **Step 3: Run full test suite**

```bash
cd ~/Documents/traceiq && python -m pytest tests/ -v
```
Expected: all pass

- [ ] **Step 4: Final commit**

```bash
cd ~/Documents/traceiq && git push
```
