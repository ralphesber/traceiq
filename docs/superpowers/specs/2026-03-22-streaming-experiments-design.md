# Streaming Experiments Analysis — Design Spec

**Date:** 2026-03-22  
**Status:** Draft  
**Author:** Jarvis (AI assistant)

---

## Problem

The `POST /experiments/analyze` endpoint is fully blocking. It builds an inline Python script string, passes it to `subprocess.run(... "-c", script)`, and waits up to 600 seconds for the process to exit before returning any data to the client.

During that time:
- The browser shows a spinner with no feedback.
- The user has no idea whether the agent is making progress, stuck, or about to time out.
- The connection is held open the entire time — any network blip kills it.

The `prompt_advisor.py` already emits progress to `sys.stderr` at three key stages:
```python
print(f"[TraceIQ/advisor] Analyzing experiment '...' on dataset '...'...", file=sys.stderr)
print(f"[TraceIQ/advisor] Starting agent investigation...", file=sys.stderr)
print(f"[TraceIQ/advisor] Agent finished. Extracting recommendations...", file=sys.stderr)
```

These lines are silently swallowed by `subprocess.run(capture_output=True)` and never reach the client.

---

## Solution

Apply the same SSE streaming pattern already used by `GET /analyze/stream` (traces flow) to a new `GET /experiments/analyze/stream` endpoint.

The traces flow works by:
1. Running a **script file** (`traceiq.py`) via `subprocess.Popen`
2. Reading `proc.stderr` line-by-line as it arrives
3. Filtering lines that start with `[TraceIQ]` via `_parse_sse_line()`
4. Formatting each as an `EventSource`-compatible SSE chunk via `_sse_event()`
5. Emitting the final JSON result as a `{"type": "result", "data": ...}` SSE event

The experiments flow can adopt this pattern with one structural change: the inline `-c` script must become a **real file on disk**.

---

## Key Difference: Inline Script → Dedicated Script File

`subprocess.Popen` can't stream stderr from an inline `-c` script any better than `subprocess.run` can — the difference is `Popen` vs `run`, but the real issue is that inline scripts make it hard to reason about path setup, imports, and the final output mechanism.

### Why extract to a file?

- Clean separation: `agent/run_advisor.py` owns the CLI entrypoint; `prompt_advisor.py` owns the logic.
- `Popen` can read `proc.stderr` as a line-by-line generator (same as traces).
- The final JSON result can be printed to `stdout` (one line), read after `proc.wait()`.
- No `-c` string escaping headaches.

### New file: `agent/run_advisor.py`

This script is the CLI wrapper. It:
1. Accepts args via `argparse` or env vars
2. Calls `run_prompt_advisor(...)`
3. Prints JSON result to stdout
4. Lets `prompt_advisor.py` emit `[TraceIQ]` lines to stderr naturally

```python
#!/usr/bin/env python3
"""CLI entrypoint for prompt advisor — used by the SSE streaming endpoint."""
import sys, os, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompt_advisor import run_prompt_advisor

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
parser.add_argument("--dataset-id", required=True)
parser.add_argument("--experiment-id", required=True)
parser.add_argument("--question", default="")
args = parser.parse_args()

result = run_prompt_advisor(
    api_key=args.api_key,
    dataset_id=args.dataset_id,
    experiment_id=args.experiment_id,
    question=args.question,
)
print(json.dumps(result, default=str))
```

---

## Progress Prints in `prompt_advisor.py`

The three existing `[TraceIQ/advisor]` prints are good but sparse. To give the user meaningful feedback during the agent's tool-calling loop, additional prints should be added:

| Stage | Message |
|---|---|
| Start | `[TraceIQ] Fetching experiment metadata...` |
| Agent start | `[TraceIQ] Agent investigating — this may take a few minutes...` |
| Tool call detected (optional) | `[TraceIQ] Agent called tool: {tool_name}` |
| Agent done | `[TraceIQ] Agent complete. Extracting recommendations...` |
| Parsing done | `[TraceIQ] Parsing structured output...` |

**Important:** The prefix must be `[TraceIQ]` (not `[TraceIQ/advisor]`) to match the existing `_parse_sse_line()` filter in `server.py`:

```python
if line.startswith("[TraceIQ]"):
    text = line[len("[TraceIQ]"):].strip()
    return {"type": "step", "text": text}
```

The existing `[TraceIQ/advisor]` prints do **not** match this filter. They should either be changed to `[TraceIQ]` or supplemented with `[TraceIQ]`-prefixed prints at each stage.

---

## New Endpoint: `GET /experiments/analyze/stream`

```
GET /experiments/analyze/stream
  ?api_key=<langsmith_key>
  &dataset_id=<id>
  &experiment_id=<id>
  &question=<text>
```

**Response:** `text/event-stream` (SSE)

### Event types

| Event | Shape | Meaning |
|---|---|---|
| `step` | `{"type": "step", "text": "..."}` | A progress line from stderr |
| `result` | `{"type": "result", "data": {...}}` | Final JSON output |
| `error` | `{"type": "error", "message": "..."}` | Subprocess failure or timeout |

### Server logic

```python
@app.route("/experiments/analyze/stream", methods=["GET"])
def experiments_analyze_stream():
    api_key = request.args.get("api_key", "").strip()
    dataset_id = request.args.get("dataset_id", "").strip()
    experiment_id = request.args.get("experiment_id", "").strip()
    question = request.args.get("question", "").strip()

    # validation ...

    cmd = [
        sys.executable, str(BASE_DIR / "agent" / "run_advisor.py"),
        "--api-key", api_key,
        "--dataset-id", dataset_id,
        "--experiment-id", experiment_id,
        "--question", question,
    ]

    def generate():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(BASE_DIR),
        )
        for line in proc.stderr:
            event = _parse_sse_line(line)
            if event:
                yield _sse_event(event)
        proc.wait(timeout=600)
        if proc.returncode != 0:
            yield _sse_event({"type": "error", "message": f"Analysis failed (exit {proc.returncode})"})
            return
        try:
            output = json.loads(proc.stdout.read())
            _save_experiment_to_history(output)
            yield _sse_event({"type": "result", "data": output})
        except Exception as e:
            yield _sse_event({"type": "error", "message": str(e)})

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
```

---

## UI Changes

The experiments tab currently calls `POST /experiments/analyze` and awaits the JSON response. It should be updated to use `EventSource`, mirroring the traces tab's streaming flow.

### Pattern

```javascript
// Replace fetch(...POST...) with:
const params = new URLSearchParams({ api_key, dataset_id, experiment_id, question });
const source = new EventSource(`/experiments/analyze/stream?${params}`);

source.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (event.type === "step") {
    appendProgressLine(event.text);       // show in the progress log area
  } else if (event.type === "result") {
    source.close();
    renderExperimentResult(event.data);   // same render as before
  } else if (event.type === "error") {
    source.close();
    showError(event.message);
  }
};

source.onerror = () => {
  source.close();
  showError("Connection lost.");
};
```

The UI should show:
- A live-updating progress log (same component as traces, or a shared `<ProgressLog>` component)
- A "thinking..." indicator while steps arrive
- The final result rendered exactly as before once the `result` event fires

---

## Non-Goals

- No change to the existing `POST /experiments/analyze` endpoint (keep for backwards compat / non-streaming clients).
- No change to the `run_prompt_advisor()` function signature.
- No change to the result schema.
