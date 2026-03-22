# Streaming Analysis — Design Spec
Date: 2026-03-22

## Problem
POST /analyze blocks for 30-90s. Browser shows a fake animated loading state disconnected from reality. Users can't tell if the agent is working or hung. This causes abandonment.

## Solution: Server-Sent Events (SSE)

Replace the blocking POST with a streaming GET endpoint. The subprocess already emits `[TraceIQ]` log lines to stderr — we pipe those live to the browser via SSE.

## Architecture

### New endpoint
`GET /analyze/stream?hypothesis=...&project=...&api_key=...&days=...&split_mode=...`

- Uses `subprocess.Popen()` (not `subprocess.run()`) to capture stderr line-by-line
- Returns `Content-Type: text/event-stream` with `Cache-Control: no-cache`
- Each `[TraceIQ]` stderr line → emits `data: {"type":"step","text":"..."}` SSE event
- On process exit: reads `hypothesis_output.json` → emits `data: {"type":"result","data":{...}}` SSE event
- On error: emits `data: {"type":"error","message":"..."}` SSE event
- Existing `POST /analyze` kept intact as fallback

### Event types
```json
{"type": "step", "text": "Fetched 50 total runs"}
{"type": "step", "text": "Classifying traces in batches of 20..."}
{"type": "result", "data": { ...full verdict JSON... }}
{"type": "error", "message": "Analysis timed out"}
```

### UI changes
- Replace `fetch('/analyze', {method:'POST'})` with `new EventSource('/analyze/stream?...')`
- `step` events: update loading card with real log text (replace fake animated steps)
- `result` event: close EventSource, call existing `renderResults(data)`
- `error` event: close EventSource, call existing `renderError(msg)`
- Fallback: if `EventSource` fails to connect, retry once with existing POST

### Agent/traceiq.py
- No structural changes needed — `[TraceIQ]` log lines already on stderr
- Add 2-3 granular progress prints inside `classify_traces` tool (batch progress)

## Error handling
- Subprocess timeout (5 min): emit error event, kill process
- Process exits non-zero: emit error event with stderr tail
- hypothesis_output.json missing on clean exit: emit error event
- Client disconnects mid-stream: server detects broken pipe, kills subprocess

## What does NOT change
- Result rendering logic
- History saving
- Experiment analysis flow (separate endpoint, out of scope)
- Deep analysis mode vs standard mode — same streaming works for both

## Success criteria
- Browser shows real log lines as they appear (< 1s delay)
- No fake steps — loading card reflects actual agent state
- Result renders the moment the agent finishes
- No regression on existing POST /analyze fallback
