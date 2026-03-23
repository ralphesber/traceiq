# Session Token for API Key — Design Spec
Date: 2026-03-23

## Problem
The LangSmith API key is currently passed as a query parameter in all GET/SSE URLs:
- `GET /analyze/stream?api_key=lsv2_pt_...`
- `GET /experiments/analyze/stream?api_key=lsv2_pt_...`
- `GET /experiments/datasets?api_key=lsv2_pt_...`
- `GET /overview?api_key=lsv2_pt_...`

Query params appear in server logs, browser history, and network inspector. This is a security risk, especially when hosting for a team.

EventSource only supports GET, so we cannot simply move to a POST body for SSE endpoints.

## Solution: Short-lived session tokens

Two-step flow:
1. UI POSTs `{api_key}` to `/api/session` → server stores it in memory, returns `{session_id: "<uuid>"}`
2. UI uses `session_id` in all subsequent GET/SSE requests instead of `api_key`
3. Server looks up the real key from the session store before executing

The `api_key` never appears in any URL. The `session_id` is a random UUID with no semantic meaning.

## Session store

In-memory dict on the server (sufficient for v1, resets on restart which is acceptable):

```python
import uuid, time, threading

_sessions: dict[str, dict] = {}  # {session_id: {api_key, created_at}}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600  # 1 hour

def _create_session(api_key: str) -> str:
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = {"api_key": api_key, "created_at": time.time()}
    return session_id

def _get_session_key(session_id: str) -> str | None:
    with _sessions_lock:
        s = _sessions.get(session_id)
        if not s:
            return None
        if time.time() - s["created_at"] > SESSION_TTL:
            del _sessions[session_id]
            return None
        return s["api_key"]
```

## New endpoint

`POST /api/session`
- Body: `{"api_key": "lsv2_pt_..."}`
- Returns: `{"session_id": "<uuid>"}`
- Validates the key is non-empty

## Endpoints to update (replace api_key param with session_id)

All GET endpoints that currently take `api_key` as a query param:
1. `GET /analyze/stream` 
2. `GET /experiments/analyze/stream`
3. `GET /experiments/datasets`
4. `GET /experiments/list`
5. `GET /overview`

Each resolves the real key via `_get_session_key(session_id)` before use.

## POST endpoints (already safe)

`POST /analyze` and `POST /experiments/analyze` already receive `api_key` in the JSON body — these are fine as-is and don't need changes.

## UI changes

1. On `connect()`: POST to `/api/session` with the API key, store returned `session_id` in `state.sessionId`
2. Replace all `api_key` query params in fetch/EventSource calls with `session_id`
3. The key itself stays in `localStorage` for auto-fill but is never sent in a URL again

## Backward compatibility

Keep `api_key` query param support as fallback for local dev (check if `api_key` present, use it; else require `session_id`). This lets existing curl commands still work.

## What does NOT change
- `POST /analyze` and `POST /experiments/analyze` (already safe)
- localStorage auto-fill (key still stored locally for UX)
- All result rendering
- History
