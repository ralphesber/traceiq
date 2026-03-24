# Session Token Implementation Plan
Date: 2026-03-23

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the LangSmith API key out of all GET/SSE URLs by introducing a short-lived session token system.

**Architecture:** New `POST /api/session` endpoint stores the key in a server-side in-memory dict and returns a UUID session_id. All GET endpoints resolve the real key via the session_id. The key never appears in any URL.

**Tech Stack:** Python uuid/threading (stdlib only), Flask, vanilla JS fetch

---

## File Map
- Modify: `server.py` — add session store, `/api/session` endpoint, update 5 GET endpoints
- Modify: `index.html` — add session creation on connect(), replace api_key query params with session_id
- Modify: `tests/test_streaming.py` — add test for `_get_session_key`

---

### Task 1: Session store + /api/session endpoint in server.py

**Files:**
- Modify: `server.py`

- [ ] **Write failing tests first**

Add to `tests/test_streaming.py`:
```python
def test_session_create_and_retrieve():
    from server import _create_session, _get_session_key
    sid = _create_session("test-key-123")
    assert len(sid) == 36  # UUID format
    assert _get_session_key(sid) == "test-key-123"

def test_session_missing_returns_none():
    from server import _get_session_key
    assert _get_session_key("nonexistent-session-id") is None

def test_session_empty_key_rejected():
    from server import _create_session
    try:
        _create_session("")
        assert False, "Should have raised"
    except ValueError:
        pass
```

- [ ] **Run tests — confirm FAIL**
```bash
cd ~/Documents/traceiq && python3 -m pytest tests/test_streaming.py::test_session_create_and_retrieve -v
```
Expected: ImportError

- [ ] **Add session store to server.py** (after imports, before app = Flask(...)):
```python
import uuid
import threading

_sessions: dict = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600  # 1 hour

def _create_session(api_key: str) -> str:
    if not api_key:
        raise ValueError("api_key cannot be empty")
    import time
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = {"api_key": api_key, "created_at": time.time()}
    return session_id

def _get_session_key(session_id: str):
    import time
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
```

- [ ] **Add POST /api/session endpoint** (after app = Flask(...), before other routes):
```python
@app.route("/api/session", methods=["POST"])
def create_session():
    data = request.get_json(force=True) or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    session_id = _create_session(api_key)
    return jsonify({"session_id": session_id})
```

- [ ] **Run tests — confirm PASS**
```bash
cd ~/Documents/traceiq && python3 -m pytest tests/test_streaming.py -v
```
Expected: all pass including 3 new session tests

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add server.py tests/test_streaming.py
git commit -m "feat: add session store and POST /api/session endpoint"
```

---

### Task 2: Update 5 GET endpoints to use session_id

**Files:**
- Modify: `server.py`

Update these endpoints to resolve api_key via `_resolve_api_key()` instead of `request.args.get("api_key")`:
1. `GET /analyze/stream` (line ~148)
2. `GET /experiments/analyze/stream` (line ~479)
3. `GET /experiments/datasets` (line ~310)
4. `GET /experiments/list` (line ~360)
5. `GET /overview` (line ~560)

For each endpoint, replace:
```python
api_key = request.args.get("api_key", "").strip()
```
With:
```python
api_key = (_resolve_api_key(request.args) or "").strip()
```

And update the error message for missing/invalid key where appropriate.

- [ ] **Make the changes** to all 5 endpoints

- [ ] **Verify syntax**
```bash
cd ~/Documents/traceiq && python3 -c "import server; print('OK')"
```

- [ ] **Smoke test — session flow works end to end**
```bash
cd ~/Documents/traceiq && pkill -f "python3 server.py" 2>/dev/null; sleep 1
nohup python3 server.py > server.log 2>&1 & sleep 2

# Create session
SID=$(curl -s -X POST http://localhost:5050/api/session \
  -H "Content-Type: application/json" \
  -d '{"api_key":"lsv2_pt_YOUR_LANGSMITH_KEY_HERE"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SID"

# Use session_id (not api_key) in GET request
curl -s "http://localhost:5050/overview?project=autograding_production&session_id=$SID" | python3 -m json.tool | head -10
```
Expected: valid overview JSON

- [ ] **Verify fallback still works (api_key direct)**
```bash
curl -s "http://localhost:5050/overview?project=autograding_production&api_key=lsv2_pt_YOUR_LANGSMITH_KEY_HERE" | python3 -m json.tool | head -5
```
Expected: same valid overview JSON

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add server.py
git commit -m "feat: update all GET endpoints to resolve api_key via session_id"
```

---

### Task 3: UI — create session on connect, use session_id everywhere

**Files:**
- Modify: `index.html`

- [ ] **Add session creation to connect() function**

In the `connect()` function (around line 1200), after validating apiKey and project, add a session creation call before `showScreen`:

```javascript
// Create server-side session so api_key never appears in URLs
try {
  const sessResp = await fetch('/api/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({api_key: apiKey}),
  });
  const sessData = await sessResp.json();
  state.sessionId = sessData.session_id || '';
} catch (e) {
  state.sessionId = '';
}
```

Also make `connect()` async if it isn't already.

- [ ] **Add sessionId to state**

Find the state initialization:
```javascript
let state = { apiKey: '', project: '', ... };
```
Add `sessionId: ''` to it.

- [ ] **Replace api_key with session_id in all GET/SSE fetch calls**

Find and update these 5 places:
1. `loadOverview()` — `GET /overview?...api_key=...` → use `session_id`
2. `loadExpDatasets()` — `GET /experiments/datasets?api_key=...` → use `session_id`
3. `loadExpExperiments()` — `GET /experiments/list?...api_key=...` → use `session_id`
4. `runAnalysis()` EventSource params — remove `api_key`, add `session_id`
5. `runExpAnalysis()` EventSource params — remove `api_key`, add `session_id`

For each, change:
```javascript
api_key: state.apiKey,
```
To:
```javascript
session_id: state.sessionId || state.apiKey,  // fallback for dev
```

- [ ] **Verify no api_key appears in URLs**
```bash
grep -n "api_key.*state\.\|apiKey.*encodeURI\|api_key.*encodeURI" ~/Documents/traceiq/index.html | grep -v "localStorage\|getElementById\|password\|placeholder\|body.*JSON\|POST"
```
Expected: no matches (all URL uses replaced)

- [ ] **Test in browser**
Open http://localhost:5050, connect, run an analysis. In browser DevTools → Network tab, confirm no request URLs contain the api_key string.

- [ ] **Commit**
```bash
cd ~/Documents/traceiq && git add index.html
git commit -m "feat: create session on connect, use session_id in all GET/SSE URLs — api_key never in URLs"
```

---

### Task 4: Integration test + push

- [ ] **Run full test suite**
```bash
cd ~/Documents/traceiq && python3 -m pytest tests/ -v
```
Expected: all pass

- [ ] **End-to-end curl test**
```bash
# Session flow
SID=$(curl -s -X POST http://localhost:5050/api/session \
  -H "Content-Type: application/json" \
  -d '{"api_key":"lsv2_pt_YOUR_LANGSMITH_KEY_HERE"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# SSE stream with session_id
curl -N --max-time 10 "http://localhost:5050/analyze/stream?hypothesis=test&project=autograding_production&session_id=$SID&days=7&split_mode=time_split" 2>/dev/null | head -5
```
Expected: SSE step events, no api_key in URL

- [ ] **Confirm api_key NOT in server.log**
```bash
grep "lsv2_pt" ~/Documents/traceiq/server.log | head -5
```
Expected: no matches

- [ ] **Push**
```bash
cd ~/Documents/traceiq && git push
```
