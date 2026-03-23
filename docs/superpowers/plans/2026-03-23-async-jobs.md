# Async Job Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Replace synchronous SSE-streamed analysis with a Postgres-backed async job queue so that long-running AI analysis (>60s) no longer times out at Railway's proxy layer.

**Architecture:** The Flask server gets two new "start" endpoints that immediately enqueue a job in Postgres and return a `job_id`. A standalone `worker.py` process polls Postgres for queued jobs, runs the analysis (calling `run_hypothesis_agent` or `run_prompt_advisor`), appends progress steps to the DB as they arrive, then writes the final result. The UI switches from `EventSource` SSE to polling `GET /jobs/{job_id}` every 2 seconds. The old SSE endpoints remain intact for local development.

**Tech Stack:** Flask, psycopg2-binary (already installed), Postgres (Railway-provided), Python threading for progress capture, Railway Worker service (Procfile).

---

### Task 1: DB Schema — jobs table + psycopg2 helpers

**Files:** `server.py`

- [ ] **Step 1 — Add jobs table creation inside `_init_db()`**

  Locate `_init_db()` in `server.py`. After the existing `CREATE TABLE IF NOT EXISTS history (...)` statement, add:

  ```python
  cur.execute("""
      CREATE TABLE IF NOT EXISTS jobs (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          status TEXT NOT NULL DEFAULT 'queued',
          job_type TEXT NOT NULL,
          input JSONB NOT NULL,
          steps TEXT[],
          result JSONB,
          error TEXT,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
      );
      CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
      CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at DESC);
  """)
  print("[server] DB: jobs table ready", flush=True)
  ```

- [ ] **Step 2 — Add the six job helper functions**

  Add these functions immediately after `_init_db()` in `server.py`:

  ```python
  # ── Job queue helpers ─────────────────────────────────────────────────────

  def _create_job(job_type: str, input_data: dict) -> str | None:
      """Insert a new queued job; returns job_id (UUID string) or None on failure."""
      import psycopg2.extras
      conn = _get_db_conn()
      if not conn:
          return None
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      """
                      INSERT INTO jobs (job_type, input, steps)
                      VALUES (%s, %s, %s)
                      RETURNING id::text
                      """,
                      (job_type, psycopg2.extras.Json(input_data), [])
                  )
                  row = cur.fetchone()
                  return row[0] if row else None
      except Exception as e:
          print(f"[server] _create_job failed: {e}", flush=True)
          return None
      finally:
          conn.close()


  def _update_job_status(job_id: str, status: str) -> None:
      """Update job status (queued → running → done/failed)."""
      conn = _get_db_conn()
      if not conn:
          return
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      "UPDATE jobs SET status = %s, updated_at = NOW() WHERE id = %s",
                      (status, job_id)
                  )
      except Exception as e:
          print(f"[server] _update_job_status failed: {e}", flush=True)
      finally:
          conn.close()


  def _append_job_step(job_id: str, step_text: str) -> None:
      """Append a progress step string to the job's steps array."""
      conn = _get_db_conn()
      if not conn:
          return
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      """
                      UPDATE jobs
                      SET steps = array_append(COALESCE(steps, '{}'), %s),
                          updated_at = NOW()
                      WHERE id = %s
                      """,
                      (step_text, job_id)
                  )
      except Exception as e:
          print(f"[server] _append_job_step failed: {e}", flush=True)
      finally:
          conn.close()


  def _complete_job(job_id: str, result: dict) -> None:
      """Mark job as done and write the final result JSON."""
      import psycopg2.extras
      conn = _get_db_conn()
      if not conn:
          return
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      """
                      UPDATE jobs
                      SET status = 'done', result = %s, updated_at = NOW()
                      WHERE id = %s
                      """,
                      (psycopg2.extras.Json(result), job_id)
                  )
      except Exception as e:
          print(f"[server] _complete_job failed: {e}", flush=True)
      finally:
          conn.close()


  def _fail_job(job_id: str, error: str) -> None:
      """Mark job as failed and store the error message."""
      conn = _get_db_conn()
      if not conn:
          return
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      """
                      UPDATE jobs
                      SET status = 'failed', error = %s, updated_at = NOW()
                      WHERE id = %s
                      """,
                      (error, job_id)
                  )
      except Exception as e:
          print(f"[server] _fail_job failed: {e}", flush=True)
      finally:
          conn.close()


  def _get_job(job_id: str) -> dict | None:
      """Return job row as dict {id, status, job_type, steps, result, error, created_at, updated_at} or None."""
      conn = _get_db_conn()
      if not conn:
          return None
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute(
                      """
                      SELECT id::text, status, job_type, steps, result, error,
                             created_at, updated_at
                      FROM jobs WHERE id = %s
                      """,
                      (job_id,)
                  )
                  row = cur.fetchone()
                  if not row:
                      return None
                  return {
                      "id": row[0],
                      "status": row[1],
                      "job_type": row[2],
                      "steps": row[3] or [],
                      "result": row[4],
                      "error": row[5],
                      "created_at": row[6].isoformat() if row[6] else None,
                      "updated_at": row[7].isoformat() if row[7] else None,
                  }
      except Exception as e:
          print(f"[server] _get_job failed: {e}", flush=True)
          return None
      finally:
          conn.close()
  ```

- [ ] **Step 3 — Verify**

  ```bash
  cd ~/Documents/traceiq
  python3 -c "from server import _create_job, _get_job, _append_job_step, _complete_job, _fail_job, _update_job_status; print('imports OK')"
  # If DATABASE_URL is set locally:
  python3 -c "
  from server import _create_job, _get_job, _append_job_step, _complete_job
  jid = _create_job('hypothesis', {'hypothesis': 'test', 'project': 'test-project'})
  print('job_id:', jid)
  _append_job_step(jid, 'Fetching traces...')
  _complete_job(jid, {'verdict': 'supported'})
  print(_get_job(jid))
  "
  ```

- [ ] **Commit:** `git commit -m "feat: add jobs table schema and psycopg2 helpers"`

---

### Task 2: worker.py — background job processor

**Files:** `worker.py` (new file in project root)

- [ ] **Step 1 — Create `worker.py`**

  ```python
  #!/usr/bin/env python3
  """
  TraceIQ Worker — polls Postgres for queued jobs and runs them.

  Loop:
    1. SELECT oldest queued job (FOR UPDATE SKIP LOCKED — safe for multiple workers)
    2. Mark it 'running'
    3. Run the analysis, appending steps to DB as they arrive
    4. Mark done (with result) or failed (with error message)
    5. Sleep 2s, repeat

  Run locally:
    python3 worker.py

  On Railway:
    Add as a Worker service (see Task 6).
  """

  import json
  import os
  import sys
  import time
  from datetime import datetime, timezone
  from pathlib import Path

  BASE_DIR = Path(__file__).parent.resolve()
  sys.path.insert(0, str(BASE_DIR))

  # Import helpers from server (they share the same DB connection logic)
  from server import (
      _get_db_conn,
      _update_job_status,
      _append_job_step,
      _complete_job,
      _fail_job,
  )


  POLL_INTERVAL = 2  # seconds between polls when queue is empty


  def _claim_next_job() -> dict | None:
      """Atomically claim the oldest queued job. Returns job row dict or None."""
      conn = _get_db_conn()
      if not conn:
          return None
      try:
          with conn:
              with conn.cursor() as cur:
                  cur.execute("""
                      SELECT id::text, job_type, input
                      FROM jobs
                      WHERE status = 'queued'
                      ORDER BY created_at ASC
                      LIMIT 1
                      FOR UPDATE SKIP LOCKED
                  """)
                  row = cur.fetchone()
                  if not row:
                      return None
                  job_id, job_type, input_data = row[0], row[1], row[2]
                  # Mark as running immediately inside the same transaction
                  cur.execute(
                      "UPDATE jobs SET status = 'running', updated_at = NOW() WHERE id = %s",
                      (job_id,)
                  )
                  return {"id": job_id, "job_type": job_type, "input": input_data}
      except Exception as e:
          print(f"[worker] _claim_next_job failed: {e}", flush=True)
          return None
      finally:
          conn.close()


  def _step_callback(job_id: str):
      """Return a callback that appends [TraceIQ] stderr lines to the job's steps."""
      def callback(line: str):
          line = line.strip()
          if not line:
              return
          if line.startswith("[TraceIQ]"):
              text = line[len("[TraceIQ]"):].strip()
              _append_job_step(job_id, text)
              print(f"[worker] step: {text}", flush=True)
      return callback


  def _run_hypothesis_job(job: dict) -> dict:
      """Run a hypothesis analysis job. Streams steps to DB. Returns result dict."""
      job_id = job["id"]
      inp = job["input"]

      api_key = inp.get("api_key", "")
      project = inp.get("project", "")
      hypothesis = inp.get("hypothesis", "")
      days = int(inp.get("days", 30))

      # Set env
      os.environ["LANGSMITH_API_KEY"] = api_key

      # We need to capture stderr from the agent to stream steps.
      # run_hypothesis_agent prints [TraceIQ] lines to stderr (sys.stderr).
      # We monkey-patch sys.stderr temporarily to intercept and forward steps.
      import io
      import threading

      step_cb = _step_callback(job_id)

      class StepCapture(io.TextIOWrapper):
          """Wraps original stderr and intercepts [TraceIQ] lines."""
          def __init__(self, original):
              self._orig = original
              self._buf = ""

          def write(self, s):
              self._orig.write(s)
              self._orig.flush()
              self._buf += s
              while "\n" in self._buf:
                  line, self._buf = self._buf.split("\n", 1)
                  step_cb(line)

          def flush(self):
              self._orig.flush()

          def fileno(self):
              return self._orig.fileno()

      orig_stderr = sys.stderr
      sys.stderr = StepCapture(orig_stderr)

      try:
          from agent.hypothesis_agent import run_hypothesis_agent
          result = run_hypothesis_agent(
              api_key=api_key,
              project=project,
              hypothesis=hypothesis,
              days=days,
          )
      finally:
          sys.stderr = orig_stderr

      return result


  def _run_experiment_job(job: dict) -> dict:
      """Run an experiment analysis job. Streams steps to DB. Returns result dict."""
      job_id = job["id"]
      inp = job["input"]

      api_key = inp.get("api_key", "")
      dataset_id = inp.get("dataset_id", "")
      experiment_id = inp.get("experiment_id", "")
      question = inp.get("question", "Analyze this experiment and recommend prompt improvements")

      os.environ["LANGSMITH_API_KEY"] = api_key

      import io
      import sys

      step_cb = _step_callback(job_id)

      class StepCapture(io.TextIOWrapper):
          def __init__(self, original):
              self._orig = original
              self._buf = ""

          def write(self, s):
              self._orig.write(s)
              self._orig.flush()
              self._buf += s
              while "\n" in self._buf:
                  line, self._buf = self._buf.split("\n", 1)
                  step_cb(line)

          def flush(self):
              self._orig.flush()

          def fileno(self):
              return self._orig.fileno()

      orig_stderr = sys.stderr
      sys.stderr = StepCapture(orig_stderr)

      try:
          from agent.prompt_advisor import run_prompt_advisor
          result = run_prompt_advisor(
              api_key=api_key,
              dataset_id=dataset_id,
              experiment_id=experiment_id,
              question=question,
          )
      finally:
          sys.stderr = orig_stderr

      return result


  def process_job(job: dict) -> None:
      """Route a job to the right runner; update DB with result or error."""
      job_id = job["id"]
      job_type = job["job_type"]

      print(f"[worker] processing job {job_id} type={job_type}", flush=True)

      try:
          if job_type == "hypothesis":
              result = _run_hypothesis_job(job)
          elif job_type == "experiment":
              result = _run_experiment_job(job)
          else:
              raise ValueError(f"Unknown job_type: {job_type}")

          # If the result itself signals an error, still mark the job done
          # (the UI will show the error from result.error)
          if result.get("error"):
              _fail_job(job_id, result["error"])
          else:
              _complete_job(job_id, result)
              # Save to history
              try:
                  from server import _save_to_history, _save_experiment_to_history
                  if job_type == "hypothesis":
                      _save_to_history(result)
                  else:
                      _save_experiment_to_history(result)
              except Exception as e:
                  print(f"[worker] warning: could not save to history: {e}", flush=True)

          print(f"[worker] job {job_id} done", flush=True)

      except Exception as e:
          import traceback
          tb = traceback.format_exc()
          print(f"[worker] job {job_id} FAILED: {e}\n{tb}", flush=True)
          _fail_job(job_id, str(e))


  def main():
      print("[worker] starting, polling for jobs...", flush=True)
      while True:
          job = _claim_next_job()
          if job:
              process_job(job)
          else:
              time.sleep(POLL_INTERVAL)


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2 — Verify worker imports**

  ```bash
  cd ~/Documents/traceiq
  python3 -c "import worker; print('worker imports OK')"
  ```

- [ ] **Step 3 — Smoke test locally (requires DATABASE_URL)**

  ```bash
  # Terminal 1: start worker
  DATABASE_URL="$DATABASE_URL" python3 worker.py

  # Terminal 2: manually insert a test job and watch it get picked up
  python3 -c "
  from server import _create_job
  jid = _create_job('hypothesis', {
      'api_key': 'test-key',
      'project': 'test-project',
      'hypothesis': 'errors are higher on Mondays',
      'days': 7,
  })
  print('inserted job:', jid)
  "
  ```

- [ ] **Commit:** `git commit -m "feat: add worker.py for async job processing"`

---

### Task 3: New start endpoints — POST /analyze/start and POST /experiments/analyze/start

**Files:** `server.py`

- [ ] **Step 1 — Add `POST /analyze/start`**

  Add immediately after the existing `@app.route("/analyze", ...)` block:

  ```python
  @app.route("/analyze/start", methods=["POST"])
  def analyze_start():
      """
      POST /analyze/start
      Body: {api_key OR session_id, hypothesis, project, days, split_mode}
      Returns: {job_id}
      Enqueues a hypothesis analysis job; worker.py picks it up asynchronously.
      """
      data = request.get_json(force=True) or {}

      # Resolve API key
      api_key = (_resolve_api_key(
          request_args=data,
          request_json=data,
      ) or data.get("api_key", "")).strip()

      hypothesis = data.get("hypothesis", "").strip()
      project = data.get("project", "").strip()
      days = int(data.get("days", 30))
      split_mode = data.get("split_mode", "agent").strip()

      if not hypothesis:
          return jsonify({"error": "hypothesis is required"}), 400
      if not api_key:
          return jsonify({"error": "api_key is required (or valid session_id)"}), 400
      if not project:
          return jsonify({"error": "project is required"}), 400
      if split_mode not in ("prompt_change", "time_split", "none", "agent"):
          split_mode = "agent"

      job_input = {
          "api_key": api_key,
          "hypothesis": hypothesis,
          "project": project,
          "days": days,
          "split_mode": split_mode,
      }

      job_id = _create_job("hypothesis", job_input)
      if not job_id:
          return jsonify({"error": "Failed to create job — DATABASE_URL may not be set"}), 500

      print(f"[server] queued hypothesis job {job_id} project={project}", flush=True)
      return jsonify({"job_id": job_id})
  ```

- [ ] **Step 2 — Add `POST /experiments/analyze/start`**

  Add immediately after the existing `@app.route("/experiments/analyze", ...)` block:

  ```python
  @app.route("/experiments/analyze/start", methods=["POST"])
  def experiments_analyze_start():
      """
      POST /experiments/analyze/start
      Body: {api_key OR session_id, dataset_id, experiment_id, question}
      Returns: {job_id}
      Enqueues an experiment analysis job; worker.py picks it up asynchronously.
      """
      data = request.get_json(force=True) or {}

      api_key = (_resolve_api_key(
          request_args=data,
          request_json=data,
      ) or data.get("api_key", "")).strip()

      dataset_id = data.get("dataset_id", "").strip()
      experiment_id = data.get("experiment_id", "").strip()
      question = data.get("question", "").strip()

      if not api_key:
          return jsonify({"error": "api_key is required (or valid session_id)"}), 400
      if not dataset_id:
          return jsonify({"error": "dataset_id is required"}), 400
      if not experiment_id:
          return jsonify({"error": "experiment_id is required"}), 400
      if not question:
          question = "Analyze this experiment and recommend prompt improvements"

      job_input = {
          "api_key": api_key,
          "dataset_id": dataset_id,
          "experiment_id": experiment_id,
          "question": question,
      }

      job_id = _create_job("experiment", job_input)
      if not job_id:
          return jsonify({"error": "Failed to create job — DATABASE_URL may not be set"}), 500

      print(f"[server] queued experiment job {job_id} dataset={dataset_id}", flush=True)
      return jsonify({"job_id": job_id})
  ```

- [ ] **Step 3 — Verify**

  ```bash
  # Start server locally
  python3 server.py &

  # Test hypothesis start
  curl -s -X POST http://localhost:5000/analyze/start \
    -H 'Content-Type: application/json' \
    -d '{"api_key":"test","hypothesis":"errors spike on weekends","project":"my-project","days":30}' | python3 -m json.tool
  # Expected: {"job_id": "<uuid>"}

  # Test experiment start
  curl -s -X POST http://localhost:5000/experiments/analyze/start \
    -H 'Content-Type: application/json' \
    -d '{"api_key":"test","dataset_id":"ds-123","experiment_id":"exp-456","question":"why is accuracy low?"}' | python3 -m json.tool
  # Expected: {"job_id": "<uuid>"}
  ```

- [ ] **Commit:** `git commit -m "feat: add /analyze/start and /experiments/analyze/start endpoints"`

---

### Task 4: Job status endpoint — GET /jobs/{job_id}

**Files:** `server.py`

- [ ] **Step 1 — Add the endpoint**

  Add after the new start endpoints:

  ```python
  @app.route("/jobs/<job_id>", methods=["GET"])
  def get_job_status(job_id: str):
      """
      GET /jobs/<job_id>
      Returns: {id, status, job_type, steps, result, error, created_at, updated_at}
      Polled by the UI every 2s to track job progress.
      """
      job = _get_job(job_id)
      if not job:
          return jsonify({"error": "Job not found"}), 404
      return jsonify(job)
  ```

- [ ] **Step 2 — Verify**

  ```bash
  # Using a real job_id from Task 3 test:
  JOB_ID="<uuid-from-above>"
  curl -s "http://localhost:5000/jobs/$JOB_ID" | python3 -m json.tool
  # Expected:
  # {
  #   "id": "<uuid>",
  #   "status": "queued",
  #   "job_type": "hypothesis",
  #   "steps": [],
  #   "result": null,
  #   "error": null,
  #   "created_at": "2026-03-23T...",
  #   "updated_at": "2026-03-23T..."
  # }
  ```

- [ ] **Commit:** `git commit -m "feat: add GET /jobs/<job_id> status endpoint"`

---

### Task 5: UI changes — replace EventSource with polling

**Files:** `index.html`

Replace both `runAnalysis()` and `runExpAnalysis()` functions to use polling instead of EventSource.

- [ ] **Step 1 — Add shared polling helper** (insert in the `<script>` block, before `runAnalysis`):

  ```javascript
  // ── Async job polling ─────────────────────────────────────────────────────

  /**
   * Poll /jobs/<job_id> every 2s until status is 'done' or 'failed'.
   * Calls onStep(stepText) for each new step, onDone(result) on success,
   * onError(message) on failure.
   */
  function pollJob(jobId, { onStep, onDone, onError }) {
    let lastStepCount = 0;
    let pollTimer = null;
    let stopped = false;

    function stop() {
      stopped = true;
      if (pollTimer) clearTimeout(pollTimer);
    }

    async function tick() {
      if (stopped) return;
      try {
        const resp = await fetch(`/jobs/${encodeURIComponent(jobId)}`);
        if (!resp.ok) {
          onError(`Server error: ${resp.status}`);
          stop();
          return;
        }
        const job = await resp.json();

        // Dispatch any new steps
        const steps = job.steps || [];
        for (let i = lastStepCount; i < steps.length; i++) {
          onStep(steps[i]);
        }
        lastStepCount = steps.length;

        if (job.status === 'done') {
          stop();
          onDone(job.result);
        } else if (job.status === 'failed') {
          stop();
          onError(job.error || 'Job failed');
        } else {
          // still queued or running — poll again
          pollTimer = setTimeout(tick, 2000);
        }
      } catch (e) {
        onError(`Polling failed: ${e.message}`);
        stop();
      }
    }

    tick();
    return { stop };
  }
  ```

- [ ] **Step 2 — Replace `runAnalysis()`**

  Find the existing `async function runAnalysis() { ... }` function and replace it entirely with:

  ```javascript
  async function runAnalysis() {
    const text = document.getElementById('hypothesis-text').value.trim();
    if (!text) {
      alert('Please describe your hypothesis first.');
      return;
    }
    const days = parseInt(document.getElementById('days-select').value);
    state.hypothesis = 'I think ' + text;

    // Show loading
    document.getElementById('loading-card').style.display = 'block';
    document.getElementById('analyse-btn').disabled = true;
    resetLoadingSteps();

    let jobPoller = null;

    function cleanup() {
      document.getElementById('loading-card').style.display = 'none';
      document.getElementById('analyse-btn').disabled = false;
      if (jobPoller) jobPoller.stop();
    }

    try {
      // 1. Enqueue the job
      const resp = await fetch('/analyze/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis: state.hypothesis,
          session_id: state.sessionId || state.apiKey,
          project: state.project,
          days: String(days),
          split_mode: state.splitMode,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        cleanup();
        renderError('Failed to start analysis: ' + (err.error || resp.status));
        showScreen('screen-results');
        return;
      }

      const { job_id } = await resp.json();
      if (!job_id) {
        cleanup();
        renderError('No job_id returned from server');
        showScreen('screen-results');
        return;
      }

      // 2. Poll for progress
      jobPoller = pollJob(job_id, {
        onStep(stepText) {
          const activeStep = document.querySelector('.loading-steps li.active');
          if (activeStep) {
            const icon = activeStep.querySelector('.step-icon');
            if (icon) icon.textContent = '⚡';
            activeStep.lastChild.textContent = ' ' + stepText;
          }
        },
        onDone(result) {
          cleanup();
          playDoneChime();
          setStep('step-done');
          state.result = result;
          renderResults(result);
          document.getElementById('results-logo-sub').textContent = 'Hypothesis Testing';
          document.getElementById('results-back-btn').textContent = '← New hypothesis';
          document.getElementById('results-back-btn').onclick = () => showHypothesisScreen();
          showScreen('screen-results');
        },
        onError(message) {
          cleanup();
          renderError('Analysis failed: ' + message);
          document.getElementById('results-logo-sub').textContent = 'Hypothesis Testing';
          document.getElementById('results-back-btn').textContent = '← New hypothesis';
          document.getElementById('results-back-btn').onclick = () => showHypothesisScreen();
          showScreen('screen-results');
        },
      });

    } catch (e) {
      cleanup();
      renderError('Failed to start analysis: ' + e.message);
      showScreen('screen-results');
    }
  }
  ```

- [ ] **Step 3 — Replace `runExpAnalysis()`**

  Find the existing `function runExpAnalysis() { ... }` and replace it entirely with:

  ```javascript
  function runExpAnalysis() {
    const question = document.getElementById('exp-question-input').value.trim();
    if (!state.apiKey || !state.expDatasetId || !state.expExperimentId) {
      alert('Please select a dataset and experiment first.');
      return;
    }
    if (!question) {
      alert('Please enter your question first.');
      return;
    }

    // Show loading
    document.getElementById('exp-loading-card').style.display = 'block';
    document.getElementById('exp-analyse-btn').disabled = true;
    document.getElementById('exp-connect-btn').disabled = true;
    resetExpLoadingSteps();

    let jobPoller = null;

    function cleanup() {
      document.getElementById('exp-loading-card').style.display = 'none';
      document.getElementById('exp-analyse-btn').disabled = false;
      document.getElementById('exp-connect-btn').disabled = false;
      if (jobPoller) jobPoller.stop();
    }

    fetch('/experiments/analyze/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId || state.apiKey,
        dataset_id: state.expDatasetId,
        experiment_id: state.expExperimentId,
        question: question,
      }),
    })
    .then(resp => {
      if (!resp.ok) return resp.json().then(e => Promise.reject(e.error || resp.status));
      return resp.json();
    })
    .then(({ job_id }) => {
      if (!job_id) throw new Error('No job_id returned');

      jobPoller = pollJob(job_id, {
        onStep(stepText) {
          const activeStep = document.querySelector('#exp-loading-steps li.active');
          if (activeStep) {
            const icon = activeStep.querySelector('.step-icon');
            if (icon) icon.textContent = '⚡';
            activeStep.lastChild.textContent = ' ' + stepText;
          }
        },
        onDone(result) {
          cleanup();
          playDoneChime();
          setExpStep('exp-step-done');
          state.result = result;
          renderExpResults(result);
          document.getElementById('results-logo-sub').textContent = 'Experiments';
          document.getElementById('results-back-btn').textContent = '← New experiment';
          document.getElementById('results-back-btn').onclick = () => showScreen('screen-experiment');
          showScreen('screen-results');
        },
        onError(message) {
          cleanup();
          renderError('Analysis failed: ' + message);
          document.getElementById('results-logo-sub').textContent = 'Experiments';
          document.getElementById('results-back-btn').textContent = '← New experiment';
          document.getElementById('results-back-btn').onclick = () => showScreen('screen-experiment');
          showScreen('screen-results');
        },
      });
    })
    .catch(e => {
      cleanup();
      renderError('Failed to start analysis: ' + e);
      document.getElementById('results-logo-sub').textContent = 'Experiments';
      document.getElementById('results-back-btn').textContent = '← New experiment';
      document.getElementById('results-back-btn').onclick = () => showScreen('screen-experiment');
      showScreen('screen-results');
    });
  }
  ```

- [ ] **Step 4 — Verify UI**

  Open the app in a browser. Click "Analyse" — the loading card should appear. Check the Network tab:
  - One `POST /analyze/start` call → should return `{job_id}`
  - Repeated `GET /jobs/<job_id>` calls every ~2s → steps should populate

- [ ] **Commit:** `git commit -m "feat: replace EventSource SSE with async job polling in UI"`

---

### Task 6: Railway worker service

**Files:** `Procfile`, Railway dashboard (manual step)

- [ ] **Step 1 — Update Procfile**

  Current:
  ```
  web: gunicorn --worker-class gevent -w 1 --worker-connections 50 server:app --bind 0.0.0.0:$PORT --timeout 600 --keep-alive 650
  ```

  Replace with:
  ```
  web: gunicorn --worker-class gevent -w 1 --worker-connections 50 server:app --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 130
  worker: python3 worker.py
  ```

  Note: `--timeout` can now be reduced from 600s to 120s since no more long-running requests.

- [ ] **Step 2 — Railway dashboard setup**

  1. Go to [railway.app](https://railway.app) → your TraceIQ project
  2. Click **New Service** → **GitHub Repo** → select the same repo as the web service
  3. Set the **Start Command** to: `python3 worker.py`
  4. Add the same environment variables as the web service:
     - `DATABASE_URL` (shared Postgres — Railway provides this automatically if you use the same linked DB)
     - `ANTHROPIC_API_KEY`
     - Any other env vars the agents need
  5. Deploy

  > **Note:** Railway's "Worker" service type runs without an HTTP port, which is correct for `worker.py`. It will auto-restart on crash.

- [ ] **Step 3 — Verify on Railway**

  After deploying both services:
  ```bash
  # Check worker logs in Railway dashboard — should see:
  # [worker] starting, polling for jobs...

  # Trigger a real job via the UI and watch the worker logs:
  # [worker] processing job <uuid> type=hypothesis
  # [worker] step: Fetching traces for 'my-project' (last 30 days)...
  # [worker] step: Running agent analysis...
  # [worker] job <uuid> done
  ```

- [ ] **Commit:** `git commit -m "feat: update Procfile for Railway worker service, reduce web timeout"`

---

### Task 7: Keep SSE for local dev (no changes needed)

**Files:** `server.py` (no changes), `index.html` (note only)

The existing SSE endpoints stay exactly as they are:
- `GET /analyze/stream` — still works
- `GET /experiments/analyze/stream` — still works

The UI now always uses async jobs (polling). This works locally too, as long as `worker.py` is running in a second terminal.

**Local dev workflow:**
```bash
# Terminal 1: Flask server
DATABASE_URL="postgresql://..." python3 server.py

# Terminal 2: Worker
DATABASE_URL="postgresql://..." python3 worker.py
```

If you want SSE locally without a worker (e.g. quick testing without Postgres), you can still `curl` the old SSE endpoints directly:
```bash
curl -N "http://localhost:5000/analyze/stream?hypothesis=test&project=my-project&api_key=lsv2_..."
```

> **Optional enhancement (not required):** Add a `USE_ASYNC_JOBS` env var (default `true` on Railway, `false` locally if `DATABASE_URL` not set). The UI `runAnalysis()` can check `window._useAsyncJobs` injected via a `/api/config` endpoint. Skip this unless the local dev workflow becomes painful.

- [ ] **Commit:** `git commit -m "docs: note that SSE endpoints are preserved for local dev"`

---

## End-to-End Verification Checklist

After all tasks are complete:

```bash
# 1. DB schema
psql $DATABASE_URL -c "\d jobs"
# Should show: id, status, job_type, input, steps, result, error, created_at, updated_at

# 2. Create a job via the API
JOB_ID=$(curl -s -X POST https://<your-app>.up.railway.app/analyze/start \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"<langsmith-key>","hypothesis":"errors are higher on Mondays","project":"my-project","days":7}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"

# 3. Poll status
curl -s "https://<your-app>.up.railway.app/jobs/$JOB_ID" | python3 -m json.tool
# Should show status: queued → running → done (as worker picks it up)

# 4. Watch worker logs in Railway dashboard
# Should see [worker] step: lines streaming in

# 5. Final state
curl -s "https://<your-app>.up.railway.app/jobs/$JOB_ID" | python3 -c "
import sys,json
j = json.load(sys.stdin)
print('Status:', j['status'])
print('Steps:', j['steps'])
print('Result keys:', list(j['result'].keys()) if j['result'] else None)
"
```

## Rollback Plan

If something goes wrong:
1. Old SSE endpoints are untouched — they still work
2. Revert `index.html` to restore `EventSource` (both `runAnalysis` and `runExpAnalysis`)
3. Remove the worker service from Railway
4. Revert `Procfile` timeout change

The DB schema changes (`jobs` table) are additive and non-breaking.
