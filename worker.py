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
    _init_db,
    _update_job_status,
    _append_job_step,
    _complete_job,
    _fail_job,
)


POLL_INTERVAL = 2  # seconds between polls when queue is empty
JOB_TIMEOUT = 480  # 8 minutes max per job before we kill it


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


def _run_job_subprocess(job: dict) -> dict:
    """Run a job in a child process with a hard timeout.
    
    Streams [TraceIQ] lines from the child's stderr to _append_job_step so the
    UI still shows progress. Kills the child process on timeout — unlike threads,
    this actually terminates the hanging agent.
    """
    import subprocess, tempfile, threading, queue as _queue

    job_id = job["id"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(job, f)
        job_file = f.name

    script = f"""
import sys, json, os, traceback
print('[subprocess] starting', flush=True)
sys.stderr.write('[subprocess] stderr ok\\n')
sys.stderr.flush()
sys.path.insert(0, {repr(str(BASE_DIR))})
try:
    with open({repr(job_file)}) as f:
        job = json.load(f)
    inp = job['input']
    os.environ['LANGSMITH_API_KEY'] = inp.get('api_key', '')
    print('[subprocess] job loaded, type=' + job['job_type'], flush=True)
    if job['job_type'] == 'experiment':
        from agent.prompt_advisor import run_prompt_advisor
        result = run_prompt_advisor(
            api_key=inp['api_key'],
            dataset_id=inp['dataset_id'],
            experiment_id=inp['experiment_id'],
            question=inp.get('question', ''),
        )
    else:
        from agent.hypothesis_agent import run_hypothesis_agent
        result = run_hypothesis_agent(
            api_key=inp['api_key'],
            project=inp['project'],
            hypothesis=inp['hypothesis'],
            days=int(inp.get('days', 30)),
        )
    print(json.dumps(result, default=str))
except Exception as e:
    tb = traceback.format_exc()
    sys.stderr.write('[subprocess] FATAL: ' + str(e) + '\\n' + tb + '\\n')
    sys.stderr.flush()
    print(json.dumps({{'error': str(e), 'traceback': tb}}))
"""
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=str(BASE_DIR),
    )

    stderr_lines = _queue.Queue()

    def _read_stderr(pipe, q):
        for line in pipe:
            q.put(line)
        q.put(None)  # sentinel

    stderr_thread = threading.Thread(target=_read_stderr, args=(proc.stderr, stderr_lines), daemon=True)
    stderr_thread.start()

    deadline = time.time() + JOB_TIMEOUT
    timed_out = False

    # Drain stderr lines and relay [TraceIQ] steps to DB
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"[worker] job {job_id} hit {JOB_TIMEOUT}s timeout — killing subprocess", flush=True)
            proc.kill()
            timed_out = True
            break
        try:
            line = stderr_lines.get(timeout=min(remaining, 5))
        except _queue.Empty:
            if proc.poll() is not None:
                break  # process finished
            continue
        if line is None:
            break  # stderr closed
        line = line.strip()
        print(f"[worker] stderr: {line}", flush=True)
        if line.startswith("[TraceIQ]"):
            text = line[len("[TraceIQ]"):].strip()
            _append_job_step(job_id, text)

    try:
        os.unlink(job_file)
    except Exception:
        pass

    if timed_out:
        return {"error": f"Job timed out after {JOB_TIMEOUT} seconds — agent did not complete"}

    proc.wait()
    if proc.returncode != 0:
        # Collect any remaining stderr
        remaining_err = ""
        try:
            while True:
                line = stderr_lines.get_nowait()
                if line is None:
                    break
                remaining_err += line
        except _queue.Empty:
            pass
        return {"error": f"Worker subprocess failed (exit {proc.returncode}): {remaining_err[-800:]}"}

    stdout = proc.stdout.read()
    try:
        return json.loads(stdout)
    except Exception as e:
        return {"error": f"Could not parse job result: {e}. stdout: {stdout[:500]}"}


def process_job(job: dict) -> None:
    """Route a job to the right runner; update DB with result or error."""
    job_id = job["id"]
    job_type = job["job_type"]

    print(f"[worker] processing job {job_id} type={job_type}", flush=True)

    try:
        result = _run_job_subprocess(job)

        # If the result signals an error, fail the job
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
    print("[worker] starting, initialising DB schema...", flush=True)
    _init_db()
    print("[worker] polling for jobs...", flush=True)
    while True:
        job = _claim_next_job()
        if job:
            process_job(job)
        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
