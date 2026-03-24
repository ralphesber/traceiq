#!/usr/bin/env python3
"""
TraceIQ Worker — polls Postgres for queued jobs and runs them.

Simple, direct execution — no subprocesses, no threads.
The agent runs in the worker process directly.
"""

import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from server import (
    _get_db_conn,
    _init_db,
    _append_job_step,
    _complete_job,
    _fail_job,
)

POLL_INTERVAL = 2  # seconds between polls when queue is empty


def _claim_next_job() -> dict | None:
    """Atomically claim the oldest queued job."""
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


def process_job(job: dict) -> None:
    """Run a job directly — no subprocess, no threads."""
    job_id = job["id"]
    job_type = job["job_type"]
    inp = job["input"]

    print(f"[worker] processing job {job_id} type={job_type}", flush=True)

    # Step logger — writes directly to DB
    def log_step(text: str):
        print(f"[worker] step: {text}", flush=True)
        _append_job_step(job_id, text)

    try:
        os.environ["LANGSMITH_API_KEY"] = inp.get("api_key", "")

        if job_type == "experiment":
            from agent.prompt_advisor import run_prompt_advisor
            log_step(f"Starting analysis...")
            result = run_prompt_advisor(
                api_key=inp["api_key"],
                dataset_id=inp["dataset_id"],
                experiment_id=inp["experiment_id"],
                question=inp.get("question", ""),
                step_callback=log_step,
            )

        elif job_type == "hypothesis":
            from agent.hypothesis_agent import run_hypothesis_agent
            log_step(f"Starting hypothesis analysis...")
            result = run_hypothesis_agent(
                api_key=inp["api_key"],
                project=inp["project"],
                hypothesis=inp["hypothesis"],
                days=int(inp.get("days", 30)),
            )

        else:
            raise ValueError(f"Unknown job_type: {job_type}")

        if result.get("error"):
            print(f"[worker] job {job_id} completed with error: {result['error']}", flush=True)
            _fail_job(job_id, result["error"])
        else:
            print(f"[worker] job {job_id} done", flush=True)
            _complete_job(job_id, result)
            # Save to history
            try:
                from server import _save_experiment_to_history, _save_to_history
                if job_type == "experiment":
                    _save_experiment_to_history(result)
                else:
                    _save_to_history(result)
            except Exception as e:
                print(f"[worker] warning: could not save to history: {e}", flush=True)

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
