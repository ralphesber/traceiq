#!/usr/bin/env python3
"""
TraceIQ - LangSmith Agent Trace Analysis CLI

Analyzes LangSmith traces to generate insights on latency, costs,
errors, and detect regressions or configuration changes.

Modes:
  --mode insights  (default) LLM-synthesized product-language insights
  --mode metrics   Legacy metrics report (latency, cost, error rates)
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


LANGSMITH_API_BASE = "https://api.smith.langchain.com"

# Default cost model (per 1K tokens, input/output averaged)
DEFAULT_MODEL_COSTS = {
    "gpt-4o": 0.0075,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.015,
    "gpt-4": 0.045,
    "gpt-3.5-turbo": 0.001,
    "claude-3-opus": 0.0375,
    "claude-3-sonnet": 0.009,
    "claude-3-haiku": 0.000625,
    "claude-3-5-sonnet": 0.009,
    "default": 0.005,
}

# Cost model loaded from config file
MODEL_COSTS: dict[str, float] = {}


def load_cost_model() -> dict[str, float]:
    """Load cost model from cost_model.json, creating it if it doesn't exist."""
    cost_file = Path(__file__).parent / "cost_model.json"

    if not cost_file.exists():
        with open(cost_file, "w") as f:
            json.dump(DEFAULT_MODEL_COSTS, f, indent=2)
        print(f"[TraceIQ] Created default cost model config: {cost_file}", file=sys.stderr)
        return DEFAULT_MODEL_COSTS.copy()

    try:
        with open(cost_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[TraceIQ] Warning: could not load cost_model.json ({e}), using defaults", file=sys.stderr)
        return DEFAULT_MODEL_COSTS.copy()


def get_model_cost(model: str) -> float:
    """Get cost per 1K tokens for a model, with fallback to default."""
    if model in MODEL_COSTS:
        return MODEL_COSTS[model]
    print(f"[TraceIQ] Note: model '{model}' not found in cost_model.json, using default rate", file=sys.stderr)
    return MODEL_COSTS.get("default", DEFAULT_MODEL_COSTS["default"])


def get_api_key(args_api_key: str | None) -> str:
    """Get API key from args or environment."""
    api_key = args_api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set LANGSMITH_API_KEY env var.")
        sys.exit(1)
    return api_key


def resolve_session_id(api_key: str, project_name: str) -> str:
    """Resolve a project name to its LangSmith session UUID."""
    headers = {"x-api-key": api_key}
    try:
        response = requests.get(
            f"{LANGSMITH_API_BASE}/api/v1/sessions",
            headers=headers,
            params={"name": project_name},
            timeout=30,
        )
        response.raise_for_status()
        sessions = response.json()
        if not sessions:
            print(f"Error: No project found with name '{project_name}'", file=sys.stderr)
            sys.exit(1)
        return sessions[0]["id"]
    except requests.RequestException as e:
        print(f"Error resolving project session: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_runs(
    api_key: str,
    project_name: str,
    days: int,
    page_delay: float = 0.1,
    max_retries: int = 3,
    default_retry_after: int = 60,
) -> list[dict]:
    """Fetch runs from LangSmith API with rate limiting."""
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    session_id = resolve_session_id(api_key, project_name)
    print(f"[TraceIQ] Resolved project '{project_name}' → session {session_id}", file=sys.stderr)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_runs = []
    cursor = None
    is_first_request = True

    while True:
        payload: dict[str, Any] = {
            "session": [session_id],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "limit": 100,
        }
        if cursor:
            payload["cursor"] = cursor

        if not is_first_request:
            time.sleep(page_delay)
        is_first_request = False

        retries = 0
        while retries <= max_retries:
            try:
                response = requests.post(
                    f"{LANGSMITH_API_BASE}/api/v1/runs/query",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 429:
                    retries += 1
                    if retries > max_retries:
                        print(f"Error: Rate limited after {max_retries} retries", file=sys.stderr)
                        sys.exit(1)
                    retry_after = int(response.headers.get("Retry-After", default_retry_after))
                    print(f"[TraceIQ] Rate limited (429), waiting {retry_after}s (retry {retries}/{max_retries})", file=sys.stderr)
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                break

            except requests.RequestException as e:
                print(f"Error fetching runs: {e}", file=sys.stderr)
                sys.exit(1)

        data = response.json()
        runs = data.get("runs", data) if isinstance(data, dict) else data

        if not runs:
            break

        all_runs.extend(runs)

        if isinstance(data, dict) and data.get("cursors", {}).get("next"):
            cursor = data["cursors"]["next"]
        else:
            break

    return all_runs


# ─── Helper extractors (shared between modes) ──────────────────────────────

def calculate_percentile(values: list[float], percentile: int) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[-1]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def extract_latency(run: dict) -> float | None:
    """Extract latency in seconds from a run."""
    start = run.get("start_time")
    end = run.get("end_time")
    if not start or not end:
        return None
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end - start).total_seconds()


def extract_tokens(run: dict) -> int:
    """Extract total tokens from a run."""
    if "total_tokens" in run:
        return run["total_tokens"]
    feedback = run.get("feedback_stats", {})
    if "total_tokens" in feedback:
        return feedback["total_tokens"]
    extra = run.get("extra", {}) or {}
    if "total_tokens" in extra:
        return extra["total_tokens"]
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict):
        usage = outputs.get("usage", {}) or {}
        total = usage.get("total_tokens", 0)
        if total:
            return total
    prompt_tokens = run.get("prompt_tokens", 0) or extra.get("prompt_tokens", 0)
    completion_tokens = run.get("completion_tokens", 0) or extra.get("completion_tokens", 0)
    total = prompt_tokens + completion_tokens
    if total == 0:
        run_id = run.get("id", "unknown")
        print(f"[TraceIQ] Warning: could not extract token count for run {run_id}", file=sys.stderr)
    return total


def extract_model(run: dict) -> str | None:
    """Extract model name from a run."""
    extra = run.get("extra", {}) or {}
    if run.get("model"):
        return run["model"]
    invocation = extra.get("invocation_params", {}) or {}
    if invocation.get("model"):
        return invocation["model"]
    if invocation.get("model_name"):
        return invocation["model_name"]
    metadata = extra.get("metadata", {}) or {}
    if metadata.get("model"):
        return metadata["model"]
    if metadata.get("ls_model_name"):
        return metadata["ls_model_name"]
    serialized = run.get("serialized", {}) or {}
    kwargs = serialized.get("kwargs", {}) or {}
    if kwargs.get("model"):
        return kwargs["model"]
    if kwargs.get("model_name"):
        return kwargs["model_name"]
    return None


def extract_system_prompt(run: dict) -> str | None:
    """Extract system prompt from a run."""
    inputs = run.get("inputs", {}) or {}
    messages = inputs.get("messages", [])
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "") or msg.get("type", "")
                if role in ("system", "SystemMessage"):
                    return msg.get("content", "")
    if inputs.get("system"):
        return inputs["system"]
    if inputs.get("system_prompt"):
        return inputs["system_prompt"]
    return None


def hash_prompt(prompt: str | None) -> str:
    """Create a short hash of a prompt."""
    if not prompt:
        return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def extract_error(run: dict) -> str | None:
    """Extract error message from a failed run."""
    if run.get("status") != "error" and not run.get("error"):
        return None
    error = run.get("error")
    if error:
        if len(error) > 100:
            return error[:100] + "..."
        return error
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict) and outputs.get("error"):
        err = outputs["error"]
        if len(err) > 100:
            return err[:100] + "..."
        return err
    return "Unknown error"


def get_run_date(run: dict) -> datetime | None:
    """Get the date of a run, always returning a timezone-aware datetime."""
    start = run.get("start_time")
    if not start:
        return None
    if isinstance(start, str):
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(start, datetime):
        if start.tzinfo is None:
            return start.replace(tzinfo=timezone.utc)
        return start
    return None


# ─── Smart sampling ────────────────────────────────────────────────────────

def smart_sample_runs(runs: list[dict], target: int = 40) -> list[dict]:
    """
    Select ~target representative runs for LLM analysis:
    - Most recent errors (up to 1/3 of target)
    - Highest latency outliers (up to 1/4 of target)
    - Recent successful runs (baseline, up to remainder)
    Deduplicates by run ID.
    """
    seen_ids: set[str] = set()
    sampled: list[dict] = []

    def add(r: dict) -> bool:
        rid = r.get("id", "")
        if rid in seen_ids:
            return False
        seen_ids.add(rid)
        sampled.append(r)
        return True

    # 1. Most recent errors (cap at target//3)
    error_runs = sorted(
        [r for r in runs if r.get("status") == "error" or r.get("error")],
        key=lambda r: r.get("start_time", ""),
        reverse=True,
    )
    error_quota = max(5, target // 3)
    for r in error_runs[:error_quota]:
        add(r)

    # 2. Highest latency outliers (cap at target//4)
    latency_runs = sorted(
        [r for r in runs if extract_latency(r) is not None],
        key=lambda r: extract_latency(r) or 0,
        reverse=True,
    )
    latency_quota = max(5, target // 4)
    for r in latency_runs[:latency_quota]:
        add(r)

    # 3. Fill the rest with most recent successes
    success_runs = sorted(
        [r for r in runs if r.get("status") != "error" and not r.get("error")],
        key=lambda r: r.get("start_time", ""),
        reverse=True,
    )
    remaining = target - len(sampled)
    for r in success_runs:
        if len(sampled) >= target:
            break
        add(r)

    print(f"[TraceIQ] Sampled {len(sampled)} traces ({len(error_runs)} errors, {len(latency_runs)} latency, {len(success_runs)} successes in pool)", file=sys.stderr)
    return sampled


# ─── Trace serialisation for LLM ──────────────────────────────────────────

def _truncate(text: str | None, max_len: int = 300) -> str:
    if not text:
        return ""
    s = str(text)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def trace_to_llm_dict(run: dict) -> dict:
    """Convert a run to a compact dict suitable for LLM consumption."""
    inputs = run.get("inputs", {}) or {}
    outputs = run.get("outputs", {}) or {}

    # Build input snippet
    if isinstance(inputs, dict):
        # Try common keys
        inp = inputs.get("input") or inputs.get("question") or inputs.get("query") or inputs.get("messages")
        if isinstance(inp, list):
            # messages list - grab last user message
            for msg in reversed(inp):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    inp = msg.get("content", "")
                    break
            else:
                inp = str(inp)
        input_snippet = _truncate(inp)
    else:
        input_snippet = _truncate(str(inputs))

    # Build output snippet
    if isinstance(outputs, dict):
        out = outputs.get("output") or outputs.get("answer") or outputs.get("result") or outputs.get("content") or outputs.get("text")
        if not out and outputs.get("messages"):
            msgs = outputs["messages"]
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    out = last.get("content", "")
        output_snippet = _truncate(out)
    else:
        output_snippet = _truncate(str(outputs))

    latency = extract_latency(run)
    tokens = extract_tokens(run)
    model = extract_model(run)
    error = extract_error(run)
    run_date = get_run_date(run)

    return {
        "id": run.get("id", ""),
        "timestamp": run_date.isoformat() if run_date else "",
        "status": "error" if (run.get("status") == "error" or run.get("error")) else "success",
        "latency_s": round(latency, 2) if latency is not None else None,
        "tokens": tokens or None,
        "model": model,
        "input_snippet": input_snippet,
        "output_snippet": output_snippet,
        "error": error,
    }


# ─── LLM synthesis ────────────────────────────────────────────────────────

def synthesize_with_llm(sampled_runs: list[dict], project: str) -> dict:
    """
    Call OpenAI with sampled traces and return structured insight dict
    matching the agreed output schema.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[TraceIQ] Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[TraceIQ] Error: OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    n = len(sampled_runs)

    # Prepare trace summaries for the LLM
    trace_dicts = [trace_to_llm_dict(r) for r in sampled_runs]
    traces_json = json.dumps(trace_dicts, indent=2)

    # Step 1: qualitative analysis prompt
    analysis_prompt = f"""You are analyzing AI agent traces for a product team. Here are {n} traces from LangSmith project "{project}".

TRACES:
{traces_json}

Identify:
1. What's working well (reliable patterns, consistent outputs, good performance)
2. What's broken or struggling (errors, slow responses, failure patterns — be specific about which traces)
3. What has changed recently (shifts in latency, error rate, output patterns compared to older traces)

Respond in product language a PM can act on. Be specific — point to patterns across trace IDs, not generalities. 
Note which trace IDs support each observation.
Write a single headline sentence summarizing the overall state of the agent right now.

Format your response as valid JSON with this structure:
{{
  "headline": "One sentence: overall state of the agent right now",
  "working": [
    {{"summary": "...", "trace_ids": ["id1", "id2"], "since": "approximate date or timeframe"}}
  ],
  "broken": [
    {{"summary": "...", "severity": "high|medium|low", "trace_ids": ["id1"], "since": "approximate date or timeframe"}}
  ],
  "changed": [
    {{"summary": "...", "trace_ids": ["id1"], "direction": "better|worse", "since": "approximate date or timeframe"}}
  ]
}}

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""

    print(f"[TraceIQ] Calling OpenAI for insight synthesis ({n} traces)...", file=sys.stderr)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=4000,
    )

    raw = response.choices[0].message.content or "{}"

    try:
        insights_raw = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[TraceIQ] Warning: LLM returned invalid JSON ({e}), attempting recovery", file=sys.stderr)
        insights_raw = {
            "headline": "Unable to parse LLM response",
            "working": [],
            "broken": [],
            "changed": [],
        }

    return insights_raw


# ─── Snapshot storage ─────────────────────────────────────────────────────

def get_snapshots_dir() -> Path:
    """Return (and create if needed) the snapshots directory."""
    d = Path(__file__).parent / "snapshots"
    d.mkdir(exist_ok=True)
    return d


def save_snapshot(output: dict, project: str) -> Path:
    """Save output JSON to snapshots/<timestamp>_<project>.json and return path."""
    snapshots_dir = get_snapshots_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_project = project.replace("/", "_").replace(" ", "_")
    filename = f"{ts}_{safe_project}.json"
    path = snapshots_dir / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[TraceIQ] Snapshot saved: {path}", file=sys.stderr)
    return path


def find_previous_snapshot(project: str, before_ts: datetime) -> dict | None:
    """
    Find the most recent snapshot for `project` with generated_at < before_ts.
    Returns parsed JSON or None.
    """
    snapshots_dir = get_snapshots_dir()
    safe_project = project.replace("/", "_").replace(" ", "_")
    candidates = []

    for f in snapshots_dir.glob(f"*_{safe_project}.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            gen_at_str = data.get("generated_at", "")
            if not gen_at_str:
                continue
            gen_at = datetime.fromisoformat(gen_at_str.replace("Z", "+00:00"))
            if gen_at.tzinfo is None:
                gen_at = gen_at.replace(tzinfo=timezone.utc)
            if gen_at < before_ts:
                candidates.append((gen_at, data))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def compute_since_labels(current_insights: dict, prev_snapshot: dict | None) -> dict:
    """
    If prev_snapshot exists, annotate insights with how long they've been present.
    Returns the insights dict with updated `since` fields where patterns overlap.
    """
    if not prev_snapshot:
        return current_insights

    prev_gen = prev_snapshot.get("generated_at", "")
    prev_broken_summaries = {
        i.get("summary", "")[:60]
        for i in prev_snapshot.get("insights", {}).get("broken", [])
    }
    prev_working_summaries = {
        i.get("summary", "")[:60]
        for i in prev_snapshot.get("insights", {}).get("working", [])
    }

    def label(summary: str, prev_set: set) -> str | None:
        # Simple substring match for now
        for p in prev_set:
            if p and (p[:40] in summary or summary[:40] in p):
                return prev_gen[:10] if prev_gen else None
        return None

    for item in current_insights.get("broken", []):
        match = label(item.get("summary", ""), prev_broken_summaries)
        if match:
            item["since"] = f"since {match} (persisting)"

    for item in current_insights.get("working", []):
        match = label(item.get("summary", ""), prev_working_summaries)
        if match:
            item["since"] = f"since {match}"

    return current_insights


# ─── Insights mode (new) ───────────────────────────────────────────────────

def run_insights_mode(runs: list[dict], project: str, since_ts: datetime | None) -> dict:
    """
    Full insights pipeline:
    1. Smart sampling
    2. LLM synthesis
    3. Build output matching agreed schema
    4. Since-diff if applicable
    """
    sampled = smart_sample_runs(runs, target=25)

    # LLM synthesis
    llm_result = synthesize_with_llm(sampled, project)

    # Build traces map
    traces_map: dict[str, dict] = {}
    for run in sampled:
        rid = run.get("id", "")
        if not rid:
            continue
        run_date = get_run_date(run)
        latency = extract_latency(run)
        is_error = run.get("status") == "error" or bool(run.get("error"))

        inputs = run.get("inputs", {}) or {}
        outputs = run.get("outputs", {}) or {}

        inp = inputs.get("input") or inputs.get("question") or inputs.get("query") or ""
        if isinstance(inp, list):
            for msg in reversed(inp):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    inp = msg.get("content", "")
                    break
            else:
                inp = str(inp)

        out = None
        if isinstance(outputs, dict):
            out = outputs.get("output") or outputs.get("answer") or outputs.get("result") or outputs.get("content") or outputs.get("text")
            if not out and outputs.get("messages"):
                msgs = outputs["messages"]
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict):
                        out = last.get("content", "")

        traces_map[rid] = {
            "input_snippet": _truncate(inp, 200),
            "output_snippet": _truncate(out, 200),
            "url": f"https://smith.langchain.com/public/{rid}/r",
            "timestamp": run_date.isoformat() if run_date else "",
            "status": "error" if is_error else "success",
        }

    # Find previous snapshot for diffing
    prev_snapshot = None
    compared_to = None
    if since_ts:
        prev_snapshot = find_previous_snapshot(project, since_ts)
        if prev_snapshot:
            compared_to = prev_snapshot.get("snapshot_id")
            print(f"[TraceIQ] Diffing against snapshot {compared_to}", file=sys.stderr)

    # Apply since labels
    insights_annotated = compute_since_labels(
        {
            "working": llm_result.get("working", []),
            "broken": llm_result.get("broken", []),
            "changed": llm_result.get("changed", []),
        },
        prev_snapshot,
    )

    snapshot_id = str(uuid.uuid4())

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "headline": llm_result.get("headline", ""),
        "trace_count_analyzed": len(sampled),
        "insights": insights_annotated,
        "traces": traces_map,
        "snapshot_id": snapshot_id,
        "compared_to": compared_to,
    }

    return output


# ─── Legacy metrics mode (unchanged logic) ────────────────────────────────

def analyze_runs(runs: list[dict], days: int) -> dict[str, Any]:
    """Perform full metrics analysis on runs."""
    if not runs:
        return {"error": "No runs found"}

    now = datetime.now(timezone.utc)
    period_midpoint = now - timedelta(days=days // 2)

    current_runs = []
    previous_runs = []

    for run in runs:
        run_date = get_run_date(run)
        if run_date:
            if run_date >= period_midpoint:
                current_runs.append(run)
            else:
                previous_runs.append(run)

    total_runs = len(runs)
    error_runs = [r for r in runs if r.get("status") == "error" or r.get("error")]
    error_count = len(error_runs)
    error_rate = (error_count / total_runs * 100) if total_runs > 0 else 0

    current_latencies = [l for r in current_runs if (l := extract_latency(r)) is not None]
    previous_latencies = [l for r in previous_runs if (l := extract_latency(r)) is not None]

    latency_stats = {
        "current": {
            "p50": calculate_percentile(current_latencies, 50),
            "p95": calculate_percentile(current_latencies, 95),
            "p99": calculate_percentile(current_latencies, 99),
        },
        "previous": {
            "p50": calculate_percentile(previous_latencies, 50),
            "p95": calculate_percentile(previous_latencies, 95),
            "p99": calculate_percentile(previous_latencies, 99),
        },
    }

    total_tokens = sum(extract_tokens(r) for r in runs)
    models_used = [m for r in runs if (m := extract_model(r))]
    primary_model = Counter(models_used).most_common(1)[0][0] if models_used else "default"
    cost_per_1k = get_model_cost(primary_model)
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    cost_per_run = estimated_cost / total_runs if total_runs > 0 else 0

    prompts_by_date: dict[str, list[tuple[datetime, str]]] = {}
    for run in runs:
        prompt = extract_system_prompt(run)
        prompt_hash = hash_prompt(prompt)
        run_date = get_run_date(run)
        if run_date and prompt_hash != "none":
            date_str = run_date.strftime("%Y-%m-%d")
            if date_str not in prompts_by_date:
                prompts_by_date[date_str] = []
            prompts_by_date[date_str].append((run_date, prompt_hash))

    prompt_changes = []
    sorted_dates = sorted(prompts_by_date.keys())
    if len(sorted_dates) >= 2:
        prev_hash = None
        for date in sorted_dates:
            hashes = [h for _, h in prompts_by_date[date]]
            current_hash = Counter(hashes).most_common(1)[0][0] if hashes else None
            if prev_hash and current_hash and prev_hash != current_hash:
                prompt_changes.append({
                    "date": date,
                    "from_hash": prev_hash,
                    "to_hash": current_hash,
                })
            if current_hash:
                prev_hash = current_hash

    models_by_date: dict[str, list[str]] = {}
    for run in runs:
        model = extract_model(run)
        run_date = get_run_date(run)
        if run_date and model:
            date_str = run_date.strftime("%Y-%m-%d")
            if date_str not in models_by_date:
                models_by_date[date_str] = []
            models_by_date[date_str].append(model)

    model_changes = []
    prev_model = None
    for date in sorted(models_by_date.keys()):
        models = models_by_date[date]
        current_model = Counter(models).most_common(1)[0][0] if models else None
        if prev_model and current_model and prev_model != current_model:
            model_changes.append({
                "date": date,
                "from_model": prev_model,
                "to_model": current_model,
            })
        if current_model:
            prev_model = current_model

    error_messages = [extract_error(r) for r in error_runs]
    error_patterns = Counter(e for e in error_messages if e).most_common(5)

    flags = []

    if latency_stats["previous"]["p95"] > 0:
        p95_change = (
            (latency_stats["current"]["p95"] - latency_stats["previous"]["p95"])
            / latency_stats["previous"]["p95"]
            * 100
        )
        if p95_change > 20:
            flags.append({
                "type": "latency_regression",
                "message": f"p95 increased {p95_change:.0f}% vs previous period "
                          f"({latency_stats['previous']['p95']:.1f}s → {latency_stats['current']['p95']:.1f}s)",
            })

    current_errors = len([r for r in current_runs if r.get("status") == "error" or r.get("error")])
    previous_errors = len([r for r in previous_runs if r.get("status") == "error" or r.get("error")])
    current_error_rate = (current_errors / len(current_runs) * 100) if current_runs else 0
    previous_error_rate = (previous_errors / len(previous_runs) * 100) if previous_runs else 0
    error_rate_change = current_error_rate - previous_error_rate
    if error_rate_change > 5:
        flags.append({
            "type": "error_rate_increase",
            "message": f"Error rate increased {error_rate_change:.1f}pp "
                      f"({previous_error_rate:.1f}% → {current_error_rate:.1f}%)",
        })

    for change in model_changes:
        flags.append({
            "type": "model_change",
            "message": f"Model changed on {change['date']}: {change['from_model']} → {change['to_model']} "
                      f"(possible quality tradeoff)",
        })

    return {
        "volume": {
            "total_runs": total_runs,
            "errors": error_count,
            "error_rate": error_rate,
            "success": total_runs - error_count,
        },
        "latency": latency_stats,
        "cost": {
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "cost_per_run": cost_per_run,
            "primary_model": primary_model,
        },
        "prompt_changes": prompt_changes,
        "model_changes": model_changes,
        "error_patterns": error_patterns,
        "flags": flags,
    }


def format_number(n: float | int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n)) if isinstance(n, float) and n == int(n) else f"{n:.2f}"


def generate_markdown_report(analysis: dict, project: str, days: int) -> str:
    """Generate markdown report from metrics analysis."""
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# TraceIQ Analysis Report",
        f"**Project:** {project} | **Period:** last {days} days | **Generated:** {now}",
        "",
    ]

    flags = analysis.get("flags", [])
    if flags:
        lines.append("## 🔴 Flags (Attention Required)")
        for flag in flags:
            lines.append(f"- {flag['message']}")
        lines.append("")

    vol = analysis["volume"]
    lines.append("## 📊 Volume")
    lines.append(
        f"- Total runs: {vol['total_runs']:,} | Errors: {vol['errors']:,} "
        f"({vol['error_rate']:.1f}%) | Success: {vol['success']:,}"
    )
    lines.append("")

    lat = analysis["latency"]
    lines.append("## ⚡ Latency")
    lines.append("| Metric | This period | Previous period | Change |")
    lines.append("|--------|-------------|-----------------|--------|")

    for metric in ["p50", "p95", "p99"]:
        current = lat["current"][metric]
        previous = lat["previous"][metric]
        if previous > 0:
            change_pct = ((current - previous) / previous) * 100
            change_str = f"+{change_pct:.0f}%" if change_pct > 0 else f"{change_pct:.0f}%"
            flag = " 🔴" if change_pct > 20 and metric == "p95" else ""
        else:
            change_str = "N/A"
            flag = ""
        lines.append(f"| {metric} | {current:.1f}s | {previous:.1f}s | {change_str}{flag} |")
    lines.append("")

    cost = analysis["cost"]
    lines.append("## 💰 Cost")
    lines.append(
        f"- Total tokens: {format_number(cost['total_tokens'])} | "
        f"Estimated cost: ${cost['estimated_cost']:.2f} | "
        f"Cost/run: ${cost['cost_per_run']:.4f}"
    )
    lines.append("")

    prompt_changes = analysis.get("prompt_changes", [])
    model_changes = analysis.get("model_changes", [])
    if prompt_changes or model_changes:
        lines.append("## 🔄 Changes Detected")
        for change in prompt_changes:
            lines.append(
                f"- {change['date']}: System prompt changed "
                f"(hash: {change['from_hash']} → {change['to_hash']})"
            )
        for change in model_changes:
            lines.append(
                f"- {change['date']}: Model changed: {change['from_model']} → {change['to_model']}"
            )
        lines.append("")

    error_patterns = analysis.get("error_patterns", [])
    if error_patterns:
        lines.append("## 🚨 Top Failure Patterns")
        total_errors = analysis["volume"]["errors"]
        for i, (error, count) in enumerate(error_patterns, 1):
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            lines.append(f'{i}. "{error}" — {count} occurrences ({pct:.0f}%)')
        lines.append("")

    return "\n".join(lines)


def generate_json_report(analysis: dict, project: str, days: int) -> str:
    """Generate JSON report from metrics analysis with standardized schema v1."""
    now = datetime.now(timezone.utc)

    alerts = []
    for flag in analysis.get("flags", []):
        severity = "warning"
        if flag["type"] == "latency_regression":
            severity = "critical"
        elif flag["type"] == "error_rate_increase":
            severity = "critical"
        elif flag["type"] == "model_change":
            severity = "warning"
        elif flag["type"] == "prompt_change":
            severity = "info"

        alerts.append({
            "severity": severity,
            "type": flag["type"],
            "message": flag["message"],
        })

    lat = analysis.get("latency", {})
    current_lat = lat.get("current", {})
    previous_lat = lat.get("previous", {})

    p95_change_pct = 0.0
    if previous_lat.get("p95", 0) > 0:
        p95_change_pct = ((current_lat.get("p95", 0) - previous_lat.get("p95", 0)) / previous_lat["p95"]) * 100

    latency = {
        "p50": current_lat.get("p50", 0.0),
        "p95": current_lat.get("p95", 0.0),
        "p99": current_lat.get("p99", 0.0),
        "prev_p50": previous_lat.get("p50", 0.0),
        "prev_p95": previous_lat.get("p95", 0.0),
        "prev_p99": previous_lat.get("p99", 0.0),
        "p95_change_pct": p95_change_pct,
    }

    cost_data = analysis.get("cost", {})
    cost = {
        "total_tokens": cost_data.get("total_tokens", 0),
        "estimated_usd": cost_data.get("estimated_cost", 0.0),
        "per_run_usd": cost_data.get("cost_per_run", 0.0),
        "primary_model": cost_data.get("primary_model", "unknown"),
    }

    vol = analysis.get("volume", {})
    prev_error_rate = 0.0
    for flag in analysis.get("flags", []):
        if flag["type"] == "error_rate_increase":
            msg = flag["message"]
            if "→" in msg and "%" in msg:
                try:
                    prev_part = msg.split("(")[1].split("→")[0].strip().rstrip("%")
                    prev_error_rate = float(prev_part)
                except (IndexError, ValueError):
                    pass
            break

    volume = {
        "total": vol.get("total_runs", 0),
        "errors": vol.get("errors", 0),
        "error_rate_pct": vol.get("error_rate", 0.0),
        "prev_error_rate_pct": prev_error_rate,
    }

    changes = {
        "prompts": analysis.get("prompt_changes", []),
        "models": analysis.get("model_changes", []),
    }

    error_patterns = analysis.get("error_patterns", [])
    total_errors = vol.get("errors", 0)
    top_patterns = []
    for error_msg, count in error_patterns:
        pct = (count / total_errors * 100) if total_errors > 0 else 0.0
        top_patterns.append({
            "message": error_msg,
            "count": count,
            "pct": pct,
        })

    errors = {
        "top_patterns": top_patterns,
    }

    report = {
        "meta": {
            "project": project,
            "window_days": days,
            "generated_at": now.isoformat(),
        },
        "alerts": alerts,
        "latency": latency,
        "cost": cost,
        "volume": volume,
        "changes": changes,
        "errors": errors,
    }

    return json.dumps(report, indent=2, default=str)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    global MODEL_COSTS

    parser = argparse.ArgumentParser(
        description="TraceIQ - LangSmith Agent Trace Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY env var)")
    parser.add_argument("--project", required=True, help="LangSmith project name")
    parser.add_argument("--days", type=int, default=7, help="Days of traces to analyze (default: 7)")
    parser.add_argument(
        "--output",
        choices=["markdown", "json"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--mode",
        choices=["insights", "metrics"],
        default="insights",
        help="Analysis mode: insights (LLM synthesis, default) or metrics (legacy latency/cost report)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp for diff mode: compare against most recent snapshot before this time (e.g. 2025-03-01T00:00:00Z)",
    )
    parser.add_argument(
        "--mock-data",
        type=str,
        help="Path to mock data JSON file (for testing without API)",
    )

    args = parser.parse_args()

    MODEL_COSTS = load_cost_model()

    # Load runs
    if args.mock_data:
        with open(args.mock_data) as f:
            runs = json.load(f)
    else:
        api_key = get_api_key(args.api_key)
        runs = fetch_runs(api_key, args.project, args.days)

    if not runs:
        print("No runs found for the specified project and time range.")
        sys.exit(1)

    print(f"[TraceIQ] Fetched {len(runs)} total runs", file=sys.stderr)

    # Parse --since
    since_ts: datetime | None = None
    if args.since:
        try:
            since_ts = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
            if since_ts.tzinfo is None:
                since_ts = since_ts.replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"[TraceIQ] Error: invalid --since timestamp ({e})", file=sys.stderr)
            sys.exit(1)

    if args.mode == "insights":
        output = run_insights_mode(runs, args.project, since_ts)
        # Save snapshot
        save_snapshot(output, args.project)
        report = json.dumps(output, indent=2, default=str)
        # Write insights_output.json for dashboard consumption
        insights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insights_output.json")
        with open(insights_path, "w") as f:
            f.write(report)
        print(f"[TraceIQ] Insights written to insights_output.json", file=sys.stderr)
    else:
        # Legacy metrics mode
        analysis = analyze_runs(runs, args.days)
        if args.output == "json":
            report = generate_json_report(analysis, args.project, args.days)
        else:
            report = generate_markdown_report(analysis, args.project, args.days)

    print(report)


if __name__ == "__main__":
    main()
