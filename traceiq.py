#!/usr/bin/env python3
"""
TraceIQ - LangSmith Agent Trace Analysis CLI

Modes:
  --mode insights    (default) LLM-synthesized product-language insights
  --mode metrics     Legacy metrics report (latency, cost, error rates)
  --hypothesis "..."  Hypothesis testing mode: before/after prompt change analysis
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

MODEL_COSTS: dict[str, float] = {}


def load_cost_model() -> dict[str, float]:
    cost_file = Path(__file__).parent / "cost_model.json"
    if not cost_file.exists():
        with open(cost_file, "w") as f:
            json.dump(DEFAULT_MODEL_COSTS, f, indent=2)
        return DEFAULT_MODEL_COSTS.copy()
    try:
        with open(cost_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_MODEL_COSTS.copy()


def get_model_cost(model: str) -> float:
    if model in MODEL_COSTS:
        return MODEL_COSTS[model]
    return MODEL_COSTS.get("default", DEFAULT_MODEL_COSTS["default"])


def get_api_key(args_api_key: str | None) -> str:
    api_key = args_api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set LANGSMITH_API_KEY env var.")
        sys.exit(1)
    return api_key


def resolve_session_id(api_key: str, project_name: str) -> str:
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
                    print(f"[TraceIQ] Rate limited (429), waiting {retry_after}s", file=sys.stderr)
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


def fetch_prompt_commits(api_key: str, project_name: str) -> list[dict]:
    """
    Try to fetch prompt commit history from LangSmith.
    Returns list of commits sorted by date ascending, or [] if unavailable.
    Each commit: {"commit_hash": str, "created_at": str, "parent_id": str|None}
    """
    headers = {"x-api-key": api_key}

    # First try to list repos for the org/user
    try:
        resp = requests.get(
            f"{LANGSMITH_API_BASE}/api/v1/repos",
            headers=headers,
            params={"limit": 20},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        repos = resp.json()
        repo_list = repos.get("repos", repos) if isinstance(repos, dict) else repos
        if not repo_list:
            return []

        # Pick the most recently updated repo
        repo = repo_list[0]
        owner = repo.get("owner", "")
        name = repo.get("repo_name", repo.get("name", ""))
        if not owner or not name:
            return []

        # Fetch commits for this repo
        commits_resp = requests.get(
            f"{LANGSMITH_API_BASE}/api/v1/commits/{owner}/{name}",
            headers=headers,
            params={"limit": 50},
            timeout=15,
        )
        if commits_resp.status_code != 200:
            return []

        commits_data = commits_resp.json()
        commits = commits_data.get("commits", commits_data) if isinstance(commits_data, dict) else commits_data

        result = []
        for c in commits:
            result.append({
                "commit_hash": c.get("commit_hash", c.get("id", "")),
                "created_at": c.get("created_at", ""),
                "parent_id": c.get("parent_id", None),
            })

        # Sort ascending by date
        result.sort(key=lambda x: x.get("created_at", ""))
        print(f"[TraceIQ] Found {len(result)} prompt commits for {owner}/{name}", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[TraceIQ] Could not fetch prompt commits: {e}", file=sys.stderr)
        return []


# ─── Helper extractors ────────────────────────────────────────────────────

def calculate_percentile(values: list[float], percentile: int) -> float:
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
    if "total_tokens" in run:
        return run["total_tokens"]
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
    return prompt_tokens + completion_tokens


def extract_model(run: dict) -> str | None:
    extra = run.get("extra", {}) or {}
    if run.get("model"):
        return run["model"]
    invocation = extra.get("invocation_params", {}) or {}
    if invocation.get("model"):
        return invocation["model"]
    if invocation.get("model_name"):
        return invocation["model_name"]
    metadata = extra.get("metadata", {}) or {}
    if metadata.get("ls_model_name"):
        return metadata["ls_model_name"]
    serialized = run.get("serialized", {}) or {}
    kwargs = serialized.get("kwargs", {}) or {}
    if kwargs.get("model_name"):
        return kwargs["model_name"]
    return None


def extract_system_prompt(run: dict) -> str | None:
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
    return None


def hash_prompt(prompt: str | None) -> str:
    if not prompt:
        return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def extract_error(run: dict) -> str | None:
    if run.get("status") != "error" and not run.get("error"):
        return None
    error = run.get("error")
    if error:
        return error[:100] + "..." if len(error) > 100 else error
    return "Unknown error"


def get_run_date(run: dict) -> datetime | None:
    start = run.get("start_time")
    if not start:
        return None
    if isinstance(start, str):
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(start, datetime):
        return start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    return None


def _truncate(text: str | None, max_len: int = 300) -> str:
    if not text:
        return ""
    s = str(text)
    return s[:max_len] + "…" if len(s) > max_len else s


def extract_input_text(run: dict) -> str:
    """Extract the main input text from a run (student response / user message)."""
    inputs = run.get("inputs", {}) or {}
    if isinstance(inputs, dict):
        # Try common keys
        for key in ("input", "question", "query", "student_response", "response", "text", "content"):
            val = inputs.get(key)
            if val and isinstance(val, str):
                return val

        # Try messages list - grab last user message
        messages = inputs.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        return str(content)

        # Try kwargs
        kwargs = inputs.get("kwargs", {}) or {}
        for key in ("input", "question", "query"):
            val = kwargs.get(key)
            if val and isinstance(val, str):
                return val

    return str(inputs)[:500] if inputs else ""


def extract_output_text(run: dict) -> str:
    """Extract the main output text from a run (agent feedback / reasoning)."""
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict):
        for key in ("output", "answer", "result", "content", "text", "feedback", "reasoning"):
            val = outputs.get(val_key := key)
            if val and isinstance(val, str):
                return val

        # Check messages list
        messages = outputs.get("messages", [])
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
                if content:
                    return str(content)

        # Flatten any nested content
        for val in outputs.values():
            if isinstance(val, str) and len(val) > 20:
                return val

    return str(outputs)[:500] if outputs else ""


# ─── Smart sampling ────────────────────────────────────────────────────────

def smart_sample_runs(runs: list[dict], target: int = 40) -> list[dict]:
    seen_ids: set[str] = set()
    sampled: list[dict] = []

    def add(r: dict) -> bool:
        rid = r.get("id", "")
        if rid in seen_ids:
            return False
        seen_ids.add(rid)
        sampled.append(r)
        return True

    error_runs = sorted(
        [r for r in runs if r.get("status") == "error" or r.get("error")],
        key=lambda r: r.get("start_time", ""),
        reverse=True,
    )
    error_quota = max(5, target // 3)
    for r in error_runs[:error_quota]:
        add(r)

    latency_runs = sorted(
        [r for r in runs if extract_latency(r) is not None],
        key=lambda r: extract_latency(r) or 0,
        reverse=True,
    )
    latency_quota = max(5, target // 4)
    for r in latency_runs[:latency_quota]:
        add(r)

    success_runs = sorted(
        [r for r in runs if r.get("status") != "error" and not r.get("error")],
        key=lambda r: r.get("start_time", ""),
        reverse=True,
    )
    for r in success_runs:
        if len(sampled) >= target:
            break
        add(r)

    print(f"[TraceIQ] Sampled {len(sampled)} traces for analysis", file=sys.stderr)
    return sampled


def trace_to_llm_dict(run: dict) -> dict:
    inputs = run.get("inputs", {}) or {}
    outputs = run.get("outputs", {}) or {}

    inp = inputs.get("input") or inputs.get("question") or inputs.get("query") or inputs.get("messages")
    if isinstance(inp, list):
        for msg in reversed(inp):
            if isinstance(msg, dict) and msg.get("role") == "user":
                inp = msg.get("content", "")
                break
        else:
            inp = str(inp)
    input_snippet = _truncate(inp)

    out = None
    if isinstance(outputs, dict):
        out = outputs.get("output") or outputs.get("answer") or outputs.get("result") or outputs.get("content") or outputs.get("text")
        if not out and outputs.get("messages"):
            msgs = outputs["messages"]
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    out = last.get("content", "")
    output_snippet = _truncate(out)

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


# ─── Prompt change detection ───────────────────────────────────────────────

def detect_prompt_change_from_runs(runs: list[dict]) -> datetime | None:
    """
    Detect the most recent prompt change by tracking system prompt hash over time.
    Returns the datetime of the most recent change, or None if no change detected.
    """
    # Sort runs by time ascending
    sorted_runs = sorted(
        [r for r in runs if get_run_date(r)],
        key=lambda r: get_run_date(r),
    )

    if len(sorted_runs) < 4:
        return None

    # Track prompt hashes over time
    prev_hash = None
    last_change_dt = None

    for run in sorted_runs:
        prompt = extract_system_prompt(run)
        if not prompt:
            continue
        h = hash_prompt(prompt)
        if prev_hash is not None and h != prev_hash:
            last_change_dt = get_run_date(run)
        prev_hash = h

    return last_change_dt


# ─── Hypothesis mode ──────────────────────────────────────────────────────

def build_trace_summary_for_hypothesis(run: dict, max_input_len: int = 600, max_output_len: int = 800) -> dict:
    """Build a detailed trace summary for hypothesis analysis."""
    input_text = extract_input_text(run)
    output_text = extract_output_text(run)
    run_date = get_run_date(run)
    latency = extract_latency(run)

    return {
        "id": run.get("id", "")[:16],  # truncate for readability
        "timestamp": run_date.isoformat() if run_date else "",
        "input": input_text[:max_input_len] + ("…" if len(input_text) > max_input_len else ""),
        "input_length": len(input_text),
        "output": output_text[:max_output_len] + ("…" if len(output_text) > max_output_len else ""),
        "output_length": len(output_text),
        "latency_s": round(latency, 2) if latency is not None else None,
        "status": "error" if (run.get("status") == "error" or run.get("error")) else "success",
    }


def analyze_hypothesis_with_llm(
    hypothesis: str,
    before_traces: list[dict],
    after_traces: list[dict],
    prompt_change_date: str | None,
) -> dict:
    """
    Use LLM (OpenAI or Anthropic) to analyze the hypothesis across before/after traces.
    Returns the full analysis result.
    """
    # Prefer OpenAI, fall back to Anthropic
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("[TraceIQ] Error: No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Build trace summaries
    before_summaries = [build_trace_summary_for_hypothesis(r) for r in before_traces]
    after_summaries = [build_trace_summary_for_hypothesis(r) for r in after_traces]

    before_json = json.dumps(before_summaries, indent=2)
    after_json = json.dumps(after_summaries, indent=2)

    change_context = f"The prompt was last changed on {prompt_change_date}." if prompt_change_date else "No prompt change was detected; traces split at temporal midpoint."

    prompt_text = f"""You are analyzing AI agent traces to test a hypothesis.

HYPOTHESIS: "{hypothesis}"

{change_context}

TRACES BEFORE PROMPT CHANGE ({len(before_traces)} traces):
{before_json}

TRACES AFTER PROMPT CHANGE ({len(after_traces)} traces):
{after_json}

Your task:
1. First assess if the hypothesis is specific and testable with the given data. If it is too vague or broad (e.g. "is the agent performing well?"), set verdict to "too_broad" and skip the before/after analysis — instead provide good next_steps to narrow it.
2. If the hypothesis is specific but the data is insufficient to reach a conclusion (e.g. very few relevant traces), set verdict to "needs_more_data".
3. Otherwise, for each time period (before/after), analyze whether the pattern described in the hypothesis is present.
4. Look at the actual content: input lengths, output lengths, reasoning quality, and any correlations.
5. Determine if the hypothesis is supported, not_supported, inconclusive, needs_more_data, or too_broad.
6. Pick 2-3 example traces from each period that best illustrate your finding (skip if too_broad).
7. Always provide 2-3 specific, actionable next_steps — even for supported verdicts (suggest how to go deeper).
8. Always provide data_gaps — what is missing that would make this more conclusive.

Be specific and evidence-based. Reference actual trace data (lengths, content snippets) to support your conclusions.

Respond ONLY with valid JSON matching this exact schema:
{{
  "verdict": "supported|not_supported|inconclusive|needs_more_data|too_broad",
  "confidence": "high|medium|low",
  "summary": "Plain language explanation (2-4 sentences) of what you found, with specific evidence. If too_broad, explain why and what would be a better hypothesis.",
  "next_steps": [
    "Specific actionable follow-up investigation 1",
    "Specific actionable follow-up investigation 2",
    "Specific actionable follow-up investigation 3"
  ],
  "data_gaps": "What is missing that would make this more conclusive (e.g. 'Only 8 traces with very short responses — need 30+ to be confident', or 'Ground truth labels would confirm whether shorter reasoning leads to wrong scores')",
  "before_change": {{
    "pattern_present": true,
    "signal": "What pattern you observe in this period (1-2 sentences with data)",
    "example_traces": [
      {{
        "id": "trace_id",
        "input_preview": "first 80 chars of input",
        "output_preview": "first 120 chars of output",
        "relevance": "why this trace is relevant to the hypothesis"
      }}
    ]
  }},
  "after_change": {{
    "pattern_present": true,
    "signal": "What pattern you observe in this period (1-2 sentences with data)",
    "example_traces": [
      {{
        "id": "trace_id",
        "input_preview": "first 80 chars of input",
        "output_preview": "first 120 chars of output",
        "relevance": "why this trace is relevant to the hypothesis"
      }}
    ]
  }}
}}

Note: For "too_broad" verdict, before_change and after_change can be null or empty objects. Focus on providing rich next_steps instead."""

    if openai_key:
        print(f"[TraceIQ] Calling OpenAI (gpt-4o) for hypothesis analysis...", file=sys.stderr)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=3000,
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as e:
            print(f"[TraceIQ] OpenAI error: {e}", file=sys.stderr)
            if not anthropic_key:
                raise

    # Anthropic fallback
    print(f"[TraceIQ] Calling Anthropic (claude-3-5-sonnet) for hypothesis analysis...", file=sys.stderr)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt_text + "\n\nRespond with ONLY the JSON object, no other text."}],
        )
        raw = message.content[0].text
        # Strip any markdown code blocks
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"[TraceIQ] Anthropic error: {e}", file=sys.stderr)
        raise


def run_hypothesis_mode(
    api_key: str,
    project: str,
    hypothesis: str,
    days: int = 30,
    max_traces: int = 30,
) -> dict:
    """
    Full hypothesis testing pipeline:
    1. Fetch traces
    2. Detect prompt change date
    3. Split into before/after
    4. Sample up to max_traces
    5. LLM analysis
    6. Write hypothesis_output.json
    """
    print(f"[TraceIQ] Hypothesis mode: '{hypothesis}'", file=sys.stderr)

    # Fetch traces
    runs = fetch_runs(api_key, project, days=days)
    if not runs:
        print(f"[TraceIQ] No runs found for project '{project}'", file=sys.stderr)
        sys.exit(1)

    print(f"[TraceIQ] Fetched {len(runs)} total runs", file=sys.stderr)

    # Try to get prompt change date from LangSmith commits API
    prompt_change_dt = None
    prompt_change_date = None

    commits = fetch_prompt_commits(api_key, project)
    if commits and len(commits) >= 2:
        # Most recent commit date
        last_commit = commits[-1]
        created_at = last_commit.get("created_at", "")
        if created_at:
            try:
                prompt_change_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if prompt_change_dt.tzinfo is None:
                    prompt_change_dt = prompt_change_dt.replace(tzinfo=timezone.utc)
                prompt_change_date = prompt_change_dt.strftime("%Y-%m-%d")
                print(f"[TraceIQ] Prompt change from commits API: {prompt_change_date}", file=sys.stderr)
            except ValueError:
                pass

    # Fall back to hash-based detection from runs
    if not prompt_change_dt:
        prompt_change_dt = detect_prompt_change_from_runs(runs)
        if prompt_change_dt:
            prompt_change_date = prompt_change_dt.strftime("%Y-%m-%d")
            print(f"[TraceIQ] Prompt change detected from traces: {prompt_change_date}", file=sys.stderr)

    # Sort runs by time
    sorted_runs = sorted(
        [r for r in runs if get_run_date(r)],
        key=lambda r: get_run_date(r),
    )

    # Split into before/after
    if prompt_change_dt:
        before_runs = [r for r in sorted_runs if get_run_date(r) < prompt_change_dt]
        after_runs = [r for r in sorted_runs if get_run_date(r) >= prompt_change_dt]
    else:
        # No change detected — split at midpoint
        mid = len(sorted_runs) // 2
        before_runs = sorted_runs[:mid]
        after_runs = sorted_runs[mid:]
        print(f"[TraceIQ] No prompt change detected, splitting at temporal midpoint", file=sys.stderr)

    print(f"[TraceIQ] Before: {len(before_runs)} traces, After: {len(after_runs)} traces", file=sys.stderr)

    # Sample up to max_traces/2 from each period (balanced sample)
    per_period = max_traces // 2

    def sample_period(period_runs: list[dict], n: int) -> list[dict]:
        """Sample n traces from a period, preferring recent and diverse ones."""
        if len(period_runs) <= n:
            return period_runs
        # Take most recent
        return sorted(period_runs, key=lambda r: get_run_date(r) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:n]

    before_sample = sample_period(before_runs, per_period)
    after_sample = sample_period(after_runs, per_period)

    total_analyzed = len(before_sample) + len(after_sample)
    print(f"[TraceIQ] Analyzing {len(before_sample)} before + {len(after_sample)} after = {total_analyzed} traces", file=sys.stderr)

    # LLM analysis
    llm_result = analyze_hypothesis_with_llm(
        hypothesis=hypothesis,
        before_traces=before_sample,
        after_traces=after_sample,
        prompt_change_date=prompt_change_date,
    )

    # Build output
    output = {
        "hypothesis": hypothesis,
        "verdict": llm_result.get("verdict", "inconclusive"),
        "confidence": llm_result.get("confidence", "low"),
        "summary": llm_result.get("summary", ""),
        "next_steps": llm_result.get("next_steps", []),
        "data_gaps": llm_result.get("data_gaps", ""),
        "before_change": llm_result.get("before_change", {"pattern_present": False, "example_traces": [], "signal": ""}),
        "after_change": llm_result.get("after_change", {"pattern_present": False, "example_traces": [], "signal": ""}),
        "prompt_change_date": prompt_change_date,
        "traces_analyzed": total_analyzed,
        "project": project,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to file
    output_path = Path(__file__).parent / "hypothesis_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[TraceIQ] Results written to hypothesis_output.json", file=sys.stderr)

    return output


# ─── LLM synthesis (insights mode) ────────────────────────────────────────

def synthesize_with_llm(sampled_runs: list[dict], project: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        print("[TraceIQ] Error: openai package not installed.", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[TraceIQ] Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    n = len(sampled_runs)
    trace_dicts = [trace_to_llm_dict(r) for r in sampled_runs]
    traces_json = json.dumps(trace_dicts, indent=2)

    analysis_prompt = f"""You are analyzing AI agent traces for a product team. Here are {n} traces from LangSmith project "{project}".

TRACES:
{traces_json}

Identify: working patterns, broken/struggling areas, recent changes.

Format your response as valid JSON:
{{
  "headline": "One sentence overall state",
  "working": [{{"summary": "...", "trace_ids": [], "since": ""}}],
  "broken": [{{"summary": "...", "severity": "high|medium|low", "trace_ids": [], "since": ""}}],
  "changed": [{{"summary": "...", "trace_ids": [], "direction": "better|worse", "since": ""}}]
}}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=4000,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"headline": "Unable to parse LLM response", "working": [], "broken": [], "changed": []}


# ─── Snapshot storage ─────────────────────────────────────────────────────

def get_snapshots_dir() -> Path:
    d = Path(__file__).parent / "snapshots"
    d.mkdir(exist_ok=True)
    return d


def save_snapshot(output: dict, project: str) -> Path:
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
    if not prev_snapshot:
        return current_insights
    prev_gen = prev_snapshot.get("generated_at", "")
    prev_broken = {i.get("summary", "")[:60] for i in prev_snapshot.get("insights", {}).get("broken", [])}
    prev_working = {i.get("summary", "")[:60] for i in prev_snapshot.get("insights", {}).get("working", [])}

    def label(summary: str, prev_set: set) -> str | None:
        for p in prev_set:
            if p and (p[:40] in summary or summary[:40] in p):
                return prev_gen[:10] if prev_gen else None
        return None

    for item in current_insights.get("broken", []):
        match = label(item.get("summary", ""), prev_broken)
        if match:
            item["since"] = f"since {match} (persisting)"
    for item in current_insights.get("working", []):
        match = label(item.get("summary", ""), prev_working)
        if match:
            item["since"] = f"since {match}"
    return current_insights


# ─── Insights mode ────────────────────────────────────────────────────────

def run_insights_mode(runs: list[dict], project: str, since_ts: datetime | None) -> dict:
    sampled = smart_sample_runs(runs, target=25)
    llm_result = synthesize_with_llm(sampled, project)

    traces_map: dict[str, dict] = {}
    for run in sampled:
        rid = run.get("id", "")
        if not rid:
            continue
        run_date = get_run_date(run)
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

    prev_snapshot = None
    compared_to = None
    if since_ts:
        prev_snapshot = find_previous_snapshot(project, since_ts)
        if prev_snapshot:
            compared_to = prev_snapshot.get("snapshot_id")

    insights_annotated = compute_since_labels(
        {"working": llm_result.get("working", []), "broken": llm_result.get("broken", []), "changed": llm_result.get("changed", [])},
        prev_snapshot,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "headline": llm_result.get("headline", ""),
        "trace_count_analyzed": len(sampled),
        "insights": insights_annotated,
        "traces": traces_map,
        "snapshot_id": str(uuid.uuid4()),
        "compared_to": compared_to,
    }


# ─── Legacy metrics mode ──────────────────────────────────────────────────

def analyze_runs(runs: list[dict], days: int) -> dict[str, Any]:
    if not runs:
        return {"error": "No runs found"}

    now = datetime.now(timezone.utc)
    period_midpoint = now - timedelta(days=days // 2)

    current_runs = []
    previous_runs = []
    for run in runs:
        run_date = get_run_date(run)
        if run_date:
            (current_runs if run_date >= period_midpoint else previous_runs).append(run)

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
                prompt_changes.append({"date": date, "from_hash": prev_hash, "to_hash": current_hash})
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
            model_changes.append({"date": date, "from_model": prev_model, "to_model": current_model})
        if current_model:
            prev_model = current_model

    error_messages = [extract_error(r) for r in error_runs]
    error_patterns = Counter(e for e in error_messages if e).most_common(5)

    flags = []
    if latency_stats["previous"]["p95"] > 0:
        p95_change = ((latency_stats["current"]["p95"] - latency_stats["previous"]["p95"]) / latency_stats["previous"]["p95"] * 100)
        if p95_change > 20:
            flags.append({"type": "latency_regression", "message": f"p95 increased {p95_change:.0f}%"})

    current_errors = len([r for r in current_runs if r.get("status") == "error" or r.get("error")])
    previous_errors = len([r for r in previous_runs if r.get("status") == "error" or r.get("error")])
    current_error_rate = (current_errors / len(current_runs) * 100) if current_runs else 0
    previous_error_rate = (previous_errors / len(previous_runs) * 100) if previous_runs else 0
    if current_error_rate - previous_error_rate > 5:
        flags.append({"type": "error_rate_increase", "message": f"Error rate up {current_error_rate - previous_error_rate:.1f}pp"})

    return {
        "volume": {"total_runs": total_runs, "errors": error_count, "error_rate": error_rate, "success": total_runs - error_count},
        "latency": latency_stats,
        "cost": {"total_tokens": total_tokens, "estimated_cost": estimated_cost, "cost_per_run": cost_per_run, "primary_model": primary_model},
        "prompt_changes": prompt_changes,
        "model_changes": model_changes,
        "error_patterns": error_patterns,
        "flags": flags,
    }


def generate_json_report(analysis: dict, project: str, days: int) -> str:
    now = datetime.now(timezone.utc)
    alerts = []
    for flag in analysis.get("flags", []):
        severity = "critical" if flag["type"] in ("latency_regression", "error_rate_increase") else "warning"
        alerts.append({"severity": severity, "type": flag["type"], "message": flag["message"]})

    lat = analysis.get("latency", {})
    current_lat = lat.get("current", {})
    previous_lat = lat.get("previous", {})
    p95_change_pct = 0.0
    if previous_lat.get("p95", 0) > 0:
        p95_change_pct = ((current_lat.get("p95", 0) - previous_lat.get("p95", 0)) / previous_lat["p95"]) * 100

    cost_data = analysis.get("cost", {})
    vol = analysis.get("volume", {})

    return json.dumps({
        "meta": {"project": project, "window_days": days, "generated_at": now.isoformat()},
        "alerts": alerts,
        "latency": {**current_lat, "prev_p50": previous_lat.get("p50", 0), "prev_p95": previous_lat.get("p95", 0), "prev_p99": previous_lat.get("p99", 0), "p95_change_pct": p95_change_pct},
        "cost": {"total_tokens": cost_data.get("total_tokens", 0), "estimated_usd": cost_data.get("estimated_cost", 0), "per_run_usd": cost_data.get("cost_per_run", 0), "primary_model": cost_data.get("primary_model", "unknown")},
        "volume": {"total": vol.get("total_runs", 0), "errors": vol.get("errors", 0), "error_rate_pct": vol.get("error_rate", 0)},
        "changes": {"prompts": analysis.get("prompt_changes", []), "models": analysis.get("model_changes", [])},
        "errors": {"top_patterns": [{"message": e, "count": c} for e, c in analysis.get("error_patterns", [])]},
    }, indent=2, default=str)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    global MODEL_COSTS

    parser = argparse.ArgumentParser(
        description="TraceIQ - LangSmith Agent Trace Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", help="LangSmith API key (or set LANGSMITH_API_KEY env var)")
    parser.add_argument("--project", required=True, help="LangSmith project name")
    parser.add_argument("--days", type=int, default=30, help="Days of traces to analyze (default: 30)")
    parser.add_argument("--hypothesis", type=str, default=None, help="Hypothesis to test (enables hypothesis mode)")
    parser.add_argument("--mode", choices=["insights", "metrics"], default="insights", help="Analysis mode (used when --hypothesis is not set)")
    parser.add_argument("--output", choices=["markdown", "json"], default="json", help="Output format for metrics mode")
    parser.add_argument("--since", type=str, default=None, help="ISO timestamp for diff mode")
    parser.add_argument("--mock-data", type=str, help="Path to mock data JSON (for testing)")

    args = parser.parse_args()
    MODEL_COSTS = load_cost_model()

    # Load runs
    if args.mock_data:
        with open(args.mock_data) as f:
            runs = json.load(f)
        api_key = "mock"
    else:
        api_key = get_api_key(args.api_key)
        runs = fetch_runs(api_key, args.project, args.days)

    if not runs:
        print("No runs found for the specified project and time range.")
        sys.exit(1)

    print(f"[TraceIQ] Fetched {len(runs)} total runs", file=sys.stderr)

    # Hypothesis mode
    if args.hypothesis:
        output = run_hypothesis_mode(
            api_key=api_key,
            project=args.project,
            hypothesis=args.hypothesis,
            days=args.days,
            max_traces=30,
        )
        print(json.dumps(output, indent=2, default=str))
        return

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
        save_snapshot(output, args.project)
        report = json.dumps(output, indent=2, default=str)
        insights_path = Path(__file__).parent / "insights_output.json"
        with open(insights_path, "w") as f:
            f.write(report)
        print(f"[TraceIQ] Insights written to insights_output.json", file=sys.stderr)
    else:
        analysis = analyze_runs(runs, args.days)
        report = generate_json_report(analysis, args.project, args.days)

    print(report)


if __name__ == "__main__":
    main()
