#!/usr/bin/env python3
"""
TraceIQ - LangSmith Agent Trace Analysis CLI
Local copy for integration testing.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

LANGSMITH_API_BASE = "https://api.smith.langchain.com"

MODEL_COSTS = {
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

def calculate_percentile(values, percentile):
    if not values: return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_values): return sorted_values[-1]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

def extract_latency(run):
    start = run.get("start_time")
    end = run.get("end_time")
    if not start or not end: return None
    if isinstance(start, str): start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str): end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end - start).total_seconds()

def extract_tokens(run):
    if "total_tokens" in run: return run["total_tokens"]
    extra = run.get("extra", {}) or {}
    if "total_tokens" in extra: return extra["total_tokens"]
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict):
        usage = outputs.get("usage", {}) or {}
        total = usage.get("total_tokens", 0)
        if total: return total
    return 0

def extract_model(run):
    extra = run.get("extra", {}) or {}
    if run.get("model"): return run["model"]
    invocation = extra.get("invocation_params", {}) or {}
    if invocation.get("model"): return invocation["model"]
    if invocation.get("model_name"): return invocation["model_name"]
    metadata = extra.get("metadata", {}) or {}
    if metadata.get("model"): return metadata["model"]
    return None

def extract_system_prompt(run):
    inputs = run.get("inputs", {}) or {}
    messages = inputs.get("messages", [])
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "") or msg.get("type", "")
                if role in ("system", "SystemMessage"):
                    return msg.get("content", "")
    return None

def hash_prompt(prompt):
    if not prompt: return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]

def get_run_date(run):
    start = run.get("start_time")
    if not start: return None
    if isinstance(start, str): return datetime.fromisoformat(start.replace("Z", "+00:00"))
    return start

def analyze_runs(runs, days):
    if not runs: return {"error": "No runs found"}
    now = datetime.now(timezone.utc)
    period_midpoint = now - timedelta(days=days // 2)
    current_runs, previous_runs = [], []
    for run in runs:
        run_date = get_run_date(run)
        if run_date:
            if run_date >= period_midpoint: current_runs.append(run)
            else: previous_runs.append(run)

    total_runs = len(runs)
    error_runs = [r for r in runs if r.get("status") == "error" or r.get("error")]
    error_rate = (len(error_runs) / total_runs * 100) if total_runs > 0 else 0

    current_latencies = [l for r in current_runs if (l := extract_latency(r)) is not None]
    previous_latencies = [l for r in previous_runs if (l := extract_latency(r)) is not None]

    latency = {
        "current": {m: calculate_percentile(current_latencies, p) for m, p in [("p50",50),("p95",95),("p99",99)]},
        "previous": {m: calculate_percentile(previous_latencies, p) for m, p in [("p50",50),("p95",95),("p99",99)]},
    }

    total_tokens = sum(extract_tokens(r) for r in runs)
    models_used = [m for r in runs if (m := extract_model(r))]
    primary_model = Counter(models_used).most_common(1)[0][0] if models_used else "default"
    cost_per_1k = MODEL_COSTS.get(primary_model, MODEL_COSTS["default"])
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    # Prompt changes
    prompts_by_date = {}
    for run in runs:
        prompt = extract_system_prompt(run)
        ph = hash_prompt(prompt)
        rd = get_run_date(run)
        if rd and ph != "none":
            ds = rd.strftime("%Y-%m-%d")
            prompts_by_date.setdefault(ds, []).append(ph)

    prompt_changes = []
    prev_hash = None
    for date in sorted(prompts_by_date.keys()):
        current_hash = Counter(prompts_by_date[date]).most_common(1)[0][0]
        if prev_hash and current_hash != prev_hash:
            prompt_changes.append({"date": date, "from": prev_hash, "to": current_hash})
        prev_hash = current_hash

    # Model changes
    models_by_date = {}
    for run in runs:
        model = extract_model(run)
        rd = get_run_date(run)
        if rd and model:
            ds = rd.strftime("%Y-%m-%d")
            models_by_date.setdefault(ds, []).append(model)

    model_changes = []
    prev_model = None
    for date in sorted(models_by_date.keys()):
        cm = Counter(models_by_date[date]).most_common(1)[0][0]
        if prev_model and cm != prev_model:
            model_changes.append({"date": date, "from": prev_model, "to": cm})
        prev_model = cm

    # Error patterns - count THEN truncate
    error_messages = [r.get("error", "Unknown error") for r in error_runs]
    error_counts = Counter(error_messages).most_common(5)
    error_patterns = [{"message": msg[:100], "count": count, "pct": round(count/len(error_runs)*100, 1)} for msg, count in error_counts] if error_runs else []

    # Flags
    flags = []
    if latency["previous"]["p95"] > 0:
        p95_change = ((latency["current"]["p95"] - latency["previous"]["p95"]) / latency["previous"]["p95"]) * 100
        if p95_change > 20:
            flags.append({"type": "latency_regression", "severity": "critical",
                "message": f'p95 latency increased {p95_change:.0f}% ({latency["previous"]["p95"]:.1f}s → {latency["current"]["p95"]:.1f}s)'})

    for change in model_changes:
        flags.append({"type": "model_change", "severity": "warning",
            "message": f'Model changed on {change["date"]}: {change["from"]} → {change["to"]}'})

    return {
        "meta": {"project": "demo-agent", "window_days": days, "generated_at": datetime.now(timezone.utc).isoformat()},
        "alerts": flags,
        "volume": {"total": total_runs, "errors": len(error_runs), "error_rate": round(error_rate, 1)},
        "latency": latency,
        "cost": {"total_tokens": total_tokens, "estimated_cost": round(estimated_cost, 2), "primary_model": primary_model},
        "prompts": {"changes": prompt_changes},
        "models": {"changes": model_changes},
        "errors": {"patterns": error_patterns},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceIQ Analysis")
    parser.add_argument("--mock-data", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    with open(args.mock_data) as f:
        runs = json.load(f)

    analysis = analyze_runs(runs, args.days)
    print(json.dumps(analysis, indent=2))
