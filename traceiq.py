#!/usr/bin/env python3
"""
TraceIQ - LangSmith Agent Trace Analysis CLI

Analyzes LangSmith traces to generate insights on latency, costs,
errors, and detect regressions or configuration changes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
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
        # Create default config file
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

    # Log missing model
    print(f"[TraceIQ] Note: model '{model}' not found in cost_model.json, using default rate", file=sys.stderr)
    return MODEL_COSTS.get("default", DEFAULT_MODEL_COSTS["default"])


def get_api_key(args_api_key: str | None) -> str:
    """Get API key from args or environment."""
    api_key = args_api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set LANGSMITH_API_KEY env var.")
        sys.exit(1)
    return api_key


def fetch_runs(
    api_key: str,
    project_name: str,
    days: int,
    page_delay: float = 0.1,
    max_retries: int = 3,
    default_retry_after: int = 60,
) -> list[dict]:
    """Fetch runs from LangSmith API with rate limiting.

    Args:
        api_key: LangSmith API key
        project_name: Name of the project to fetch runs from
        days: Number of days of history to fetch
        page_delay: Delay in seconds between paginated requests (default: 0.1)
        max_retries: Maximum retries on 429 rate limit (default: 3)
        default_retry_after: Default wait time in seconds if Retry-After header missing (default: 60)
    """
    headers = {"x-api-key": api_key}

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_runs = []
    cursor = None
    is_first_request = True

    while True:
        params = {
            "project_name": project_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor

        # Add delay between paginated requests (not before first request)
        if not is_first_request:
            time.sleep(page_delay)
        is_first_request = False

        # Retry loop for rate limiting
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(
                    f"{LANGSMITH_API_BASE}/api/v1/runs",
                    headers=headers,
                    params=params,
                    timeout=30,
                )

                # Handle rate limiting (429)
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
                break  # Success, exit retry loop

            except requests.RequestException as e:
                print(f"Error fetching runs: {e}", file=sys.stderr)
                sys.exit(1)

        data = response.json()
        runs = data.get("runs", data) if isinstance(data, dict) else data

        if not runs:
            break

        all_runs.extend(runs)

        # Check for pagination cursor
        if isinstance(data, dict) and data.get("cursors", {}).get("next"):
            cursor = data["cursors"]["next"]
        else:
            break

    return all_runs


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

    # Handle both string and datetime formats
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))

    return (end - start).total_seconds()


def extract_tokens(run: dict) -> int:
    """Extract total tokens from a run."""
    # Try various locations where token counts might be stored
    if "total_tokens" in run:
        return run["total_tokens"]

    feedback = run.get("feedback_stats", {})
    if "total_tokens" in feedback:
        return feedback["total_tokens"]

    # Check in extra metadata
    extra = run.get("extra", {}) or {}
    if "total_tokens" in extra:
        return extra["total_tokens"]

    # Check in outputs
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict):
        usage = outputs.get("usage", {}) or {}
        total = usage.get("total_tokens", 0)
        if total:
            return total

    # Sum prompt + completion tokens if available
    prompt_tokens = run.get("prompt_tokens", 0) or extra.get("prompt_tokens", 0)
    completion_tokens = run.get("completion_tokens", 0) or extra.get("completion_tokens", 0)

    total = prompt_tokens + completion_tokens

    # Log fallback warning if no token count found
    if total == 0:
        run_id = run.get("id", "unknown")
        print(f"[TraceIQ] Warning: could not extract token count for run {run_id}", file=sys.stderr)

    return total


def extract_model(run: dict) -> str | None:
    """Extract model name from a run."""
    # Check various locations
    extra = run.get("extra", {}) or {}

    # Direct model field
    if run.get("model"):
        return run["model"]

    # In extra.invocation_params
    invocation = extra.get("invocation_params", {}) or {}
    if invocation.get("model"):
        return invocation["model"]
    if invocation.get("model_name"):
        return invocation["model_name"]

    # In extra.metadata
    metadata = extra.get("metadata", {}) or {}
    if metadata.get("model"):
        return metadata["model"]
    if metadata.get("ls_model_name"):
        return metadata["ls_model_name"]

    # In serialized kwargs
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

    # Check for messages array
    messages = inputs.get("messages", [])
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "") or msg.get("type", "")
                if role in ("system", "SystemMessage"):
                    return msg.get("content", "")

    # Check for direct system prompt
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
        # Truncate long errors
        if len(error) > 100:
            return error[:100] + "..."
        return error

    # Check outputs for error
    outputs = run.get("outputs", {}) or {}
    if isinstance(outputs, dict) and outputs.get("error"):
        err = outputs["error"]
        if len(err) > 100:
            return err[:100] + "..."
        return err

    return "Unknown error"


def get_run_date(run: dict) -> datetime | None:
    """Get the date of a run."""
    start = run.get("start_time")
    if not start:
        return None
    if isinstance(start, str):
        return datetime.fromisoformat(start.replace("Z", "+00:00"))
    return start


def analyze_runs(runs: list[dict], days: int) -> dict[str, Any]:
    """Perform full analysis on runs."""
    if not runs:
        return {"error": "No runs found"}

    # Split into current and previous period for comparison
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

    # Volume & error analysis
    total_runs = len(runs)
    error_runs = [r for r in runs if r.get("status") == "error" or r.get("error")]
    error_count = len(error_runs)
    error_rate = (error_count / total_runs * 100) if total_runs > 0 else 0

    # Latency analysis
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

    # Token & cost analysis
    total_tokens = sum(extract_tokens(r) for r in runs)
    models_used = [m for r in runs if (m := extract_model(r))]
    primary_model = Counter(models_used).most_common(1)[0][0] if models_used else "default"
    cost_per_1k = get_model_cost(primary_model)
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    cost_per_run = estimated_cost / total_runs if total_runs > 0 else 0

    # Prompt change detection
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

    # Model change detection
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

    # Error patterns
    error_messages = [extract_error(r) for r in error_runs]
    error_patterns = Counter(e for e in error_messages if e).most_common(5)

    # Regression flags
    flags = []

    # P95 latency regression check (>20%)
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

    # Error rate increase check (>5pp)
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

    # Model change flag
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
    """Generate markdown report from analysis."""
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# TraceIQ Analysis Report",
        f"**Project:** {project} | **Period:** last {days} days | **Generated:** {now}",
        "",
    ]

    # Flags section
    flags = analysis.get("flags", [])
    if flags:
        lines.append("## 🔴 Flags (Attention Required)")
        for flag in flags:
            lines.append(f"- {flag['message']}")
        lines.append("")

    # Volume section
    vol = analysis["volume"]
    lines.append("## 📊 Volume")
    lines.append(
        f"- Total runs: {vol['total_runs']:,} | Errors: {vol['errors']:,} "
        f"({vol['error_rate']:.1f}%) | Success: {vol['success']:,}"
    )
    lines.append("")

    # Latency section
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

    # Cost section
    cost = analysis["cost"]
    lines.append("## 💰 Cost")
    lines.append(
        f"- Total tokens: {format_number(cost['total_tokens'])} | "
        f"Estimated cost: ${cost['estimated_cost']:.2f} | "
        f"Cost/run: ${cost['cost_per_run']:.4f}"
    )
    lines.append("")

    # Changes section
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

    # Error patterns section
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
    """Generate JSON report from analysis with standardized schema v1."""
    now = datetime.now(timezone.utc)

    # Build alerts from flags
    alerts = []
    for flag in analysis.get("flags", []):
        # Map flag type to severity
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

    # Build latency object
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

    # Build cost object
    cost_data = analysis.get("cost", {})
    cost = {
        "total_tokens": cost_data.get("total_tokens", 0),
        "estimated_usd": cost_data.get("estimated_cost", 0.0),
        "per_run_usd": cost_data.get("cost_per_run", 0.0),
        "primary_model": cost_data.get("primary_model", "unknown"),
    }

    # Build volume object with previous error rate
    vol = analysis.get("volume", {})
    # Calculate previous error rate from the analysis context
    # This requires looking at the flags for error_rate_increase info
    prev_error_rate = 0.0
    for flag in analysis.get("flags", []):
        if flag["type"] == "error_rate_increase":
            # Parse from message like "Error rate increased X.Xpp (Y.Y% → Z.Z%)"
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

    # Build changes object
    changes = {
        "prompts": analysis.get("prompt_changes", []),
        "models": analysis.get("model_changes", []),
    }

    # Build errors object with top patterns
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

    # Build final report with standardized schema
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
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--mock-data",
        type=str,
        help="Path to mock data JSON file (for testing without API)",
    )

    args = parser.parse_args()

    # Load cost model configuration
    MODEL_COSTS = load_cost_model()

    # Load runs from mock data or API
    if args.mock_data:
        with open(args.mock_data) as f:
            runs = json.load(f)
    else:
        api_key = get_api_key(args.api_key)
        runs = fetch_runs(api_key, args.project, args.days)

    if not runs:
        print("No runs found for the specified project and time range.")
        sys.exit(1)

    # Analyze
    analysis = analyze_runs(runs, args.days)

    # Generate report
    if args.output == "json":
        report = generate_json_report(analysis, args.project, args.days)
    else:
        report = generate_markdown_report(analysis, args.project, args.days)

    print(report)


if __name__ == "__main__":
    main()
