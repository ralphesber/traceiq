"""
TraceIQ agent tools — analysis tools for the hypothesis deep-analysis agent.

All tools operate on pre-fetched `all_runs` (bound via closure) so no additional
API calls are needed for basic filtering/sampling/stats.  The `classify_traces`
tool does make LLM calls (claude-haiku-4-5) in batches of 20.
"""

import json
import math
import os
import re
import statistics
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_summary(run: dict) -> dict:
    """Return a compact summary dict for a run."""
    inputs = run.get("inputs", {}) or {}
    outputs = run.get("outputs", {}) or {}

    def _text(d: Any, max_len: int = 200) -> str:
        if isinstance(d, str):
            return d[:max_len]
        if isinstance(d, dict):
            for key in ("input", "question", "query", "student_response", "response", "text", "content", "output", "answer", "result", "feedback"):
                val = d.get(key)
                if val and isinstance(val, str):
                    return val[:max_len]
            # Try messages list
            msgs = d.get("messages", [])
            if isinstance(msgs, list):
                for msg in reversed(msgs):
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if content and isinstance(content, str):
                            return content[:max_len]
            # Fallback: first string value
            for val in d.values():
                if isinstance(val, str) and len(val) > 10:
                    return val[:max_len]
        return str(d)[:max_len]

    start = run.get("start_time")
    end = run.get("end_time")
    latency = None
    if start and end:
        try:
            from datetime import datetime
            if isinstance(start, str):
                start = datetime.fromisoformat(start.replace("Z", "+00:00"))
            if isinstance(end, str):
                end = datetime.fromisoformat(end.replace("Z", "+00:00"))
            latency = round((end - start).total_seconds(), 2)
        except Exception:
            pass

    return {
        "id": run.get("id", "")[:16],
        "input_preview": _text(inputs),
        "output_preview": _text(outputs),
        "status": "error" if (run.get("status") == "error" or run.get("error")) else "success",
        "latency_s": latency,
    }


def _extract_scores(run: dict) -> list[float]:
    """Try to extract numeric score values from a run's outputs."""
    outputs = run.get("outputs", {}) or {}
    scores = []

    def _dig(obj: Any):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if 0 <= obj <= 100:
                scores.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _dig(v)
        elif isinstance(obj, list):
            for item in obj:
                _dig(item)
        elif isinstance(obj, str):
            # Look for patterns like "score: 7" or "7/10"
            for m in re.findall(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b', obj):
                scores.append(float(m))
            for m in re.findall(r'(?:score|mark|grade)[:\s]+(\d+(?:\.\d+)?)', obj, re.I):
                val = float(m)
                if 0 <= val <= 100:
                    scores.append(val)

    _dig(outputs)
    return scores


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def make_tools(api_key: str, session_id: str, all_runs: list[dict]) -> list:
    """Create tool functions with LangSmith context bound via closure.

    Returns a list of langchain @tool-decorated functions.
    """
    from langchain_core.tools import tool

    # Build a quick lookup by ID
    runs_by_id: dict[str, dict] = {r.get("id", ""): r for r in all_runs if r.get("id")}

    # -----------------------------------------------------------------------
    # Tool 1: query_traces
    # -----------------------------------------------------------------------

    @tool
    def query_traces(filter_field: str, filter_value: str, limit: int = 20) -> str:
        """Filter traces by checking if a field name + value appear in their inputs/outputs JSON.

        Args:
            filter_field: The field name or keyword to search for (e.g. "question_type", "math").
            filter_value: The value to match against (case-insensitive substring match).
            limit: Maximum number of results to return (default 20).
        """
        print(f"[TraceIQ] Querying traces where '{filter_field}' contains '{filter_value}'...", file=sys.stderr, flush=True)
        results = []
        filter_value_lower = filter_value.lower()
        filter_field_lower = filter_field.lower()

        for run in all_runs:
            # Serialize inputs+outputs as JSON for broad matching
            combined = json.dumps({
                "inputs": run.get("inputs", {}),
                "outputs": run.get("outputs", {}),
            }).lower()

            # Check both field and value appear in the combined JSON
            if filter_field_lower in combined and filter_value_lower in combined:
                results.append(_run_summary(run))
                if len(results) >= limit:
                    break

        return json.dumps(results, default=str)

    # -----------------------------------------------------------------------
    # Tool 2: sample_traces
    # -----------------------------------------------------------------------

    @tool
    def sample_traces(criteria: str, n: int = 10) -> str:
        """Return N representative traces matching a natural-language criteria using keyword matching.

        Args:
            criteria: Natural-language description of what to look for (e.g. "traces where the student got a low score").
            n: Number of traces to return (default 10).
        """
        print(f"[TraceIQ] Sampling up to {n} traces matching: '{criteria}'...", file=sys.stderr, flush=True)
        # Extract keywords from criteria (words longer than 3 chars)
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', criteria)]

        scored = []
        for run in all_runs:
            combined = json.dumps({
                "inputs": run.get("inputs", {}),
                "outputs": run.get("outputs", {}),
            }).lower()
            hits = sum(1 for kw in keywords if kw in combined)
            if hits > 0:
                scored.append((hits, run))

        # Sort by relevance score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [_run_summary(run) for _, run in scored[:n]]

        # If not enough matches, pad with random recent traces
        if len(results) < n:
            seen_ids = {r["id"] for r in results}
            for run in sorted(all_runs, key=lambda r: r.get("start_time", ""), reverse=True):
                if run.get("id", "")[:16] not in seen_ids:
                    results.append(_run_summary(run))
                    seen_ids.add(run.get("id", "")[:16])
                if len(results) >= n:
                    break

        return json.dumps(results[:n], default=str)

    # -----------------------------------------------------------------------
    # Tool 3: classify_traces
    # -----------------------------------------------------------------------

    @tool
    def classify_traces(dimension: str, categories: list[str]) -> str:
        """Classify all traces into categories using an LLM (claude-haiku-4-5) in batches of 20.

        Args:
            dimension: The dimension to classify by (e.g. "question type", "response quality", "topic area").
            categories: List of category names to classify into (e.g. ["math", "science", "history"]).
        """
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            return json.dumps({"error": "ANTHROPIC_API_KEY not set"})

        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)

        result: dict[str, list[str]] = {cat: [] for cat in categories}
        result["unknown"] = []

        batch_size = 20
        batches = [all_runs[i:i + batch_size] for i in range(0, len(all_runs), batch_size)]
        total = len(all_runs)
        processed = 0
        print(
            f"[TraceIQ] Starting trace classification: {total} traces across "
            f"{math.ceil(total / batch_size)} batches",
            file=sys.stderr,
            flush=True,
        )

        for batch in batches:
            # Build compact batch representation
            batch_data = []
            for run in batch:
                inputs = run.get("inputs", {}) or {}
                outputs = run.get("outputs", {}) or {}
                batch_data.append({
                    "id": run.get("id", "")[:16],
                    "input": str(inputs)[:300],
                    "output": str(outputs)[:300],
                })

            prompt = f"""Classify each of these AI agent traces by {dimension}.

Categories: {json.dumps(categories)}

Traces:
{json.dumps(batch_data, indent=2)}

For each trace ID, assign exactly one category from the list. If none fits, use "unknown".

Respond ONLY with a JSON object mapping trace IDs to categories:
{{"trace_id_1": "category_name", "trace_id_2": "category_name", ...}}"""

            try:
                msg = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text.strip()
                # Strip markdown code blocks if present
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:]

                classifications = json.loads(raw)
                for trace_id, category in classifications.items():
                    if category in result:
                        result[category].append(trace_id)
                    else:
                        result["unknown"].append(trace_id)

            except Exception as e:
                # On error, mark batch as unknown
                for run in batch:
                    result["unknown"].append(run.get("id", "")[:16])

            processed += len(batch)
            print(
                f"[TraceIQ] Classifying traces: {min(processed, total)}/{total} labeled",
                file=sys.stderr,
                flush=True,
            )

        # Add counts
        summary = {cat: {"count": len(ids), "trace_ids": ids} for cat, ids in result.items()}
        return json.dumps(summary, default=str)

    # -----------------------------------------------------------------------
    # Tool 4: compute_stats
    # -----------------------------------------------------------------------

    @tool
    def compute_stats(trace_ids: list[str], metrics: list[str]) -> str:
        """Compute statistics for a list of trace IDs.

        Args:
            trace_ids: List of trace IDs (from classify_traces output).
            metrics: List of metrics to compute. Supported: "avg_score", "avg_latency", "error_rate", "count", "score_distribution".
        """
        print(f"[TraceIQ] Computing stats ({', '.join(metrics)}) for {len(trace_ids)} traces...", file=sys.stderr, flush=True)
        # Match trace IDs (they may be truncated to 16 chars)
        matched_runs = []
        for run in all_runs:
            run_id = run.get("id", "")
            # Match on full ID or first 16 chars
            if run_id[:16] in trace_ids or run_id in trace_ids:
                matched_runs.append(run)

        count = len(matched_runs)
        print(f"[TraceIQ] Matched {count} traces — computing {', '.join(metrics)}...", file=sys.stderr, flush=True)
        stats: dict[str, Any] = {}

        for metric in metrics:
            if metric == "count":
                stats["count"] = count

            elif metric == "avg_score":
                all_scores = []
                for run in matched_runs:
                    all_scores.extend(_extract_scores(run))
                if all_scores:
                    stats["avg_score"] = round(statistics.mean(all_scores), 2)
                    stats["median_score"] = round(statistics.median(all_scores), 2)
                    stats["min_score"] = round(min(all_scores), 2)
                    stats["max_score"] = round(max(all_scores), 2)
                else:
                    stats["avg_score"] = None
                    stats["score_note"] = "No numeric scores found in outputs"

            elif metric == "avg_latency":
                latencies = []
                for run in matched_runs:
                    start = run.get("start_time")
                    end = run.get("end_time")
                    if start and end:
                        try:
                            from datetime import datetime
                            if isinstance(start, str):
                                start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                            if isinstance(end, str):
                                end = datetime.fromisoformat(end.replace("Z", "+00:00"))
                            latencies.append((end - start).total_seconds())
                        except Exception:
                            pass
                if latencies:
                    stats["avg_latency_s"] = round(statistics.mean(latencies), 2)
                    stats["median_latency_s"] = round(statistics.median(latencies), 2)
                    stats["p95_latency_s"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
                else:
                    stats["avg_latency_s"] = None

            elif metric == "error_rate":
                errors = sum(1 for r in matched_runs if r.get("status") == "error" or r.get("error"))
                stats["error_count"] = errors
                stats["error_rate_pct"] = round((errors / count * 100) if count > 0 else 0, 1)

            elif metric == "score_distribution":
                all_scores = []
                for run in matched_runs:
                    all_scores.extend(_extract_scores(run))
                if all_scores:
                    # Bucket into 0-3, 4-6, 7-10 ranges
                    dist = {"0-3": 0, "4-6": 0, "7-10": 0, "11+": 0}
                    for s in all_scores:
                        if s <= 3:
                            dist["0-3"] += 1
                        elif s <= 6:
                            dist["4-6"] += 1
                        elif s <= 10:
                            dist["7-10"] += 1
                        else:
                            dist["11+"] += 1
                    stats["score_distribution"] = dist
                    stats["score_count"] = len(all_scores)
                else:
                    stats["score_distribution"] = None

        stats["traces_matched"] = count
        return json.dumps(stats, default=str)

    # -----------------------------------------------------------------------
    # Tool 5: compare_groups
    # -----------------------------------------------------------------------

    @tool
    def compare_groups(group_a_ids: list[str], group_b_ids: list[str], label_a: str, label_b: str) -> str:
        """Compare two groups of traces across key metrics.

        Args:
            group_a_ids: Trace IDs for group A.
            group_b_ids: Trace IDs for group B.
            label_a: Human-readable label for group A (e.g. "Math questions").
            label_b: Human-readable label for group B (e.g. "Science questions").
        """
        print(f"[TraceIQ] Comparing '{label_a}' ({len(group_a_ids)} traces) vs '{label_b}' ({len(group_b_ids)} traces)...", file=sys.stderr, flush=True)
        all_metrics = ["count", "avg_score", "avg_latency", "error_rate", "score_distribution"]

        def _stats_for_group(trace_ids: list[str]) -> dict:
            matched = []
            for run in all_runs:
                run_id = run.get("id", "")
                if run_id[:16] in trace_ids or run_id in trace_ids:
                    matched.append(run)

            count = len(matched)
            if count == 0:
                return {"count": 0, "no_data": True}

            # Scores
            all_scores = []
            for run in matched:
                all_scores.extend(_extract_scores(run))

            # Latencies
            latencies = []
            for run in matched:
                start = run.get("start_time")
                end = run.get("end_time")
                if start and end:
                    try:
                        from datetime import datetime
                        if isinstance(start, str):
                            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        if isinstance(end, str):
                            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        latencies.append((end - start).total_seconds())
                    except Exception:
                        pass

            errors = sum(1 for r in matched if r.get("status") == "error" or r.get("error"))

            return {
                "count": count,
                "avg_score": round(statistics.mean(all_scores), 2) if all_scores else None,
                "score_count": len(all_scores),
                "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
                "error_rate_pct": round((errors / count * 100) if count > 0 else 0, 1),
            }

        stats_a = _stats_for_group(group_a_ids)
        stats_b = _stats_for_group(group_b_ids)

        # Compute diffs
        comparison = {
            "group_a": {"label": label_a, "stats": stats_a},
            "group_b": {"label": label_b, "stats": stats_b},
            "differences": {},
        }

        for metric in ("avg_score", "avg_latency_s", "error_rate_pct"):
            a_val = stats_a.get(metric)
            b_val = stats_b.get(metric)
            if a_val is not None and b_val is not None:
                diff = round(b_val - a_val, 2)
                pct = round((diff / a_val * 100) if a_val != 0 else 0, 1)
                comparison["differences"][metric] = {
                    "group_a": a_val,
                    "group_b": b_val,
                    "delta": diff,
                    "delta_pct": pct,
                    "interpretation": f"{label_b} is {'higher' if diff > 0 else 'lower'} by {abs(diff)} ({abs(pct)}%)",
                }

        # Log key finding
        diff = comparison.get("differences", {}).get("avg_score", {})
        if diff:
            print(f"[TraceIQ] Comparison done — avg score: {label_a}={diff.get('group_a')} vs {label_b}={diff.get('group_b')} (delta: {diff.get('delta')})", file=sys.stderr, flush=True)
        else:
            print(f"[TraceIQ] Comparison done — {label_a}: {stats_a.get('count')} traces, {label_b}: {stats_b.get('count')} traces", file=sys.stderr, flush=True)

        return json.dumps(comparison, default=str)

    return [query_traces, sample_traces, classify_traces, compute_stats, compare_groups]
