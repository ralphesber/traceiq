"""
TraceIQ Deep Analysis Agent — agentic hypothesis investigation using deepagents SDK.

The agent uses 5 analysis tools to iteratively investigate a hypothesis about
LangSmith traces, rather than sending all trace data in a single LLM call.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


SYSTEM_PROMPT = """You are TraceIQ, an expert AI agent that analyzes LangSmith trace data to investigate hypotheses.

You have access to 5 analysis tools:
- **query_traces**: Filter traces by field name + value (keyword search in inputs/outputs)
- **sample_traces**: Get representative traces matching natural-language criteria
- **classify_traces**: Use AI to classify ALL traces into categories (batched, uses claude-haiku)
- **compute_stats**: Compute statistics (avg_score, avg_latency, error_rate, count, score_distribution) for a group of trace IDs
- **compare_groups**: Compare two groups of traces side-by-side across metrics

## Your Investigation Process

1. **Understand the hypothesis**: Break it down into what you need to measure
2. **Plan your investigation**: Decide which tools to use and in what order
3. **Execute iteratively**: Use tools, examine results, refine your approach
4. **Draw conclusions**: Based on evidence from multiple tool calls

## Investigation Strategy

For most hypotheses you should:
1. Start with `classify_traces` to segment traces into relevant groups
2. Use `compute_stats` on each group to measure key metrics
3. Use `compare_groups` to quantify differences between groups
4. Use `sample_traces` or `query_traces` to find specific examples

## Output Format

After your investigation, you MUST end your response with a JSON block (in a ```json code fence) matching this schema:

```json
{
  "verdict": "supported|not_supported|inconclusive|needs_more_data|too_broad",
  "confidence": "high|medium|low",
  "summary": "2-4 sentence plain language explanation of what you found, with specific numbers and evidence.",
  "next_steps": [
    "Specific actionable follow-up 1",
    "Specific actionable follow-up 2",
    "Specific actionable follow-up 3"
  ],
  "data_gaps": "What is missing that would make this more conclusive",
  "evidence": {
    "pattern_present": true,
    "signal": "1-2 sentence description of the pattern observed with specific data",
    "example_traces": [
      {
        "id": "trace_id",
        "input_preview": "first 80 chars of input",
        "output_preview": "first 80 chars of output",
        "relevance": "why this trace is relevant"
      }
    ]
  }
}
```

Be specific: cite actual numbers from tool outputs. Reference trace IDs and statistics.
"""


def run_hypothesis_agent(
    api_key: str,           # LangSmith key
    project: str,
    hypothesis: str,
    days: int = 30,
    max_traces: int = 100,
) -> dict:
    """Run agentic hypothesis analysis using the deepagents SDK.

    Returns a result dict compatible with the existing TraceIQ output schema.
    """
    # Graceful import — return error if deepagents not installed
    try:
        from deepagents import create_deep_agent
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        return {
            "error": f"deepagents not installed: {e}. Run: pip3 install deepagents langchain-anthropic",
            "verdict": "error",
            "hypothesis": hypothesis,
            "project": project,
            "split_mode": "agent",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Add the traceiq parent directory to path so we can import traceiq.py
    traceiq_dir = Path(__file__).parent.parent
    if str(traceiq_dir) not in sys.path:
        sys.path.insert(0, str(traceiq_dir))

    try:
        from traceiq import fetch_runs, resolve_session_id
    except ImportError as e:
        return {
            "error": f"Could not import traceiq: {e}",
            "verdict": "error",
            "hypothesis": hypothesis,
            "project": project,
            "split_mode": "agent",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    from agent.tools import make_tools

    # Fetch runs
    print(f"[TraceIQ] Fetching traces for '{project}' (last {days} days)...", file=sys.stderr, flush=True)
    all_runs = fetch_runs(api_key, project, days=days)

    if not all_runs:
        return {
            "error": f"No traces found for project '{project}' in the last {days} days",
            "verdict": "needs_more_data",
            "hypothesis": hypothesis,
            "project": project,
            "traces_analyzed": 0,
            "split_mode": "agent",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Limit to max_traces (most recent)
    if len(all_runs) > max_traces:
        all_runs = sorted(all_runs, key=lambda r: r.get("start_time", ""), reverse=True)[:max_traces]
        print(f"[TraceIQ] Capped to {max_traces} most recent traces", file=sys.stderr, flush=True)

    traces_count = len(all_runs)
    print(f"[TraceIQ] Running deep analysis on {traces_count} traces...", file=sys.stderr, flush=True)

    # Resolve session_id (needed for tool context, though tools use all_runs directly)
    try:
        session_id = resolve_session_id(api_key, project)
    except Exception:
        session_id = "unknown"

    # Create tools
    tools = make_tools(api_key, session_id, all_runs)

    # Create agent model
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "error": "ANTHROPIC_API_KEY not set",
            "verdict": "error",
            "hypothesis": hypothesis,
            "project": project,
            "split_mode": "agent",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    model = ChatAnthropic(
        model_name="claude-sonnet-4-6",
        api_key=anthropic_key,
        max_tokens=8000,
    )

    # Create deep agent
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # Build the user message
    user_message = f"""Investigate this hypothesis about LangSmith project "{project}":

HYPOTHESIS: "{hypothesis}"

You have {traces_count} traces loaded (last {days} days).

Investigate systematically using the available tools. Start with classify_traces to segment traces, then compute_stats and compare_groups to quantify differences. Use sample_traces or query_traces to find specific examples.

End your response with a JSON verdict block as specified in your instructions."""

    print(f"[TraceIQ] Plan: (1) understand hypothesis → (2) query/classify traces → (3) compute stats → (4) compare groups → (5) synthesize verdict", file=sys.stderr, flush=True)
    print(f"[TraceIQ] Starting agent investigation (hypothesis: '{hypothesis[:80]}')...", file=sys.stderr, flush=True)

    try:
        result = None
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
        ):
            if "agent" in chunk:
                msgs = chunk["agent"].get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", "") or ""
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                    if content and len(content) > 10:
                        preview = content[:80].replace("\n", " ")
                        print(f"[TraceIQ] Agent: {preview}...", file=sys.stderr, flush=True)
            result = chunk
    except Exception as e:
        return {
            "error": f"Agent execution failed: {e}",
            "verdict": "error",
            "hypothesis": hypothesis,
            "project": project,
            "traces_analyzed": traces_count,
            "split_mode": "agent",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    if result is None:
        return {"error": "Agent produced no output", "verdict": "error", "hypothesis": hypothesis, "project": project, "traces_analyzed": traces_count, "split_mode": "agent", "generated_at": datetime.now(timezone.utc).isoformat()}
    if "agent" in result:
        result = result["agent"]

    # Extract the final message content
    messages = result.get("messages", [])
    final_content = ""
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if isinstance(content, list):
            # Handle list of content blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content = block.get("text", "")
                    break
                elif isinstance(block, str):
                    content = block
                    break

        if content and isinstance(content, str) and len(content) > 50:
            final_content = content
            break

    print(f"[TraceIQ] Agent finished. Extracting verdict...", file=sys.stderr, flush=True)

    # Parse the JSON verdict from the final message
    parsed = _extract_json_verdict(final_content)

    # Build output matching the existing schema
    output = {
        "hypothesis": hypothesis,
        "verdict": parsed.get("verdict", "inconclusive"),
        "confidence": parsed.get("confidence", "low"),
        "summary": parsed.get("summary", final_content[:500] if final_content else "No summary available"),
        "next_steps": parsed.get("next_steps", []),
        "data_gaps": parsed.get("data_gaps", ""),
        "evidence": parsed.get("evidence", {
            "pattern_present": False,
            "signal": "",
            "example_traces": [],
        }),
        # Compatibility fields
        "before_change": None,
        "after_change": None,
        "prompt_change_date": None,
        "split_mode": "agent",
        "split_method": "agent",
        "traces_analyzed": traces_count,
        "project": project,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_reasoning": final_content[:2000] if final_content else "",
    }

    return output


def _extract_json_verdict(text: str) -> dict:
    """Extract a JSON verdict block from the agent's final message."""
    if not text:
        return {}

    # Try to find a ```json code block
    import re

    # Look for ```json ... ``` block
    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the last { ... } block that looks like our schema
    # Find all JSON-like blocks
    brace_starts = [i for i, c in enumerate(text) if c == '{']
    for start in reversed(brace_starts):
        # Find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        # Check if it has verdict field
                        if "verdict" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    return {}
