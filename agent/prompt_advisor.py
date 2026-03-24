"""
TraceIQ Prompt Advisor — agentic analysis of experiment results to recommend prompt improvements.

Uses the deepagents SDK with experiment tools to analyze LangSmith experiment data
and generate actionable prompt improvement recommendations.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


SYSTEM_PROMPT = """You are a prompt engineering advisor for AI grading/evaluation systems.

You have tools to fetch LangSmith experiment results:
- **list_datasets**: List all datasets
- **list_experiments**: List experiments for a dataset
- **fetch_experiment_rows**: Get all rows in an experiment with scores and previews
- **get_failing_rows**: Get full inputs/outputs for rows where a metric is below threshold
- **compare_experiments**: Compare two experiments side-by-side

## How to Plan Your Investigation

**Read the user's question first.** Use it to decide which tools to call and in what order. Optimize your investigation for exactly what they're asking — don't follow a fixed script.

Examples:
- "Why is marks_exact_match low?" → go straight to `get_failing_rows(metric="marks_exact_match")`, read the patterns, explain the root cause
- "Compare the last two experiments" → use `compare_experiments`, highlight the deltas
- "Which question types score worst?" → `fetch_experiment_rows` to get scores, then look for patterns across question types in the data
- "What should I improve?" → `fetch_experiment_rows` to identify weakest metrics, then `get_failing_rows` for each to find patterns

**Use only the tools you need.** If the question is specific, skip broad overview steps. If it's open-ended, start with `fetch_experiment_rows` to find the signal, then drill in.

**Be evidence-driven.** Every finding must cite actual data from the tool output — real metric values, real input/output excerpts, real counts.

## CRITICAL Output Requirement

You MUST end your FINAL response with a JSON block in a ```json code fence. NO markdown tables, NO prose after the JSON block. The JSON block MUST be the very last thing in your response.

The JSON MUST match this schema exactly:

```json
{
  "experiment_name": "name of the experiment",
  "dataset_name": "name of the dataset",
  "overall_scores": {"metric_name": 0.85},
  "weakest_metrics": ["metric1", "metric2"],
  "failure_patterns": [
    {
      "pattern": "short description of failure pattern",
      "affected_metric": "metric_name",
      "frequency": "X of Y rows",
      "example_input": "actual input excerpt (max 300 chars)",
      "example_output": "actual output excerpt (max 300 chars)",
      "root_cause": "what in the current prompt causes this"
    }
  ],
  "recommendations": [
    {
      "priority": 1,
      "change": "specific, concrete prompt change to make",
      "rationale": "why this change will fix the identified pattern",
      "expected_impact": "which metric this improves and by roughly how much"
    }
  ],
  "generated_at": "ISO timestamp"
}
```

IMPORTANT: 
- The ```json block must appear LAST in your response
- overall_scores must be a flat dict of metric_name -> average float (0.0 to 1.0)
- weakest_metrics is a list of the 2-3 metric names with the lowest average scores
- Include at least 2-5 failure_patterns and 3-5 recommendations
- Be specific: cite real metric values, real score numbers, real examples from tool output
"""


def run_prompt_advisor(
    api_key: str,           # LangSmith key
    dataset_id: str,
    experiment_id: str,
    question: str = "",     # User's free-text question / focus area
) -> dict:
    """Run agentic prompt advisor analysis using the deepagents SDK.

    Returns a result dict with experiment_name, dataset_name, overall_scores,
    weakest_metrics, failure_patterns, recommendations, generated_at.
    """
    try:
        from deepagents import create_deep_agent
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        return {
            "error": f"deepagents not installed: {e}. Run: pip3 install deepagents langchain-anthropic",
            "experiment_id": experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    from agent.experiment_tools import make_experiment_tools

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "error": "ANTHROPIC_API_KEY not set",
            "experiment_id": experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Fetch experiment + dataset metadata for context
    experiment_name = experiment_id
    dataset_name = dataset_id
    try:
        from agent.experiment_tools import _get
        # Fetch dataset name
        try:
            ds_data = _get(api_key, f"/datasets/{dataset_id}")
            dataset_name = ds_data.get("name", dataset_id)
        except Exception:
            pass

        # Fetch experiment name
        try:
            sessions_data = _get(api_key, "/sessions", params={"reference_dataset_id": dataset_id})
            sessions = sessions_data if isinstance(sessions_data, list) else sessions_data.get("sessions", [])
            for s in sessions:
                if s.get("id") == experiment_id:
                    experiment_name = s.get("name", experiment_id)
                    break
        except Exception:
            pass
    except Exception:
        pass

    print(f"[TraceIQ] Analyzing experiment '{experiment_name}' on dataset '{dataset_name}'...", file=sys.stderr, flush=True)

    tools = make_experiment_tools(api_key)

    model = ChatAnthropic(
        model_name="claude-sonnet-4-6",
        api_key=anthropic_key,
        max_tokens=8000,
        timeout=120,  # 2 min per API call — prevents silent hangs
        max_retries=1,
    )

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # Build user message
    question_line = f'"{question.strip()}"' if question.strip() else "What prompt improvements would increase eval scores?"

    user_message = f"""Experiment: "{experiment_name}" (id: {experiment_id})
Dataset: "{dataset_name}" (id: {dataset_id})

User question: {question_line}

Use your tools to investigate this question. Plan your own approach based on what's being asked — use only the tools you need, in the order that makes sense for this specific question.

End your response with a JSON block matching the required schema."""

    print(f"[TraceIQ] Starting agent investigation (question: '{question[:80]}')...", file=sys.stderr, flush=True)

    try:
        # Use stream() instead of invoke() so intermediate steps keep the SSE connection alive
        result = None
        step_count = 0
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
        ):
            step_count += 1
            # Each chunk is a step — emit a heartbeat so the connection stays alive
            if "agent" in chunk:
                msgs = chunk["agent"].get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", "") or ""
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                    if content and len(content) > 10:
                        preview = content[:80].replace("\n", " ")
                        print(f"[TraceIQ] Agent: {preview}...", file=sys.stderr, flush=True)
            elif "tools" in chunk:
                tool_msgs = chunk["tools"].get("messages", [])
                for tm in tool_msgs:
                    name = getattr(tm, "name", "") or ""
                    if name:
                        print(f"[TraceIQ] Tool: {name}", file=sys.stderr, flush=True)
            result = chunk  # keep last chunk
        print(f"[TraceIQ] Agent stream finished after {step_count} chunks", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[TraceIQ] Agent execution failed: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "error": f"Agent execution failed: {e}",
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "experiment_id": experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Extract final message content — stream() returns last chunk which may be nested
    if result is None:
        return {"error": "Agent produced no output", "experiment_name": experiment_name, "dataset_name": dataset_name, "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}
    # Unwrap nested structure from stream() chunks
    if "agent" in result:
        result = result["agent"]
    messages = result.get("messages", [])
    final_content = ""
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if isinstance(content, list):
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

    print(f"[TraceIQ] Agent finished. Extracting recommendations...", file=sys.stderr, flush=True)

    # Parse the JSON verdict from the final message
    parsed = _extract_json_result(final_content)

    # Build the output, merging parsed data with our known metadata
    output = {
        "experiment_name": parsed.get("experiment_name", experiment_name),
        "dataset_name": parsed.get("dataset_name", dataset_name),
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "question": question,
        "overall_scores": parsed.get("overall_scores", {}),
        "weakest_metrics": parsed.get("weakest_metrics", []),
        "failure_patterns": parsed.get("failure_patterns", []),
        "recommendations": parsed.get("recommendations", []),
        "generated_at": parsed.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "agent_reasoning": final_content[:3000] if final_content else "",
        # Mark as experiment analysis
        "result_type": "experiment_analysis",
    }

    # If parsing failed, try to extract structured data from the raw analysis text
    if not output["recommendations"] and final_content:
        output["raw_analysis"] = final_content[:5000]
        # Try to extract recommendations from markdown table rows if present
        extracted = _extract_from_markdown(final_content, experiment_name, dataset_name)
        if extracted.get("recommendations"):
            output["recommendations"] = extracted["recommendations"]
            output["failure_patterns"] = extracted.get("failure_patterns", [])
            if extracted.get("overall_scores"):
                output["overall_scores"] = extracted["overall_scores"]
            if extracted.get("weakest_metrics"):
                output["weakest_metrics"] = extracted["weakest_metrics"]
        else:
            output["error"] = "Could not extract structured recommendations — see raw_analysis"

    return output


def _extract_from_markdown(text: str, experiment_name: str, dataset_name: str) -> dict:
    """Fallback: extract structured data from markdown-formatted agent output."""
    import re

    result = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "overall_scores": {},
        "weakest_metrics": [],
        "failure_patterns": [],
        "recommendations": [],
    }

    # Extract recommendations from markdown table (| Priority | Change | ... |)
    table_rows = re.findall(r'\|\s*\*\*?(\d+)\*\*?\s*\|(.+?)\|(.+?)\|', text)
    for row in table_rows:
        try:
            priority = int(row[0].strip())
            change = row[1].strip().strip('*').strip()
            impact = row[2].strip()
            if change and len(change) > 10:
                result["recommendations"].append({
                    "priority": priority,
                    "change": change,
                    "rationale": "",
                    "expected_impact": impact,
                })
        except Exception:
            pass

    # Also look for numbered list pattern: 1. **Change** or 1. Change
    if not result["recommendations"]:
        numbered = re.findall(r'(?:^|\n)\s*(\d+)\.\s+(?:\*\*)?(.+?)(?:\*\*)?\s*[-–]\s*(.+?)(?=\n|$)', text)
        for match in numbered:
            priority = int(match[0].strip())
            change = match[1].strip()
            impact = match[2].strip()
            if change and len(change) > 5:
                result["recommendations"].append({
                    "priority": priority,
                    "change": change,
                    "rationale": "",
                    "expected_impact": impact,
                })

    # Extract metric averages from backtick patterns like `metric_name` averages **-0.70**
    metric_matches = re.findall(r'`([a-z_]+)`.*?averages?\s+\*\*([+-]?\d+\.?\d*)\*\*', text)
    for metric, val in metric_matches:
        try:
            result["overall_scores"][metric] = float(val)
        except ValueError:
            pass

    return result


def _extract_json_result(text: str) -> dict:
    """Extract a JSON result block from the agent's final message."""
    if not text:
        return {}

    import re

    # Try ```json ... ``` block first
    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the last { ... } block that has our schema keys
    brace_starts = [i for i, c in enumerate(text) if c == '{']
    for start in reversed(brace_starts):
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
                        if "recommendations" in parsed or "overall_scores" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    return {}
