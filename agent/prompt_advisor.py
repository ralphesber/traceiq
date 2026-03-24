"""
TraceIQ Prompt Advisor — agentic analysis of experiment results to recommend prompt improvements.

Uses the Anthropic SDK directly in a simple tool-calling loop.
No deepagents/LangGraph wrappers — full control over timeouts and execution.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import anthropic


SYSTEM_PROMPT = """You are a prompt engineering advisor for AI grading/evaluation systems.

You have tools to fetch LangSmith experiment results:
- **list_datasets**: List all datasets
- **list_experiments**: List experiments for a dataset
- **fetch_experiment_rows**: Get all rows in an experiment with scores and previews
- **get_failing_rows**: Get full inputs/outputs for rows where a metric is below threshold
- **compare_experiments**: Compare two experiments side-by-side

## How to Plan Your Investigation

Read the user's question first. Use it to decide which tools to call and in what order.

Examples:
- "Why is marks_exact_match low?" → go straight to get_failing_rows, read the patterns, explain the root cause
- "What should I improve?" → fetch_experiment_rows to identify weakest metrics, then get_failing_rows for each

Be evidence-driven. Every finding must cite actual data from the tool output.

## Output Requirement

End your FINAL response with a JSON block in a ```json code fence. The JSON MUST match this schema:

```json
{
  "experiment_name": "name of the experiment",
  "dataset_name": "name of the dataset",
  "overall_scores": {"metric_name": 0.85},
  "weakest_metrics": ["metric1", "metric2"],
  "failure_patterns": [
    {
      "pattern": "short description",
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

The ```json block must be LAST in your response.
"""

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8000
API_TIMEOUT = 120  # seconds per Anthropic API call
MAX_ITERATIONS = 20  # max tool-call rounds before giving up


def run_prompt_advisor(
    api_key: str,
    dataset_id: str,
    experiment_id: str,
    question: str = "",
) -> dict:
    """Run agentic prompt advisor analysis using Anthropic SDK directly.

    Returns a result dict with experiment_name, dataset_name, overall_scores,
    weakest_metrics, failure_patterns, recommendations, generated_at.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {"error": "ANTHROPIC_API_KEY not set", "experiment_id": experiment_id,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    from agent.experiment_tools import (
        make_experiment_tools,
        _get as ls_get,
    )

    # Fetch metadata
    experiment_name = experiment_id
    dataset_name = dataset_id
    try:
        ds = ls_get(api_key, f"/datasets/{dataset_id}")
        dataset_name = ds.get("name", dataset_id)
    except Exception as e:
        print(f"[TraceIQ] Warning: could not fetch dataset name: {e}", file=sys.stderr, flush=True)
    try:
        sessions = ls_get(api_key, "/sessions", params={"reference_dataset_id": dataset_id})
        sessions = sessions if isinstance(sessions, list) else sessions.get("sessions", [])
        for s in sessions:
            if s.get("id") == experiment_id:
                experiment_name = s.get("name", experiment_id)
                break
    except Exception as e:
        print(f"[TraceIQ] Warning: could not fetch experiment name: {e}", file=sys.stderr, flush=True)

    print(f"[TraceIQ] Analyzing experiment '{experiment_name}' on dataset '{dataset_name}'...", file=sys.stderr, flush=True)

    # Build tools for Anthropic
    lc_tools = make_experiment_tools(api_key)
    anthropic_tools = _lc_tools_to_anthropic(lc_tools)
    tool_map = {t.name: t for t in lc_tools}

    question_line = f'"{question.strip()}"' if question.strip() else "What prompt improvements would increase eval scores?"

    user_message = f"""Experiment: "{experiment_name}" (id: {experiment_id})
Dataset: "{dataset_name}" (id: {dataset_id})

User question: {question_line}

Use your tools to investigate. End your response with a JSON block matching the required schema."""

    client = anthropic.Anthropic(api_key=anthropic_key, timeout=API_TIMEOUT)
    messages = [{"role": "user", "content": user_message}]

    print(f"[TraceIQ] Starting agent investigation (question: '{question[:80]}')...", file=sys.stderr, flush=True)

    final_content = ""
    for iteration in range(MAX_ITERATIONS):
        print(f"[TraceIQ] Calling Claude (iteration {iteration + 1})...", file=sys.stderr, flush=True)
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=anthropic_tools,
                messages=messages,
            )
        except anthropic.APITimeoutError:
            return {"error": f"Anthropic API timed out after {API_TIMEOUT}s",
                    "experiment_name": experiment_name, "dataset_name": dataset_name,
                    "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"error": f"Anthropic API error: {e}",
                    "experiment_name": experiment_name, "dataset_name": dataset_name,
                    "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}

        # Collect text content for final output
        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        if text_parts:
            final_content = "\n".join(text_parts)
            preview = final_content[:80].replace("\n", " ")
            print(f"[TraceIQ] Agent: {preview}...", file=sys.stderr, flush=True)

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            print(f"[TraceIQ] Agent finished after {iteration + 1} iterations.", file=sys.stderr, flush=True)
            break

        # Process tool calls
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            print(f"[TraceIQ] No tool calls, stop_reason={response.stop_reason}", file=sys.stderr, flush=True)
            break

        # Add assistant message to history
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for tu in tool_uses:
            tool_name = tu.name
            tool_input = tu.input
            print(f"[TraceIQ] Tool: {tool_name}({json.dumps(tool_input)[:80]})", file=sys.stderr, flush=True)
            try:
                lc_tool = tool_map.get(tool_name)
                if lc_tool is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    result = lc_tool.invoke(tool_input)
                    if not isinstance(result, str):
                        result = json.dumps(result, default=str)
            except Exception as e:
                result = f"Tool error: {e}"
                print(f"[TraceIQ] Tool {tool_name} error: {e}", file=sys.stderr, flush=True)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result[:8000],  # cap tool output
            })

        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"[TraceIQ] Hit max iterations ({MAX_ITERATIONS})", file=sys.stderr, flush=True)

    print(f"[TraceIQ] Agent finished. Extracting recommendations...", file=sys.stderr, flush=True)

    parsed = _extract_json_result(final_content)

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
        "result_type": "experiment_analysis",
    }

    if not output["recommendations"] and final_content:
        output["raw_analysis"] = final_content[:5000]
        extracted = _extract_from_markdown(final_content, experiment_name, dataset_name)
        if extracted.get("recommendations"):
            output["recommendations"] = extracted["recommendations"]
            output["failure_patterns"] = extracted.get("failure_patterns", [])
            if extracted.get("overall_scores"):
                output["overall_scores"] = extracted["overall_scores"]
        else:
            output["error"] = "Could not extract structured recommendations — see raw_analysis"

    return output


def _lc_tools_to_anthropic(lc_tools) -> list:
    """Convert LangChain tools to Anthropic tool format."""
    result = []
    for t in lc_tools:
        # Try to get the JSON schema from the tool
        try:
            schema = t.args_schema.model_json_schema() if hasattr(t, "args_schema") and t.args_schema else {}
        except Exception:
            schema = {}

        input_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

        result.append({
            "name": t.name,
            "description": t.description or "",
            "input_schema": input_schema,
        })
    return result


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

    return result


def _extract_json_result(text: str) -> dict:
    """Extract a JSON result block from the agent's final message."""
    if not text:
        return {}

    import re

    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

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
