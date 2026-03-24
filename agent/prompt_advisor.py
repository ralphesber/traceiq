"""
TraceIQ Prompt Advisor — agentic analysis of experiment results.

Direct Anthropic SDK tool-calling loop. No LangGraph, no wrappers.
Full control over timeouts and execution.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable

import anthropic

SYSTEM_PROMPT = """You are a prompt engineering advisor for AI grading/evaluation systems.

You have tools to fetch LangSmith experiment results:
- list_datasets: List all datasets
- list_experiments: List experiments for a dataset
- fetch_experiment_rows: Get all rows in an experiment with scores and previews
- get_failing_rows: Get full inputs/outputs for rows where a metric is below threshold
- compare_experiments: Compare two experiments side-by-side

Read the user's question. Use only the tools you need. Be evidence-driven.

End your FINAL response with a JSON block in a ```json code fence:

```json
{
  "experiment_name": "string",
  "dataset_name": "string",
  "overall_scores": {"metric_name": 0.85},
  "weakest_metrics": ["metric1"],
  "failure_patterns": [
    {
      "pattern": "description",
      "affected_metric": "metric_name",
      "frequency": "X of Y rows",
      "example_input": "excerpt",
      "example_output": "excerpt",
      "root_cause": "what causes this"
    }
  ],
  "recommendations": [
    {
      "priority": 1,
      "change": "specific change to make",
      "rationale": "why",
      "expected_impact": "which metric improves"
    }
  ],
  "generated_at": "ISO timestamp"
}
```
"""

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8000
API_TIMEOUT = 120  # seconds per API call
MAX_ITERATIONS = 15


def run_prompt_advisor(
    api_key: str,
    dataset_id: str,
    experiment_id: str,
    question: str = "",
    step_callback: Callable[[str], None] = None,
) -> dict:

    def log(msg: str):
        print(f"[TraceIQ] {msg}", file=sys.stderr, flush=True)
        if step_callback:
            step_callback(msg)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {"error": "ANTHROPIC_API_KEY not set", "experiment_id": experiment_id,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    from agent.experiment_tools import make_experiment_tools, _get as ls_get

    # Resolve metadata
    experiment_name = experiment_id
    dataset_name = dataset_id
    try:
        ds = ls_get(api_key, f"/datasets/{dataset_id}")
        dataset_name = ds.get("name", dataset_id)
    except Exception:
        pass
    try:
        sessions = ls_get(api_key, "/sessions", params={"reference_dataset_id": dataset_id})
        sessions = sessions if isinstance(sessions, list) else sessions.get("sessions", [])
        for s in sessions:
            if s.get("id") == experiment_id:
                experiment_name = s.get("name", experiment_id)
                break
    except Exception:
        pass

    log(f"Analyzing '{experiment_name}' on '{dataset_name}'...")

    lc_tools = make_experiment_tools(api_key)
    anthropic_tools = _build_anthropic_tools(lc_tools)
    tool_map = {t.name: t for t in lc_tools}

    question_line = question.strip() or "What prompt improvements would increase eval scores?"
    messages = [{"role": "user", "content": f"""Experiment: "{experiment_name}" (id: {experiment_id})
Dataset: "{dataset_name}" (id: {dataset_id})
Question: {question_line}
Use your tools to investigate. End with a JSON block matching the required schema."""}]

    client = anthropic.Anthropic(api_key=anthropic_key, timeout=API_TIMEOUT)
    final_content = ""

    for iteration in range(MAX_ITERATIONS):
        log(f"Thinking... (step {iteration + 1})")
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=anthropic_tools,
                messages=messages,
            )
        except anthropic.APITimeoutError:
            return {"error": f"API timed out after {API_TIMEOUT}s on step {iteration+1}",
                    "experiment_name": experiment_name, "dataset_name": dataset_name,
                    "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"error": f"API error on step {iteration+1}: {e}",
                    "experiment_name": experiment_name, "dataset_name": dataset_name,
                    "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}

        # Capture text
        for block in response.content:
            if hasattr(block, "text") and block.text:
                final_content = block.text

        if response.stop_reason == "end_turn":
            log("Analysis complete.")
            break

        # Execute tool calls
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            log(f"Using tool: {tu.name}")
            try:
                lc_tool = tool_map.get(tu.name)
                result = lc_tool.invoke(tu.input) if lc_tool else f"Unknown tool: {tu.name}"
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
            except Exception as e:
                result = f"Tool error: {e}"
                log(f"Tool {tu.name} failed: {e}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result[:8000],
            })

        messages.append({"role": "user", "content": tool_results})

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
        output["error"] = "Could not extract structured recommendations — see raw_analysis"

    return output


def _build_anthropic_tools(lc_tools) -> list:
    result = []
    for t in lc_tools:
        try:
            schema = t.args_schema.model_json_schema() if hasattr(t, "args_schema") and t.args_schema else {}
        except Exception:
            schema = {}
        result.append({
            "name": t.name,
            "description": t.description or "",
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        })
    return result


def _extract_json_result(text: str) -> dict:
    if not text:
        return {}
    import re
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    for start in reversed([i for i, c in enumerate(text) if c == '{']):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i+1])
                        if "recommendations" in parsed or "overall_scores" in parsed:
                            return parsed
                    except Exception:
                        pass
                    break
    return {}
