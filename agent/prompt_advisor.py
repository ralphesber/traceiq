"""
TraceIQ Prompt Advisor — agentic analysis of experiment results.

Uses LangGraph's prebuilt ReAct agent.
Claude decides which LangSmith tools to call based on the user's question.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent


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


def run_prompt_advisor(
    api_key: str,
    dataset_id: str,
    experiment_id: str,
    question: str = "",
    step_callback: Callable[[str], None] = None,
) -> dict:
    """Run agentic prompt advisor using LangGraph ReAct agent (invoke, no streaming)."""

    def log(msg: str):
        print(f"[TraceIQ] {msg}", file=sys.stderr, flush=True)
        if step_callback:
            step_callback(msg)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {"error": "ANTHROPIC_API_KEY not set", "experiment_id": experiment_id,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    from agent.experiment_tools import make_experiment_tools, _get as ls_get

    # Fetch metadata
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

    log(f"Analyzing '{experiment_name}' on dataset '{dataset_name}'...")

    tools = make_experiment_tools(api_key)

    model = ChatAnthropic(
        model_name="claude-sonnet-4-6",
        api_key=anthropic_key,
        max_tokens=8000,
        timeout=120,
        max_retries=1,
    )

    agent = create_react_agent(model=model, tools=tools, prompt=SYSTEM_PROMPT)

    question_line = question.strip() or "What prompt improvements would increase eval scores?"

    user_message = f"""Experiment: "{experiment_name}" (id: {experiment_id})
Dataset: "{dataset_name}" (id: {dataset_id})

Question: {question_line}

Use your tools to investigate. End with a JSON block matching the required schema."""

    log(f"Calling agent (question: '{question[:60]}')...")

    try:
        result_state = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
        )
    except Exception as e:
        import traceback
        log(f"Agent failed: {e}")
        return {
            "error": f"Agent execution failed: {e}",
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "experiment_id": experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Extract final text from last AI message
    final_content = ""
    for msg in reversed(result_state.get("messages", [])):
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        if content and len(content) > 50:
            final_content = content
            break

    log("Agent finished. Extracting recommendations...")

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


def _extract_json_result(text: str) -> dict:
    if not text:
        return {}
    import re
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
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
                    except json.JSONDecodeError:
                        pass
                    break
    return {}
