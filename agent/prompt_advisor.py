"""
TraceIQ Prompt Advisor — agentic analysis of experiment results.

Uses LangGraph create_react_agent with:
- recursion_limit=10 (max 10 tool calls per run)
- ainvoke() so asyncio.wait_for() can enforce a hard timeout
"""

import asyncio
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

AGENT_TIMEOUT = 300     # 5 min hard wall for the entire agent run
RECURSION_LIMIT = 10    # max tool-call rounds before LangGraph forces a stop


def run_prompt_advisor(
    api_key: str,
    dataset_id: str,
    experiment_id: str,
    question: str = "",
    step_callback: Callable[[str], None] = None,
) -> dict:
    """Run agentic prompt advisor. Blocking — uses asyncio.run() internally."""

    def log(msg: str):
        print(f"[TraceIQ] {msg}", file=sys.stderr, flush=True)
        if step_callback:
            step_callback(msg)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {"error": "ANTHROPIC_API_KEY not set", "experiment_id": experiment_id,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    from agent.experiment_tools import make_experiment_tools, _get as ls_get

    # Resolve names
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

    tools = make_experiment_tools(api_key)

    model = ChatAnthropic(
        model_name="claude-sonnet-4-6",
        api_key=anthropic_key,
        max_tokens=8000,
        timeout=60,      # per API call
        max_retries=1,
    )

    agent = create_react_agent(model=model, tools=tools, prompt=SYSTEM_PROMPT)

    question_line = question.strip() or "What prompt improvements would increase eval scores?"
    user_message = f"""Experiment: "{experiment_name}" (id: {experiment_id})
Dataset: "{dataset_name}" (id: {dataset_id})
Question: {question_line}
Use your tools to investigate. End with a JSON block matching the required schema."""

    log(f"Starting agent (max {RECURSION_LIMIT} tool calls, {AGENT_TIMEOUT}s timeout)...")

    async def _run():
        final_state = None
        async with asyncio.timeout(AGENT_TIMEOUT):
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_message}]},
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode="updates",
            ):
                if "agent" in chunk:
                    # Claude is thinking / responding
                    msgs = chunk["agent"].get("messages", [])
                    for msg in msgs:
                        content = getattr(msg, "content", "") or ""
                        if isinstance(content, list):
                            content = " ".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in content if isinstance(b, dict) and b.get("type") == "text"
                            )
                        if content and len(content) > 10:
                            preview = content[:80].replace("\n", " ")
                            log(f"Agent: {preview}...")
                elif "tools" in chunk:
                    # Tool was called and returned
                    msgs = chunk["tools"].get("messages", [])
                    for msg in msgs:
                        name = getattr(msg, "name", "") or ""
                        if name:
                            log(f"Tool done: {name}")
                final_state = chunk
        return final_state

    try:
        last_chunk = asyncio.run(_run())
    except TimeoutError:
        log(f"Agent timed out after {AGENT_TIMEOUT}s")
        return {"error": f"Agent timed out after {AGENT_TIMEOUT} seconds",
                "experiment_name": experiment_name, "dataset_name": dataset_name,
                "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        import traceback
        log(f"Agent failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Agent failed: {e}",
                "experiment_name": experiment_name, "dataset_name": dataset_name,
                "experiment_id": experiment_id, "generated_at": datetime.now(timezone.utc).isoformat()}

    log("Agent finished. Extracting recommendations...")

    # Extract final text from the last agent chunk
    final_content = ""
    if last_chunk and "agent" in last_chunk:
        for msg in reversed(last_chunk["agent"].get("messages", [])):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            if content and len(content) > 50:
                final_content = content
                break

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
