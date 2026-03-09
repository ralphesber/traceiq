#!/usr/bin/env python3
"""
TraceIQ Demo - Generate realistic mock trace data and run analysis.

This demo creates synthetic LangSmith-style trace data that demonstrates:
- A latency regression mid-period
- A prompt change event
- A model switch
- Various error patterns

Run with: python demo.py
"""

import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone


def generate_mock_runs(num_runs: int = 1247, days: int = 7) -> list[dict]:
    """Generate realistic mock trace data."""
    runs = []
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    # Define the timeline events
    day_3 = start_date + timedelta(days=2)  # Prompt change on day 3
    day_5 = start_date + timedelta(days=4)  # Model change and latency regression on day 5

    # System prompts
    old_prompt = (
        "You are a helpful AI assistant. Answer questions accurately and concisely. "
        "Always cite your sources when providing factual information."
    )
    new_prompt = (
        "You are a helpful AI assistant specialized in technical support. "
        "Provide step-by-step solutions when possible. Be thorough but efficient. "
        "Always confirm the user's issue is resolved before ending the conversation."
    )

    # Error templates
    error_templates = [
        ("Context length exceeded", 0.38),
        ("Tool call parsing error", 0.31),
        ("Timeout", 0.20),
        ("Rate limit exceeded", 0.08),
        ("Invalid response format", 0.03),
    ]

    for i in range(num_runs):
        # Distribute runs across the time period with some randomness
        progress = i / num_runs
        run_time = start_date + timedelta(
            seconds=progress * days * 24 * 3600 + random.uniform(-3600, 3600)
        )

        # Determine which period we're in
        is_after_prompt_change = run_time >= day_3
        is_after_model_change = run_time >= day_5

        # Select model based on timeline
        if is_after_model_change:
            model = "gpt-4o-mini"
            # After model change: higher latency (regression!)
            base_latency = random.gauss(1.4, 0.4)
        else:
            model = "gpt-4o"
            base_latency = random.gauss(0.8, 0.2)

        # Ensure positive latency
        latency = max(0.1, base_latency)

        # Occasional slow runs
        if random.random() < 0.05:
            latency *= random.uniform(2, 5)

        # Select prompt
        system_prompt = new_prompt if is_after_prompt_change else old_prompt

        # Determine if this run is an error
        # Higher error rate in second half to show regression
        error_rate = 0.09 if is_after_model_change else 0.05
        is_error = random.random() < error_rate

        # Calculate tokens (somewhat correlated with latency)
        base_tokens = int(latency * 800 + random.uniform(200, 600))
        prompt_tokens = int(base_tokens * 0.3)
        completion_tokens = base_tokens - prompt_tokens

        # Build the run object
        end_time = run_time + timedelta(seconds=latency)

        run = {
            "id": f"run_{i:05d}",
            "name": "agent_chain",
            "run_type": "chain",
            "start_time": run_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status": "error" if is_error else "success",
            "inputs": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query #{i}"},
                ]
            },
            "outputs": {
                "result": None if is_error else f"Response to query #{i}",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": base_tokens,
                },
            },
            "extra": {
                "invocation_params": {
                    "model": model,
                    "temperature": 0.7,
                },
                "metadata": {
                    "session_id": f"session_{i % 100:03d}",
                },
            },
            "total_tokens": base_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        # Add error details if this is an error run
        if is_error:
            # Select error based on weighted distribution
            rand = random.random()
            cumulative = 0
            for error_msg, weight in error_templates:
                cumulative += weight
                if rand <= cumulative:
                    run["error"] = error_msg
                    break
            else:
                run["error"] = error_templates[0][0]

        runs.append(run)

    # Shuffle to make it more realistic (not perfectly ordered)
    random.shuffle(runs)

    return runs


def main():
    print("=" * 60)
    print("TraceIQ Demo - Generating mock trace data...")
    print("=" * 60)
    print()

    # Generate mock data
    runs = generate_mock_runs(num_runs=1247, days=7)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(runs, f)
        mock_file = f.name

    print(f"Generated {len(runs)} mock runs")
    print(f"Saved to: {mock_file}")
    print()
    print("Running TraceIQ analysis...")
    print()
    print("-" * 60)
    print()

    # Run traceiq with the mock data
    result = subprocess.run(
        [
            sys.executable,
            "traceiq.py",
            "--project", "demo-agent",
            "--days", "7",
            "--mock-data", mock_file,
            "--output", "markdown",
        ],
        capture_output=False,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
