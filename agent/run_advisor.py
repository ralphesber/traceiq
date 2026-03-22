#!/usr/bin/env python3
"""CLI entrypoint for prompt advisor — used by GET /experiments/analyze/stream."""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompt_advisor import run_prompt_advisor

def main():
    parser = argparse.ArgumentParser(description="Run TraceIQ prompt advisor")
    parser.add_argument("--api-key", required=True, help="LangSmith API key")
    parser.add_argument("--dataset-id", required=True, help="LangSmith dataset ID")
    parser.add_argument("--experiment-id", required=True, help="LangSmith experiment ID")
    parser.add_argument("--question", default="", help="User question/focus")
    args = parser.parse_args()

    result = run_prompt_advisor(
        api_key=args.api_key,
        dataset_id=args.dataset_id,
        experiment_id=args.experiment_id,
        question=args.question,
    )
    print(json.dumps(result, default=str))

if __name__ == "__main__":
    main()
