"""
TraceIQ experiment tools — analysis tools for the prompt advisor agent.

All tools make live LangSmith API calls using the api_key bound via closure.
Created via `make_experiment_tools(api_key)` factory function.
"""

import json
import statistics
from typing import Any


BASE_URL = "https://api.smith.langchain.com/api/v1"


def _ssl_context():
    """Return an SSL context that works on macOS where certifi may not be installed."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    # Try macOS system certificates
    ctx = ssl.create_default_context()
    try:
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    except Exception:
        pass
    return ctx


def _get(api_key: str, path: str, params: dict = None) -> Any:
    """Make a GET request to the LangSmith API."""
    import urllib.request
    import urllib.parse

    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _post(api_key: str, path: str, body: dict) -> Any:
    """Make a POST request to the LangSmith API."""
    import urllib.request

    url = BASE_URL + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _avg_scores_from_feedback_stats(feedback_stats: dict) -> dict:
    """Extract average scores per metric from feedback_stats."""
    scores = {}
    if not feedback_stats:
        return scores
    for metric, stats in feedback_stats.items():
        if isinstance(stats, dict):
            avg = stats.get("avg")
            if avg is not None:
                try:
                    scores[metric] = round(float(avg), 4)
                except (TypeError, ValueError):
                    pass
        elif isinstance(stats, (int, float)):
            scores[metric] = round(float(stats), 4)
    return scores


def _text_preview(obj: Any, max_len: int = 200) -> str:
    """Extract a short text preview from inputs/outputs."""
    if obj is None:
        return ""
    raw = json.dumps(obj) if not isinstance(obj, str) else obj
    return raw[:max_len]


def make_experiment_tools(api_key: str) -> list:
    """Create experiment analysis tool functions with api_key bound via closure.

    Returns a list of langchain @tool-decorated functions.
    """
    from langchain_core.tools import tool

    # -----------------------------------------------------------------------
    # Tool 1: list_datasets
    # -----------------------------------------------------------------------

    @tool
    def list_datasets() -> str:
        """List all LangSmith datasets available in the account.

        Returns a JSON list of datasets with name, id, example_count, session_count,
        and last_session_start_time.
        """
        try:
            import sys
            print(f"[TraceIQ] Listing available datasets...", file=sys.stderr, flush=True)
            data = _get(api_key, "/datasets")
            # API returns list directly or wrapped in a key
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("datasets", data.get("results", []))
            else:
                rows = []

            result = []
            for d in rows:
                result.append({
                    "id": d.get("id", ""),
                    "name": d.get("name", ""),
                    "example_count": d.get("example_count", 0),
                    "session_count": d.get("session_count", 0),
                    "last_session_start_time": d.get("last_session_start_time"),
                })
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -----------------------------------------------------------------------
    # Tool 2: list_experiments
    # -----------------------------------------------------------------------

    @tool
    def list_experiments(dataset_id: str) -> str:
        """List all experiments (sessions) for a given dataset.

        Args:
            dataset_id: The dataset UUID to fetch experiments for.

        Returns a JSON list of experiments with name, id, created_at.
        """
        try:
            data = _get(api_key, "/sessions", params={"reference_dataset_id": dataset_id})
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("sessions", data.get("results", []))
            else:
                rows = []

            result = []
            for s in rows:
                result.append({
                    "id": s.get("id", ""),
                    "name": s.get("name", ""),
                    "created_at": s.get("start_time") or s.get("created_at"),
                })
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -----------------------------------------------------------------------
    # Tool 3: fetch_experiment_rows
    # -----------------------------------------------------------------------

    @tool
    def fetch_experiment_rows(experiment_id: str, limit: int = 100) -> str:
        """Fetch runs for an experiment session with scores and previews.

        Args:
            experiment_id: The experiment session UUID.
            limit: Maximum number of rows to fetch (default 100).

        Returns a JSON list of rows with id, input_preview, output_preview, scores,
        status, and error.
        """
        try:
            import sys
            print(f"[TraceIQ] Fetching experiment rows (up to {limit})...", file=sys.stderr, flush=True)
            data = _post(api_key, "/runs/query", {
                "session": [experiment_id],
                "filter": "eq(is_root, true)",
                "limit": limit,
            })

            runs = data.get("runs", []) if isinstance(data, dict) else data
            if not isinstance(runs, list):
                runs = []

            print(f"[TraceIQ] Got {len(runs)} experiment rows — extracting scores...", file=sys.stderr, flush=True)
            result = []
            for run in runs:
                feedback_stats = run.get("feedback_stats") or {}
                scores = _avg_scores_from_feedback_stats(feedback_stats)

                result.append({
                    "id": run.get("id", ""),
                    "input_preview": _text_preview(run.get("inputs"), 200),
                    "output_preview": _text_preview(run.get("outputs"), 200),
                    "scores": scores,
                    "status": run.get("status", ""),
                    "error": run.get("error"),
                })

            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -----------------------------------------------------------------------
    # Tool 4: get_failing_rows
    # -----------------------------------------------------------------------

    @tool
    def get_failing_rows(experiment_id: str, metric: str, threshold: float = 0.5) -> str:
        """Fetch rows where a specific metric score is below threshold, with full inputs/outputs.

        Args:
            experiment_id: The experiment session UUID.
            metric: The metric name to filter by (e.g. 'marks_exact_match', 'llm_strictness').
            threshold: Score threshold below which a row is considered failing (default 0.5).

        Returns full inputs + outputs for failing rows so you can analyze the content.
        """
        try:
            import sys
            print(f"[TraceIQ] Fetching rows to find failures on metric '{metric}' (threshold: {threshold})...", file=sys.stderr, flush=True)
            data = _post(api_key, "/runs/query", {
                "session": [experiment_id],
                "filter": "eq(is_root, true)",
                "limit": 100,
            })

            runs = data.get("runs", []) if isinstance(data, dict) else data
            if not isinstance(runs, list):
                runs = []

            failing = []
            for run in runs:
                feedback_stats = run.get("feedback_stats") or {}
                scores = _avg_scores_from_feedback_stats(feedback_stats)

                score = scores.get(metric)
                if score is not None and score < threshold:
                    # Truncate inputs/outputs to keep context manageable
                    inputs = run.get("inputs") or {}
                    outputs = run.get("outputs") or {}
                    inputs_str = json.dumps(inputs, default=str)[:1000]
                    outputs_str = json.dumps(outputs, default=str)[:1000]
                    failing.append({
                        "id": run.get("id", ""),
                        "inputs": inputs_str,
                        "outputs": outputs_str,
                        "scores": scores,
                        f"{metric}_score": score,
                    })

            print(f"[TraceIQ] Found {len(failing)} failing rows out of {len(runs)} checked for '{metric}' < {threshold}", file=sys.stderr, flush=True)
            # Cap at 10 samples — enough to identify patterns without flooding Claude's context
            sample = failing[:10]
            return json.dumps({
                "metric": metric,
                "threshold": threshold,
                "total_failing": len(failing),
                "total_checked": len(runs),
                "sample_size": len(sample),
                "failing_rows": sample,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -----------------------------------------------------------------------
    # Tool 5: compare_experiments
    # -----------------------------------------------------------------------

    @tool
    def compare_experiments(experiment_id_a: str, experiment_id_b: str) -> str:
        """Compare two experiments side-by-side by computing avg score per metric.

        Args:
            experiment_id_a: UUID of the first experiment session.
            experiment_id_b: UUID of the second experiment session.

        Returns a side-by-side comparison JSON with per-metric averages and deltas.
        """
        def _fetch_metrics(exp_id: str) -> tuple[dict, int]:
            """Returns (metric -> list of scores, row_count)."""
            data = _post(api_key, "/runs/query", {
                "session": [exp_id],
                "filter": "eq(is_root, true)",
                "limit": 200,
            })
            runs = data.get("runs", []) if isinstance(data, dict) else data
            if not isinstance(runs, list):
                runs = []

            metric_values: dict[str, list[float]] = {}
            for run in runs:
                feedback_stats = run.get("feedback_stats") or {}
                scores = _avg_scores_from_feedback_stats(feedback_stats)
                for metric, val in scores.items():
                    metric_values.setdefault(metric, []).append(val)

            return metric_values, len(runs)

        try:
            metrics_a, count_a = _fetch_metrics(experiment_id_a)
            metrics_b, count_b = _fetch_metrics(experiment_id_b)

            # Combine all metrics seen in either experiment
            all_metrics = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

            comparison = []
            for metric in all_metrics:
                vals_a = metrics_a.get(metric, [])
                vals_b = metrics_b.get(metric, [])
                avg_a = round(statistics.mean(vals_a), 4) if vals_a else None
                avg_b = round(statistics.mean(vals_b), 4) if vals_b else None
                delta = round(avg_b - avg_a, 4) if (avg_a is not None and avg_b is not None) else None
                comparison.append({
                    "metric": metric,
                    "experiment_a_avg": avg_a,
                    "experiment_b_avg": avg_b,
                    "delta": delta,
                    "direction": ("improved" if delta and delta > 0 else "regressed" if delta and delta < 0 else "unchanged") if delta is not None else "unknown",
                })

            return json.dumps({
                "experiment_a_id": experiment_id_a,
                "experiment_a_row_count": count_a,
                "experiment_b_id": experiment_id_b,
                "experiment_b_row_count": count_b,
                "metrics_comparison": comparison,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [list_datasets, list_experiments, fetch_experiment_rows, get_failing_rows, compare_experiments]
