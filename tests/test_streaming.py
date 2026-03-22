# tests/test_streaming.py
import pytest
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_parse_traceiq_log_line_step():
    """[TraceIQ] lines become step events"""
    from server import _parse_sse_line
    event = _parse_sse_line("[TraceIQ] Fetched 50 total runs")
    assert event == {"type": "step", "text": "Fetched 50 total runs"}

def test_parse_non_traceiq_line_returns_none():
    """Non-[TraceIQ] lines are ignored"""
    from server import _parse_sse_line
    assert _parse_sse_line("some random output") is None

def test_parse_empty_line_returns_none():
    from server import _parse_sse_line
    assert _parse_sse_line("") is None

def test_parse_traceiq_strips_prefix_whitespace():
    from server import _parse_sse_line
    event = _parse_sse_line("[TraceIQ]   Classifying traces")
    assert event == {"type": "step", "text": "Classifying traces"}
