import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_get_db_conn_returns_none_without_env(monkeypatch):
    """Returns None when DATABASE_URL not set"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from server import _get_db_conn
    assert _get_db_conn() is None

def test_history_uses_file_fallback_without_db(monkeypatch, tmp_path):
    """Falls back to file when no DATABASE_URL"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import server
    # Patch HISTORY_DIR to tmp_path
    original = server.HISTORY_DIR
    server.HISTORY_DIR = tmp_path
    try:
        filename = server._save_to_history({
            "hypothesis": "test hypothesis",
            "verdict": "supported",
            "confidence": "high",
            "project": "test",
            "traces_analyzed": 10,
            "generated_at": "2026-03-23T00:00:00Z",
        })
        assert filename is not None
        assert (tmp_path / filename).exists()
    finally:
        server.HISTORY_DIR = original
