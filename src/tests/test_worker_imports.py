"""Smoke test: verify worker.py can be imported without errors."""

import pytest


def test_worker_module_imports():
    """Importing the worker module should not raise ImportError."""
    try:
        from homelab_subs.server.worker import process_job  # noqa: F401
    except ModuleNotFoundError as e:
        pytest.skip(f"Server dependencies not installed: {e}")
    assert callable(process_job)
