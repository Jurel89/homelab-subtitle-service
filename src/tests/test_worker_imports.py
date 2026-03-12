"""Smoke test: verify worker.py can be imported without errors."""


def test_worker_module_imports():
    """Importing the worker module should not raise ImportError."""
    from homelab_subs.server.worker import process_job, JobContext
    assert callable(process_job)
