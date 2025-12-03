"""
Tests for performance monitoring module.
"""

import time
from unittest.mock import MagicMock, patch

import pytest


def test_system_metrics_dataclass():
    """Test SystemMetrics dataclass structure."""
    from homelab_subs.core.performance import SystemMetrics

    # Create a metrics instance
    timestamp = time.time()
    metrics = SystemMetrics(
        timestamp=timestamp,
        cpu_percent=45.5,
        cpu_count=8,
        memory_used_mb=8192.5,
        memory_total_mb=16384.0,
        memory_percent=60.2,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_count=1,
        gpu_utilization=75.0,
        gpu_memory_used_mb=4096.0,
        gpu_memory_total_mb=8192.0,
        gpu_memory_percent=50.0,
        gpu_temperature=65.0,
    )

    # Verify all fields exist
    assert metrics.cpu_percent == 45.5
    assert metrics.cpu_count == 8
    assert metrics.memory_percent == 60.2
    assert metrics.memory_used_mb == 8192.5
    assert metrics.memory_total_mb == 16384.0
    assert metrics.gpu_utilization == 75.0
    assert metrics.timestamp == timestamp


def test_system_metrics_optional_gpu_fields():
    """Test SystemMetrics with optional GPU fields as None."""
    from homelab_subs.core.performance import SystemMetrics

    # Create metrics without GPU data
    timestamp = time.time()
    metrics = SystemMetrics(
        timestamp=timestamp,
        cpu_percent=45.5,
        cpu_count=8,
        memory_used_mb=8192.5,
        memory_total_mb=16384.0,
        memory_percent=60.2,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_count=None,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_memory_total_mb=None,
        gpu_memory_percent=None,
        gpu_temperature=None,
    )

    assert metrics.gpu_count is None
    assert metrics.gpu_utilization is None
    assert metrics.gpu_memory_used_mb is None
    assert metrics.gpu_temperature is None


def test_system_metrics_to_dict():
    """Test SystemMetrics to_dict method."""
    from homelab_subs.core.performance import SystemMetrics

    timestamp = time.time()
    metrics = SystemMetrics(
        timestamp=timestamp,
        cpu_percent=45.567,
        cpu_count=8,
        memory_used_mb=8192.567,
        memory_total_mb=16384.0,
        memory_percent=60.234,
        disk_read_mb=100.123,
        disk_write_mb=50.456,
        gpu_count=None,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_memory_total_mb=None,
        gpu_memory_percent=None,
        gpu_temperature=None,
    )

    d = metrics.to_dict()
    assert isinstance(d, dict)
    assert d["cpu_percent"] == 45.6  # Rounded to 1 decimal
    assert d["memory_percent"] == 60.2  # Rounded to 1 decimal
    assert d["disk_read_mb"] == 100.12  # Rounded to 2 decimals
    assert d["gpu_utilization"] is None


def test_performance_monitor_without_psutil():
    """Test that PerformanceMonitor raises error without psutil."""
    with patch("homelab_subs.core.performance.psutil", None):
        from homelab_subs.core.performance import PerformanceMonitor

        with pytest.raises(ImportError, match="psutil is required"):
            PerformanceMonitor()


def test_performance_monitor_get_summary_stats():
    """Test calculating summary statistics."""
    from homelab_subs.core.performance import PerformanceMonitor, SystemMetrics

    # Don't initialize the monitor (would need psutil)
    # Just test the static method
    timestamp = time.time()
    metrics_list = [
        SystemMetrics(
            timestamp=timestamp,
            cpu_percent=40.0,
            cpu_count=8,
            memory_used_mb=2000.0,
            memory_total_mb=16384.0,
            memory_percent=12.2,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_count=None,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_memory_total_mb=None,
            gpu_memory_percent=None,
            gpu_temperature=None,
        ),
        SystemMetrics(
            timestamp=timestamp + 1,
            cpu_percent=50.0,
            cpu_count=8,
            memory_used_mb=2500.0,
            memory_total_mb=16384.0,
            memory_percent=15.3,
            disk_read_mb=110.0,
            disk_write_mb=60.0,
            gpu_count=None,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_memory_total_mb=None,
            gpu_memory_percent=None,
            gpu_temperature=None,
        ),
        SystemMetrics(
            timestamp=timestamp + 2,
            cpu_percent=60.0,
            cpu_count=8,
            memory_used_mb=3000.0,
            memory_total_mb=16384.0,
            memory_percent=18.3,
            disk_read_mb=120.0,
            disk_write_mb=70.0,
            gpu_count=None,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_memory_total_mb=None,
            gpu_memory_percent=None,
            gpu_temperature=None,
        ),
    ]

    summary = PerformanceMonitor.get_summary_stats(metrics_list)

    assert summary["cpu_avg"] == 50.0  # Average of 40, 50, 60
    assert summary["cpu_max"] == 60.0
    assert summary["cpu_min"] == 40.0
    assert summary["memory_avg_mb"] == 2500.0  # Average of 2000, 2500, 3000
    assert summary["memory_max_mb"] == 3000.0
    assert "disk_read_avg_mb" in summary


def test_performance_monitor_summary_with_empty_list():
    """Test summary stats with empty list."""
    from homelab_subs.core.performance import PerformanceMonitor

    summary = PerformanceMonitor.get_summary_stats([])

    # Should return empty dict or handle gracefully
    assert isinstance(summary, dict)
    assert len(summary) == 0 or summary.get("cpu_avg") is None


def test_performance_monitor_summary_with_gpu():
    """Test summary stats with GPU metrics."""
    from homelab_subs.core.performance import PerformanceMonitor, SystemMetrics

    timestamp = time.time()
    metrics_list = [
        SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            cpu_count=8,
            memory_used_mb=2000.0,
            memory_total_mb=16384.0,
            memory_percent=12.2,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_count=1,
            gpu_utilization=70.0,
            gpu_memory_used_mb=4000.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=48.8,
            gpu_temperature=65.0,
        ),
        SystemMetrics(
            timestamp=timestamp + 1,
            cpu_percent=60.0,
            cpu_count=8,
            memory_used_mb=2500.0,
            memory_total_mb=16384.0,
            memory_percent=15.3,
            disk_read_mb=110.0,
            disk_write_mb=60.0,
            gpu_count=1,
            gpu_utilization=80.0,
            gpu_memory_used_mb=4200.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=51.3,
            gpu_temperature=70.0,
        ),
    ]

    summary = PerformanceMonitor.get_summary_stats(metrics_list)

    assert "gpu_avg" in summary
    assert "gpu_max" in summary
    assert summary["gpu_avg"] == 75.0  # Average of 70, 80
    assert summary["gpu_max"] == 80.0


def test_performance_monitor_summary_partial_gpu():
    """Test summary stats when some metrics have GPU and others don't."""
    from homelab_subs.core.performance import PerformanceMonitor, SystemMetrics

    timestamp = time.time()
    metrics_list = [
        SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            cpu_count=8,
            memory_used_mb=2000.0,
            memory_total_mb=16384.0,
            memory_percent=12.2,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_count=1,
            gpu_utilization=70.0,
            gpu_memory_used_mb=4000.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=48.8,
            gpu_temperature=65.0,
        ),
        SystemMetrics(
            timestamp=timestamp + 1,
            cpu_percent=60.0,
            cpu_count=8,
            memory_used_mb=2500.0,
            memory_total_mb=16384.0,
            memory_percent=15.3,
            disk_read_mb=110.0,
            disk_write_mb=60.0,
            gpu_count=None,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_memory_total_mb=None,
            gpu_memory_percent=None,
            gpu_temperature=None,
        ),
    ]

    summary = PerformanceMonitor.get_summary_stats(metrics_list)

    # Should handle None values gracefully
    assert isinstance(summary, dict)
    assert "cpu_avg" in summary
    assert summary["cpu_avg"] == 55.0
