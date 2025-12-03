"""
Performance monitoring utilities for tracking system resources during transcription.

This module provides real-time monitoring of:
- CPU usage (per-core and overall)
- Memory usage (RSS, VMS, percent)
- GPU usage and memory (if available)
- Disk I/O
- Network I/O (optional)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


@dataclass
class SystemMetrics:
    """
    Snapshot of system resource usage at a point in time.
    """
    timestamp: float
    cpu_percent: float  # Overall CPU usage (0-100)
    cpu_count: int  # Number of CPU cores
    memory_used_mb: float  # Memory used in MB
    memory_total_mb: float  # Total memory in MB
    memory_percent: float  # Memory usage percentage (0-100)
    disk_read_mb: float  # Disk read in MB since process start
    disk_write_mb: float  # Disk write in MB since process start
    
    # GPU metrics (None if not available)
    gpu_count: Optional[int] = None
    gpu_utilization: Optional[float] = None  # GPU compute usage (0-100)
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None  # Temperature in Celsius

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "memory_percent": round(self.memory_percent, 1),
            "disk_read_mb": round(self.disk_read_mb, 2),
            "disk_write_mb": round(self.disk_write_mb, 2),
            "gpu_count": self.gpu_count,
            "gpu_utilization": round(self.gpu_utilization, 1) if self.gpu_utilization else None,
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 1) if self.gpu_memory_used_mb else None,
            "gpu_memory_total_mb": round(self.gpu_memory_total_mb, 1) if self.gpu_memory_total_mb else None,
            "gpu_memory_percent": round(self.gpu_memory_percent, 1) if self.gpu_memory_percent else None,
            "gpu_temperature": round(self.gpu_temperature, 1) if self.gpu_temperature else None,
        }


class PerformanceMonitor:
    """
    Monitor system performance metrics during transcription jobs.
    """

    def __init__(self):
        self._process = None
        self._gpu_initialized = False
        self._gpu_available = False
        self._initial_disk_io = None
        
        if psutil is None:
            raise ImportError(
                "psutil is required for performance monitoring. "
                "Install with: pip install psutil"
            )
        
        self._process = psutil.Process(os.getpid())
        
        # Initialize GPU monitoring if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_available = True
                self._gpu_initialized = True
            except Exception:
                self._gpu_available = False
        
        # Get initial disk I/O counters
        try:
            self._initial_disk_io = self._process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self._initial_disk_io = None

    def __del__(self):
        """Cleanup GPU monitoring on destruction."""
        if self._gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_metrics(self) -> SystemMetrics:
        """
        Get current system metrics snapshot.
        
        Returns
        -------
        SystemMetrics
            Current system resource usage
        """
        # CPU metrics
        cpu_percent = self._process.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory_info = self._process.memory_info()
        memory_used_mb = memory_info.rss / (1024 * 1024)
        
        virtual_memory = psutil.virtual_memory()
        memory_total_mb = virtual_memory.total / (1024 * 1024)
        memory_percent = virtual_memory.percent
        
        # Disk I/O metrics
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        if self._initial_disk_io:
            try:
                current_io = self._process.io_counters()
                disk_read_mb = (current_io.read_bytes - self._initial_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (current_io.write_bytes - self._initial_disk_io.write_bytes) / (1024 * 1024)
            except (AttributeError, psutil.AccessDenied):
                pass
        
        # GPU metrics
        gpu_count = None
        gpu_utilization = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_memory_percent = None
        gpu_temperature = None
        
        if self._gpu_available:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                # Get metrics from first GPU (index 0)
                if gpu_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = float(utilization.gpu)
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used_mb = mem_info.used / (1024 * 1024)
                    gpu_memory_total_mb = mem_info.total / (1024 * 1024)
                    gpu_memory_percent = (mem_info.used / mem_info.total) * 100
                    
                    # GPU temperature
                    try:
                        gpu_temperature = float(pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ))
                    except Exception:
                        gpu_temperature = None
                        
            except Exception:
                # GPU metrics unavailable, leave as None
                pass
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_percent=memory_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            gpu_count=gpu_count,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature,
        )

    @staticmethod
    def get_summary_stats(metrics_list: list[SystemMetrics]) -> dict:
        """
        Calculate summary statistics from a list of metrics.
        
        Parameters
        ----------
        metrics_list : list[SystemMetrics]
            List of metrics snapshots
            
        Returns
        -------
        dict
            Summary statistics (avg, max, min)
        """
        if not metrics_list:
            return {}
        
        cpu_values = [m.cpu_percent for m in metrics_list]
        memory_used_values = [m.memory_used_mb for m in metrics_list]
        disk_read_values = [m.disk_read_mb for m in metrics_list]
        disk_write_values = [m.disk_write_mb for m in metrics_list]

        summary = {
            "cpu_avg": round(sum(cpu_values) / len(cpu_values), 1),
            "cpu_max": round(max(cpu_values), 1),
            "cpu_min": round(min(cpu_values), 1),
            "memory_avg_mb": round(sum(memory_used_values) / len(memory_used_values), 1),
            "memory_max_mb": round(max(memory_used_values), 1),
            "memory_min_mb": round(min(memory_used_values), 1),
            "disk_read_avg_mb": round(sum(disk_read_values) / len(disk_read_values), 2),
            "disk_write_avg_mb": round(sum(disk_write_values) / len(disk_write_values), 2),
            "samples": len(metrics_list),
        }
        
        # Add GPU stats if available
        gpu_util_values = [m.gpu_utilization for m in metrics_list if m.gpu_utilization is not None]
        if gpu_util_values:
            summary["gpu_avg"] = round(sum(gpu_util_values) / len(gpu_util_values), 1)
            summary["gpu_max"] = round(max(gpu_util_values), 1)
            summary["gpu_min"] = round(min(gpu_util_values), 1)
        
        gpu_mem_values = [m.gpu_memory_percent for m in metrics_list if m.gpu_memory_percent is not None]
        if gpu_mem_values:
            summary["gpu_memory_avg"] = round(sum(gpu_mem_values) / len(gpu_mem_values), 1)
            summary["gpu_memory_max"] = round(max(gpu_mem_values), 1)
        
        return summary
