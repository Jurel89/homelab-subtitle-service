# Performance Monitoring & Logging

This document explains the monitoring and logging features of the homelab-subtitle-service, which capture performance metrics and job history for analysis and future Web UI integration.

## Overview

The monitoring system provides:
- **Real-time performance tracking**: CPU, memory, disk I/O, and GPU metrics (if available)
- **Persistent job logs**: SQLite database storing job metadata and performance history
- **Query API**: Programmatic access to job history and statistics
- **CLI tools**: View job history and performance metrics

## Installation

The monitoring features require additional dependencies:

```bash
# Install with monitoring dependencies
pip install psutil

# For GPU monitoring (Linux only)
pip install nvidia-ml-py
```

Or install all development dependencies:

```bash
pip install -e ".[dev]"
```

## Features

### 1. Performance Monitoring

The `PerformanceMonitor` class tracks:
- **CPU**: Usage percentage (overall and per-core if needed)
- **Memory**: Used MB, percentage, and available memory
- **Disk I/O**: Read/write bytes and operations
- **GPU** (NVIDIA only, Linux): Utilization %, memory usage MB, temperature °C

Metrics are collected every 2 seconds during subtitle generation.

### 2. Database Logging

Job information and metrics are stored in a SQLite database located at:
```
~/.homelab-subs/logs.db
```

The database contains two tables:

#### `jobs` table
Stores job metadata and performance summary:
```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    video_path TEXT NOT NULL,
    output_path TEXT,
    language TEXT,
    model TEXT,
    task TEXT,
    device TEXT,
    status TEXT NOT NULL,  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    error_message TEXT,
    -- Performance summary stats
    cpu_avg REAL,
    cpu_max REAL,
    memory_avg_mb REAL,
    memory_max_mb REAL,
    gpu_avg REAL,
    gpu_max REAL
);
```

#### `metrics` table
Stores time-series performance data:
```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    memory_used_mb REAL,
    disk_read_mb REAL,
    disk_write_mb REAL,
    gpu_utilization REAL,
    gpu_memory_used_mb REAL,
    gpu_temperature REAL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);
```

### 3. Pipeline Runner

The `PipelineRunner` class now owns the core subtitle generation pipeline and:
- Automatically creates job records (when database logging is enabled)
- Starts background metric collection (when monitoring is enabled)
- Updates job status (pending → running → completed/failed)
- Calculates performance summaries
- Stores all data in the database

The higher-level `JobService` composes a `PipelineRunner` instance and exposes
simple helpers used by the CLI (generate, batch, history) or any other
automation.

## Usage

### Command-Line Interface

#### Generate subtitles with monitoring (default)
```bash
subsvc generate input.mp4
```

Monitoring and database logging are **enabled by default**.

#### Disable monitoring
```bash
# Disable all monitoring
subsvc generate input.mp4 --no-monitoring

# Keep monitoring but don't save to database
subsvc generate input.mp4 --no-db-logging
```

#### Custom database location
```bash
subsvc generate input.mp4 --db-path /path/to/custom.db
```

#### View job history
```bash
# Show last 20 jobs
subsvc history

# Show last 50 jobs
subsvc history --limit 50

# Filter by status
subsvc history --status completed
subsvc history --status failed

# Show overall statistics
subsvc history --stats

# Show detailed job information
subsvc history --job-id <job-id>

# Custom database location
subsvc history --db-path /path/to/custom.db
```

### Programmatic Usage

#### Query job history
```python
from homelab_subs.core.db_logger import DatabaseLogger

db = DatabaseLogger()  # Uses default location

# Get recent jobs
recent_jobs = db.get_recent_jobs(limit=10)
for job in recent_jobs:
    print(f"{job.job_id}: {job.status} - {job.video_path}")

# Get specific job
job = db.get_job("job-id-here")
if job:
    print(f"Duration: {job.duration_seconds}s")
    print(f"Avg CPU: {job.cpu_avg}%")
    print(f"Avg Memory: {job.memory_avg_mb} MB")

# Get job metrics timeline
metrics = db.get_job_metrics("job-id-here")
for metric in metrics:
    print(f"{metric.timestamp}: CPU={metric.cpu_percent}%")

# Get overall statistics
stats = db.get_statistics()
print(f"Total jobs: {stats['total_jobs']}")
print(f"Completed: {stats['status_counts']['completed']}")
print(f"Avg duration: {stats['avg_duration_seconds']}s")
```

#### Use the JobService directly
```python
from pathlib import Path
from homelab_subs.services.job_service import JobService

service = JobService(enable_monitoring=True, enable_db_logging=True)

# Optional: progress callback
def on_progress(percent, segments):
    print(f"Progress: {percent:.1f}% ({segments} segments)")

srt_path = service.generate_subtitles(
    video_path=Path("input.mp4"),
    output_path=Path("input.en.srt"),
    lang="en",
    task="transcribe",
    progress_callback=on_progress,
)

print(f"Subtitles saved to: {srt_path}")
```

## Performance Considerations

### Resource Overhead
- **Metric collection**: ~0.1% CPU overhead per 2-second sample
- **Database writes**: Minimal I/O, batched inserts
- **Memory**: <10 MB for monitoring thread
- **Disk**: ~1-5 KB per job, ~100 bytes per metric sample

### GPU Monitoring
GPU monitoring requires:
- NVIDIA GPU with CUDA support
- Linux operating system
- `nvidia-ml-py` package installed
- NVIDIA drivers with NVML support

If GPU monitoring is unavailable, the tool continues without GPU metrics.

## Privacy & Data Storage

### What is stored?
- **Job metadata**: Video file paths, output paths, model settings, timestamps
- **Performance metrics**: CPU, memory, disk, GPU usage (system-level, not video content)
- **No video content**: Only file paths and metadata, never video frames or audio

### Local storage only
All data is stored locally on your machine in:
```
~/.homelab-subs/logs.db
```

No data is transmitted externally. The database is a standard SQLite file that you can:
- Back up manually
- Delete at any time
- Move to another location
- Query with any SQLite client

### Cleanup
```bash
# Delete all logs
rm ~/.homelab-subs/logs.db

# Or keep some logs
sqlite3 ~/.homelab-subs/logs.db "DELETE FROM jobs WHERE started_at < datetime('now', '-30 days');"
```

## Future Web UI Integration

The monitoring system is designed for easy Web UI integration:

### Ready-to-use API
The `DatabaseLogger` class provides all necessary query methods:
```python
# For job list pages
jobs = db.get_recent_jobs(limit=50, status="completed")

# For job detail pages
job = db.get_job(job_id)
metrics = db.get_job_metrics(job_id)

# For dashboard statistics
stats = db.get_statistics()
```

### Visualization-ready data
Metrics are time-series data perfect for charts:
```python
metrics = db.get_job_metrics(job_id)
timestamps = [m.timestamp for m in metrics]
cpu_values = [m.cpu_percent for m in metrics]
memory_values = [m.memory_used_mb for m in metrics]

# Use with matplotlib, plotly, or any charting library
```

### REST API skeleton
Example Flask endpoint:
```python
from flask import Flask, jsonify
from homelab_subs.core.db_logger import DatabaseLogger

app = Flask(__name__)
db = DatabaseLogger()

@app.route('/api/jobs')
def get_jobs():
    jobs = db.get_recent_jobs(limit=20)
    return jsonify([{
        'id': j.job_id,
        'video': j.video_path,
        'status': j.status,
        'duration': j.duration_seconds,
        'cpu_avg': j.cpu_avg,
    } for j in jobs])

@app.route('/api/jobs/<job_id>')
def get_job(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    
    metrics = db.get_job_metrics(job_id)
    return jsonify({
        'job': job.__dict__,
        'metrics': [m.__dict__ for m in metrics],
    })
```

## Troubleshooting

### "History command requires monitoring dependencies"
This message now means database logging was disabled when jobs were created.
Re-run jobs without `--no-db-logging` (default behavior) so history has data
to read. If monitoring dependencies are missing you'll still see warnings, but
history only needs the SQLite database.

### GPU metrics not appearing
GPU monitoring requires:
1. NVIDIA GPU hardware
2. Linux OS
3. Install: `pip install nvidia-ml-py`
4. NVIDIA drivers with NVML support

If not available, the tool continues without GPU metrics.

### Database locked errors
SQLite databases can't be accessed by multiple processes simultaneously. Ensure only one instance of the tool is running, or use different database paths:
```bash
subsvc generate video1.mp4 --db-path /tmp/db1.db &
subsvc generate video2.mp4 --db-path /tmp/db2.db &
```

### Viewing database directly
Use any SQLite client:
```bash
# Command-line
sqlite3 ~/.homelab-subs/logs.db "SELECT * FROM jobs;"

# Or use GUI tools like DB Browser for SQLite
```

## Examples

### Monitor a long-running job
```bash
# Start generation with monitoring
subsvc generate long_video.mp4

# In another terminal, check progress
subsvc history --limit 1

# After completion, view detailed metrics
subsvc history --job-id <job-id>
```

### Compare model performance
```bash
# Test different models
subsvc generate video.mp4 --model tiny --output tiny.srt
subsvc generate video.mp4 --model base --output base.srt
subsvc generate video.mp4 --model small --output small.srt

# Compare results
subsvc history --limit 3
```

### Export job data
```bash
# Export to CSV
sqlite3 -header -csv ~/.homelab-subs/logs.db \
  "SELECT job_id, video_path, duration_seconds, cpu_avg, memory_avg_mb FROM jobs;" \
  > jobs_export.csv

# Export metrics for specific job
sqlite3 -header -csv ~/.homelab-subs/logs.db \
  "SELECT * FROM metrics WHERE job_id='<job-id>';" \
  > job_metrics.csv
```

## Configuration

### Environment Variables

```bash
# Custom default database location
export HOMELAB_SUBS_DB_PATH=/custom/path/logs.db

# Disable monitoring by default
export HOMELAB_SUBS_NO_MONITORING=1
```

(Note: These are examples - actual environment variable support would need to be implemented)

## Architecture

```
┌─────────────┐        ┌────────────────────────────┐
│   CLI or    │        │        JobService          │
│  API Client │───────▶│  - Configures PipelineRunner│
└─────────────┘        │  - Exposes history helpers │
              └──────────────┬─────────────┘
                       │
                       v
              ┌────────────────────────────┐
              │      PipelineRunner        │
              │  ┌──────────────────────┐  │
              │  │  Core Pipeline       │  │
              │  │  (audio → text → srt)│  │
              │  └──────────────────────┘  │
              │          │                 │
              │  ┌───────┴─────────────┐   │
              │  │ Metrics thread (2s) │   │
              │  └───────┬─────────────┘   │
              └──────────┼─────────────────┘
                      │
                      v
              ┌────────────────────────────┐
              │    PerformanceMonitor      │
              │  psutil / nvidia-ml-py     │
              └──────────┬─────────────────┘
                      │
                      v
              ┌────────────────────────────┐
              │       DatabaseLogger        │
              │   SQLite jobs + metrics     │
              └────────────────────────────┘
```

## Contributing

When adding new metrics:
1. Update `SystemMetrics` dataclass in `performance.py`
2. Update database schema in `db_logger.py`
3. Update `PipelineRunner` to collect new metrics
4. Add to history display in `cli.py`
5. Update this documentation

## License

Same as the main project.
