#!/usr/bin/env python3
"""
Simple web-based log viewer for homelab-subtitle-service.

This is a basic example showing how to parse and display the structured
JSON logs in a web interface. It uses Flask for simplicity.

Usage:
    pip install flask
    python log_viewer.py --log-file /path/to/logs.json
    
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

try:
    from flask import Flask, render_template_string, jsonify
except ImportError:
    print("Flask is required. Install it with: pip install flask")
    exit(1)

app = Flask(__name__)

# Global variable to store log file path
LOG_FILE_PATH: Path = Path("subsvc.log")


def parse_log_file(log_path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Parse the JSON log file and return recent entries.
    
    Parameters
    ----------
    log_path : Path
        Path to the log file
    limit : int
        Maximum number of entries to return (most recent)
    
    Returns
    -------
    List[Dict[str, Any]]
        List of log entries
    """
    if not log_path.exists():
        return []
    
    entries = []
    with open(log_path) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    # Return most recent entries
    return entries[-limit:]


def get_jobs_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract summary information about jobs from log entries.
    
    Parameters
    ----------
    entries : List[Dict[str, Any]]
        Log entries
    
    Returns
    -------
    Dict[str, Any]
        Summary information by job_id
    """
    jobs = {}
    
    for entry in entries:
        job_id = entry.get("job_id")
        if not job_id:
            continue
        
        if job_id not in jobs:
            jobs[job_id] = {
                "job_id": job_id,
                "video_file": entry.get("video_file", "Unknown"),
                "stages": {},
                "status": "running",
                "started_at": entry.get("timestamp"),
                "errors": []
            }
        
        # Track stage progress
        stage = entry.get("stage")
        if stage:
            if stage not in jobs[job_id]["stages"]:
                jobs[job_id]["stages"][stage] = {"status": "started"}
            
            if "duration" in entry:
                jobs[job_id]["stages"][stage]["duration"] = entry["duration"]
                jobs[job_id]["stages"][stage]["status"] = "completed"
        
        # Track errors
        if entry.get("level") == "ERROR":
            jobs[job_id]["errors"].append(entry.get("message", "Unknown error"))
            jobs[job_id]["status"] = "failed"
        
        # Check if completed
        if "Successfully generated subtitles" in entry.get("message", ""):
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = entry.get("timestamp")
    
    return jobs


# HTML Template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Subtitle Service - Log Viewer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 32px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            font-weight: normal;
            margin-bottom: 8px;
        }
        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .jobs-grid {
            display: grid;
            gap: 20px;
            margin-bottom: 30px;
        }
        .job-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .job-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .job-id {
            font-size: 12px;
            color: #666;
            font-family: monospace;
        }
        .status {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status.completed {
            background: #d4edda;
            color: #155724;
        }
        .status.running {
            background: #fff3cd;
            color: #856404;
        }
        .status.failed {
            background: #f8d7da;
            color: #721c24;
        }
        .stages {
            margin-top: 15px;
        }
        .stage {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .stage:last-child {
            border-bottom: none;
        }
        .stage-name {
            color: #666;
            text-transform: capitalize;
        }
        .stage-duration {
            color: #999;
            font-size: 14px;
        }
        .errors {
            margin-top: 15px;
            padding: 10px;
            background: #f8d7da;
            border-radius: 4px;
            color: #721c24;
            font-size: 14px;
        }
        .logs-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .logs-section h2 {
            margin-bottom: 15px;
            color: #333;
        }
        .log-entry {
            padding: 8px;
            margin-bottom: 4px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
            border-left: 3px solid #ddd;
        }
        .log-entry.DEBUG {
            background: #f8f9fa;
            border-left-color: #6c757d;
        }
        .log-entry.INFO {
            background: #d4edda;
            border-left-color: #28a745;
        }
        .log-entry.WARNING {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        .log-entry.ERROR {
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        .log-timestamp {
            color: #666;
            margin-right: 10px;
        }
        .log-level {
            font-weight: bold;
            margin-right: 10px;
        }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
    </style>
    <script>
        function refreshData() {
            location.reload();
        }
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</head>
<body>
    <div class="container">
        <h1>🎬 Subtitle Service - Log Viewer</h1>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Jobs</h3>
                <div class="value">{{ stats.total }}</div>
            </div>
            <div class="stat-card">
                <h3>Completed</h3>
                <div class="value">{{ stats.completed }}</div>
            </div>
            <div class="stat-card">
                <h3>Running</h3>
                <div class="value">{{ stats.running }}</div>
            </div>
            <div class="stat-card">
                <h3>Failed</h3>
                <div class="value">{{ stats.failed }}</div>
            </div>
        </div>
        
        <h2>Jobs</h2>
        <div class="jobs-grid">
            {% for job in jobs %}
            <div class="job-card">
                <div class="job-header">
                    <div>
                        <div class="job-title">{{ job.video_file }}</div>
                        <div class="job-id">Job ID: {{ job.job_id }}</div>
                    </div>
                    <div class="status {{ job.status }}">{{ job.status }}</div>
                </div>
                
                {% if job.stages %}
                <div class="stages">
                    {% for stage_name, stage_info in job.stages.items() %}
                    <div class="stage">
                        <span class="stage-name">{{ stage_name.replace('_', ' ') }}</span>
                        <span class="stage-duration">
                            {% if stage_info.duration %}
                                ✓ {{ "%.2f"|format(stage_info.duration) }}s
                            {% else %}
                                In progress...
                            {% endif %}
                        </span>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if job.errors %}
                <div class="errors">
                    <strong>Errors:</strong>
                    {% for error in job.errors %}
                    <div>• {{ error }}</div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="logs-section">
            <h2>Recent Logs ({{ logs|length }} entries)</h2>
            {% for log in logs[-50:]|reverse %}
            <div class="log-entry {{ log.level }}">
                <span class="log-timestamp">{{ log.timestamp }}</span>
                <span class="log-level">{{ log.level }}</span>
                <span>{{ log.message }}</span>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    """Main dashboard page."""
    entries = parse_log_file(LOG_FILE_PATH)
    jobs = get_jobs_summary(entries)
    
    # Calculate stats
    stats = {
        "total": len(jobs),
        "completed": sum(1 for j in jobs.values() if j["status"] == "completed"),
        "running": sum(1 for j in jobs.values() if j["status"] == "running"),
        "failed": sum(1 for j in jobs.values() if j["status"] == "failed"),
    }
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        jobs=list(jobs.values()),
        logs=entries,
        stats=stats
    )


@app.route("/api/logs")
def api_logs():
    """API endpoint to get logs as JSON."""
    entries = parse_log_file(LOG_FILE_PATH, limit=100)
    return jsonify(entries)


@app.route("/api/jobs")
def api_jobs():
    """API endpoint to get jobs summary as JSON."""
    entries = parse_log_file(LOG_FILE_PATH)
    jobs = get_jobs_summary(entries)
    return jsonify(list(jobs.values()))


def main():
    parser = argparse.ArgumentParser(
        description="Web-based log viewer for homelab-subtitle-service"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("subsvc.log"),
        help="Path to the JSON log file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    
    args = parser.parse_args()
    
    global LOG_FILE_PATH
    LOG_FILE_PATH = args.log_file
    
    print(f"Starting log viewer...")
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"API Logs: http://{args.host}:{args.port}/api/logs")
    print(f"API Jobs: http://{args.host}:{args.port}/api/jobs")
    print("\nPress Ctrl+C to stop")
    
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
