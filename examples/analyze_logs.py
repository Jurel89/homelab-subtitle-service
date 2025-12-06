#!/usr/bin/env python3
"""
Example script showing how to programmatically analyze JSON logs
from homelab-subtitle-service.

This demonstrates various ways to extract useful information from the logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_logs(log_file: Path) -> List[Dict[str, Any]]:
    """Load and parse JSON logs from file."""
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    logs = []
    with open(log_file) as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
                continue
    
    return logs


def get_all_jobs(logs: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique job IDs from logs."""
    job_ids = set()
    for log in logs:
        if "job_id" in log:
            job_ids.add(log["job_id"])
    return sorted(job_ids)


def get_job_logs(logs: List[Dict[str, Any]], job_id: str) -> List[Dict[str, Any]]:
    """Get all log entries for a specific job."""
    return [log for log in logs if log.get("job_id") == job_id]


def analyze_job_performance(logs: List[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
    """Analyze performance metrics for a specific job."""
    job_logs = get_job_logs(logs, job_id)
    
    if not job_logs:
        return {"error": "Job not found"}
    
    result = {
        "job_id": job_id,
        "video_file": None,
        "stages": {},
        "total_duration": 0.0,
        "status": "unknown",
        "errors": []
    }
    
    for log in job_logs:
        # Extract video file name
        if log.get("video_file"):
            result["video_file"] = log["video_file"]
        
        # Extract stage durations
        if "stage" in log and "duration" in log:
            stage = log["stage"]
            duration = log["duration"]
            result["stages"][stage] = duration
            result["total_duration"] += duration
        
        # Check for completion
        if "Successfully generated subtitles" in log.get("message", ""):
            result["status"] = "completed"
        
        # Collect errors
        if log.get("level") == "ERROR":
            result["errors"].append({
                "timestamp": log.get("timestamp"),
                "message": log.get("message"),
                "function": log.get("function")
            })
    
    if result["errors"]:
        result["status"] = "failed"
    elif result["status"] == "unknown":
        result["status"] = "running"
    
    return result


def get_stage_statistics(logs: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    """Get statistics for a specific stage across all jobs."""
    durations = []
    
    for log in logs:
        if log.get("stage") == stage and "duration" in log:
            durations.append(log["duration"])
    
    if not durations:
        return {"error": f"No data found for stage: {stage}"}
    
    durations.sort()
    count = len(durations)
    
    return {
        "stage": stage,
        "count": count,
        "min": durations[0],
        "max": durations[-1],
        "avg": sum(durations) / count,
        "median": durations[count // 2],
        "total": sum(durations)
    }


def get_error_summary(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all errors from logs."""
    errors = []
    
    for log in logs:
        if log.get("level") in ["ERROR", "CRITICAL"]:
            errors.append({
                "timestamp": log.get("timestamp"),
                "job_id": log.get("job_id", "N/A"),
                "stage": log.get("stage", "N/A"),
                "message": log.get("message"),
                "module": log.get("module"),
                "function": log.get("function")
            })
    
    return errors


def print_job_summary(logs: List[Dict[str, Any]]) -> None:
    """Print a summary of all jobs."""
    job_ids = get_all_jobs(logs)
    
    print(f"\n{'='*80}")
    print(f"JOB SUMMARY - {len(job_ids)} jobs found")
    print(f"{'='*80}\n")
    
    for job_id in job_ids:
        analysis = analyze_job_performance(logs, job_id)
        
        status_emoji = {
            "completed": "✅",
            "running": "⏳",
            "failed": "❌",
            "unknown": "❓"
        }
        
        emoji = status_emoji.get(analysis["status"], "❓")
        
        print(f"{emoji} Job: {job_id}")
        print(f"   Video: {analysis['video_file'] or 'Unknown'}")
        print(f"   Status: {analysis['status']}")
        print(f"   Total Duration: {analysis['total_duration']:.2f}s")
        
        if analysis['stages']:
            print("   Stages:")
            for stage, duration in analysis['stages'].items():
                print(f"      • {stage}: {duration:.2f}s")
        
        if analysis['errors']:
            print(f"   Errors: {len(analysis['errors'])}")
            for error in analysis['errors']:
                print(f"      • {error['message']}")
        
        print()


def print_stage_statistics(logs: List[Dict[str, Any]]) -> None:
    """Print statistics for all stages."""
    # Find all unique stages
    stages = set()
    for log in logs:
        if "stage" in log and "duration" in log:
            stages.add(log["stage"])
    
    if not stages:
        print("No stage data found in logs.")
        return
    
    print(f"\n{'='*80}")
    print("STAGE PERFORMANCE STATISTICS")
    print(f"{'='*80}\n")
    
    for stage in sorted(stages):
        stats = get_stage_statistics(logs, stage)
        
        print(f"📊 {stage.replace('_', ' ').title()}")
        print(f"   Count: {stats['count']} runs")
        print(f"   Min: {stats['min']:.2f}s")
        print(f"   Max: {stats['max']:.2f}s")
        print(f"   Avg: {stats['avg']:.2f}s")
        print(f"   Median: {stats['median']:.2f}s")
        print(f"   Total: {stats['total']:.2f}s")
        print()


def print_error_summary(logs: List[Dict[str, Any]]) -> None:
    """Print summary of all errors."""
    errors = get_error_summary(logs)
    
    if not errors:
        print("\n✅ No errors found in logs!")
        return
    
    print(f"\n{'='*80}")
    print(f"ERROR SUMMARY - {len(errors)} errors found")
    print(f"{'='*80}\n")
    
    for error in errors:
        print(f"❌ [{error['timestamp']}] Job: {error['job_id']}")
        print(f"   Stage: {error['stage']}")
        print(f"   Location: {error['module']}.{error['function']}")
        print(f"   Message: {error['message']}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file.json> [options]")
        print()
        print("Options:")
        print("  --jobs      Show summary of all jobs")
        print("  --stages    Show performance statistics by stage")
        print("  --errors    Show all errors")
        print("  --all       Show all reports (default)")
        print("  --job ID    Show detailed analysis for specific job")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    logs = load_logs(log_file)
    
    print(f"Loaded {len(logs)} log entries from {log_file}")
    
    # Parse options
    options = sys.argv[2:] if len(sys.argv) > 2 else ["--all"]
    
    if "--all" in options or "--jobs" in options:
        print_job_summary(logs)
    
    if "--all" in options or "--stages" in options:
        print_stage_statistics(logs)
    
    if "--all" in options or "--errors" in options:
        print_error_summary(logs)
    
    if "--job" in options:
        idx = options.index("--job")
        if idx + 1 < len(options):
            job_id = options[idx + 1]
            analysis = analyze_job_performance(logs, job_id)
            
            print(f"\n{'='*80}")
            print(f"DETAILED ANALYSIS: Job {job_id}")
            print(f"{'='*80}\n")
            
            print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
