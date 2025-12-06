import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, RefreshCw, XCircle, RotateCcw, Download } from 'lucide-react';
import { jobsApi } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/Card';
import { StatusBadge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { formatDate, formatDuration } from '@/lib/utils';
import type { Job, JobLogsResponse } from '@/types';

export function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<JobLogsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    if (!id) return;
    setIsLoading(true);
    setError(null);
    try {
      const [jobRes, logsRes] = await Promise.all([
        jobsApi.get(id),
        jobsApi.getLogs(id),
      ]);
      setJob(jobRes);
      setLogs(logsRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh if running
    const interval = setInterval(() => {
      if (job?.status === 'running' || job?.status === 'pending') {
        fetchData();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [id, job?.status]);

  const handleCancel = async () => {
    if (!id) return;
    try {
      await jobsApi.cancel(id);
      fetchData();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handleRetry = async () => {
    if (!id) return;
    try {
      await jobsApi.retry(id);
      fetchData();
    } catch (err) {
      console.error('Failed to retry job:', err);
    }
  };

  if (isLoading && !job) {
    return (
      <div className="flex justify-center py-8">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-destructive/10 p-4 text-destructive">
        {error}
      </div>
    );
  }

  if (!job) {
    return <div>Job not found</div>;
  }

  const duration = job.started_at && job.completed_at
    ? (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">Job Details</h1>
            <p className="text-sm text-muted-foreground">{job.id}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={fetchData}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          {(job.status === 'pending' || job.status === 'running') && (
            <Button variant="destructive" size="sm" onClick={handleCancel}>
              <XCircle className="mr-2 h-4 w-4" />
              Cancel
            </Button>
          )}
          {(job.status === 'failed' || job.status === 'cancelled') && (
            <Button size="sm" onClick={handleRetry}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          )}
        </div>
      </div>

      {/* Status & Progress */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Status</CardTitle>
              <CardDescription>Current job progress and status</CardDescription>
            </div>
            <StatusBadge status={job.status} />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="mb-2 flex items-center justify-between text-sm">
                <span>Progress</span>
                <span>{job.progress}%</span>
              </div>
              <Progress value={job.progress} />
            </div>
            <div className="text-sm text-muted-foreground">
              Stage: <span className="capitalize">{job.stage.replace(/_/g, ' ')}</span>
            </div>
            {job.error_message && (
              <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                {job.error_message}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Job Configuration */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Type</dt>
                <dd className="capitalize">{job.type}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Model</dt>
                <dd>{job.model_size || 'Default'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Compute Type</dt>
                <dd>{job.compute_type || 'Default'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Source Language</dt>
                <dd>{job.source_language || 'Auto'}</dd>
              </div>
              {job.target_language && (
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Target Language</dt>
                  <dd>{job.target_language}</dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Timing</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Created</dt>
                <dd>{formatDate(job.created_at)}</dd>
              </div>
              {job.started_at && (
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Started</dt>
                  <dd>{formatDate(job.started_at)}</dd>
                </div>
              )}
              {job.completed_at && (
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Completed</dt>
                  <dd>{formatDate(job.completed_at)}</dd>
                </div>
              )}
              {duration !== null && (
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Duration</dt>
                  <dd>{formatDuration(duration)}</dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>
      </div>

      {/* Files */}
      <Card>
        <CardHeader>
          <CardTitle>Files</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="space-y-3 text-sm">
            <div>
              <dt className="text-muted-foreground">Input</dt>
              <dd className="mt-1 break-all font-mono text-xs">{job.input_path}</dd>
            </div>
            {job.output_path && (
              <div>
                <dt className="flex items-center justify-between text-muted-foreground">
                  Output
                  {job.status === 'done' && (
                    <Button variant="ghost" size="sm">
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>
                  )}
                </dt>
                <dd className="mt-1 break-all font-mono text-xs">{job.output_path}</dd>
              </div>
            )}
            {job.reference_path && (
              <div>
                <dt className="text-muted-foreground">Reference</dt>
                <dd className="mt-1 break-all font-mono text-xs">{job.reference_path}</dd>
              </div>
            )}
          </dl>
        </CardContent>
      </Card>

      {/* Logs */}
      <Card>
        <CardHeader>
          <CardTitle>Logs</CardTitle>
        </CardHeader>
        <CardContent>
          {logs?.logs ? (
            <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-xs">
              {logs.logs}
            </pre>
          ) : (
            <p className="text-muted-foreground">No logs available</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
