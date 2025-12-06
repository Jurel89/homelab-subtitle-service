import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, Play, XCircle, Eye, RotateCcw, Trash2 } from 'lucide-react';
import { jobsApi, systemApi } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { StatusBadge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { formatDate } from '@/lib/utils';
import type { Job, JobStatistics, HealthStatus } from '@/types';

export function DashboardPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [stats, setStats] = useState<JobStatistics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [jobsRes, statsRes, healthRes] = await Promise.all([
        jobsApi.list({ page, page_size: 20, status: statusFilter || undefined, type: typeFilter || undefined }),
        jobsApi.getStatistics(),
        systemApi.health(),
      ]);
      setJobs(jobsRes.jobs);
      setHasMore(jobsRes.has_more);
      setStats(statsRes);
      setHealth(healthRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [page, statusFilter, typeFilter]);

  const handleCancel = async (id: string) => {
    try {
      await jobsApi.cancel(id);
      fetchData();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await jobsApi.retry(id);
      fetchData();
    } catch (err) {
      console.error('Failed to retry job:', err);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this job?')) return;
    try {
      await jobsApi.delete(id);
      fetchData();
    } catch (err) {
      console.error('Failed to delete job:', err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">Monitor and manage your subtitle jobs</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={fetchData} disabled={isLoading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Link to="/new">
            <Button size="sm">
              <Play className="mr-2 h-4 w-4" />
              New Job
            </Button>
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Jobs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total_jobs ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Running
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-warning">{stats?.running ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-success">{stats?.completed ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${health?.status === 'healthy' ? 'text-success' : 'text-warning'}`}>
              {health?.status ?? 'Unknown'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex items-center space-x-4 py-4">
          <select
            className="rounded-md border bg-background px-3 py-2 text-sm"
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="done">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <select
            className="rounded-md border bg-background px-3 py-2 text-sm"
            value={typeFilter}
            onChange={(e) => { setTypeFilter(e.target.value); setPage(1); }}
          >
            <option value="">All Types</option>
            <option value="transcribe">Transcribe</option>
            <option value="translate">Translate</option>
            <option value="sync">Sync</option>
            <option value="compare">Compare</option>
          </select>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <div className="rounded-md bg-destructive/10 p-4 text-destructive">
          {error}
        </div>
      )}

      {/* Jobs Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Jobs</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && jobs.length === 0 ? (
            <div className="flex justify-center py-8">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : jobs.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              No jobs found. Create a new job to get started.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b text-left text-sm text-muted-foreground">
                    <th className="pb-3 pr-4">Status</th>
                    <th className="pb-3 pr-4">Type</th>
                    <th className="pb-3 pr-4">Input</th>
                    <th className="pb-3 pr-4">Progress</th>
                    <th className="pb-3 pr-4">Created</th>
                    <th className="pb-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id} className="border-b last:border-0">
                      <td className="py-3 pr-4">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="py-3 pr-4 capitalize">{job.type}</td>
                      <td className="max-w-[200px] truncate py-3 pr-4 text-sm" title={job.input_path}>
                        {job.input_path.split('/').pop()}
                      </td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center space-x-2">
                          <Progress value={job.progress} className="w-20" />
                          <span className="text-sm text-muted-foreground">{job.progress}%</span>
                        </div>
                      </td>
                      <td className="py-3 pr-4 text-sm text-muted-foreground">
                        {formatDate(job.created_at)}
                      </td>
                      <td className="py-3">
                        <div className="flex items-center space-x-1">
                          <Link to={`/jobs/${job.id}`}>
                            <Button variant="ghost" size="icon" title="View details">
                              <Eye className="h-4 w-4" />
                            </Button>
                          </Link>
                          {(job.status === 'pending' || job.status === 'running') && (
                            <Button
                              variant="ghost"
                              size="icon"
                              title="Cancel"
                              onClick={() => handleCancel(job.id)}
                            >
                              <XCircle className="h-4 w-4" />
                            </Button>
                          )}
                          {(job.status === 'failed' || job.status === 'cancelled') && (
                            <Button
                              variant="ghost"
                              size="icon"
                              title="Retry"
                              onClick={() => handleRetry(job.id)}
                            >
                              <RotateCcw className="h-4 w-4" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Delete"
                            onClick={() => handleDelete(job.id)}
                          >
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {(page > 1 || hasMore) && (
            <div className="mt-4 flex justify-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                Previous
              </Button>
              <span className="flex items-center px-4 text-sm text-muted-foreground">
                Page {page}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => p + 1)}
                disabled={!hasMore}
              >
                Next
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
