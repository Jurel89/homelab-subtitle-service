import { useQuery } from '@tanstack/react-query';
import {
  RefreshCw,
  Clock,
  CheckCircle,
  XCircle,
  BarChart3,
  TrendingUp,
  Activity,
  Zap,
  Timer,
  FileVideo,
} from 'lucide-react';
import { jobsApi, systemApi } from '@/lib/api';
import type { Job, JobStatistics, QueueStatus } from '@/types';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';

export default function KPIsPage() {
  // Fetch all jobs for statistics
  const { data: jobsData, isLoading: jobsLoading } = useQuery({
    queryKey: ['jobs', 'all'],
    queryFn: () => jobsApi.list({ page: 1, page_size: 1000 }),
  });

  // Fetch job statistics from API
  const { data: statsData } = useQuery({
    queryKey: ['stats'],
    queryFn: jobsApi.getStatistics,
  });

  // Fetch queue status
  const { data: queueStatus, isLoading: queueLoading } = useQuery<QueueStatus>({
    queryKey: ['queue', 'status'],
    queryFn: systemApi.queueStatus,
    refetchInterval: 10000,
  });

  const jobs: Job[] = jobsData?.jobs || [];
  const stats: JobStatistics = statsData || {
    total_jobs: 0,
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
  };
  const isLoading = jobsLoading || queueLoading;

  // Calculate success rate
  const completedAndFailed = stats.completed + stats.failed;
  const successRate =
    completedAndFailed > 0 ? Math.round((stats.completed / completedAndFailed) * 100) : 0;

  // Calculate average processing time (for completed jobs)
  const completedJobs = jobs.filter(
    (j) => j.status === 'done' && j.started_at && j.completed_at
  );
  const avgProcessingTime =
    completedJobs.length > 0
      ? completedJobs.reduce((acc, job) => {
          const start = new Date(job.started_at!).getTime();
          const end = new Date(job.completed_at!).getTime();
          return acc + (end - start);
        }, 0) /
        completedJobs.length /
        1000 // in seconds
      : 0;

  // Calculate jobs by type
  const jobsByType = jobs.reduce(
    (acc, job) => {
      acc[job.type] = (acc[job.type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  // Recent activity (last 24h)
  const last24h = new Date(Date.now() - 24 * 60 * 60 * 1000);
  const recentJobs = jobs.filter((j) => new Date(j.created_at) > last24h);
  const recentCompleted = recentJobs.filter((j) => j.status === 'done').length;
  const recentFailed = recentJobs.filter((j) => j.status === 'failed').length;

  // Jobs per hour (last 24h)
  const jobsPerHour = recentJobs.length / 24;

  // Format duration
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  // Format relative time
  const formatRelativeTime = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return date.toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">KPIs & Analytics</h1>
        <p className="text-muted-foreground mt-1">Monitor system performance and job statistics</p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileVideo className="h-4 w-4 text-accent" />
              Total Jobs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.total_jobs}</div>
            <p className="text-xs text-muted-foreground mt-1">All time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-success" />
              Success Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{successRate}%</div>
            <Progress value={successRate} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Timer className="h-4 w-4 text-warning" />
              Avg Processing Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{formatDuration(avgProcessingTime)}</div>
            <p className="text-xs text-muted-foreground mt-1">Per job</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4 text-accent" />
              Throughput
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{jobsPerHour.toFixed(1)}</div>
            <p className="text-xs text-muted-foreground mt-1">Jobs/hour (24h avg)</p>
          </CardContent>
        </Card>
      </div>

      {/* Queue Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Queue Status
            </CardTitle>
            <CardDescription>Current job queue state</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Queued</span>
                <span className="font-medium">{stats.pending}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Running</span>
                <span className="font-medium">{stats.running}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Active Workers</span>
                <span className="font-medium">{queueStatus?.workers?.length || 0}</span>
              </div>
              <div className="pt-4 border-t">
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Queue Health</span>
                  <span
                    className={
                      queueStatus?.workers && queueStatus.workers.length > 0
                        ? 'text-success'
                        : 'text-warning'
                    }
                  >
                    {queueStatus?.workers && queueStatus.workers.length > 0
                      ? 'Healthy'
                      : 'No Workers'}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Last 24 Hours
            </CardTitle>
            <CardDescription>Recent activity summary</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileVideo className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Jobs Submitted</span>
                </div>
                <span className="font-medium">{recentJobs.length}</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-success" />
                  <span className="text-sm">Completed</span>
                </div>
                <span className="font-medium text-success">{recentCompleted}</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-destructive" />
                  <span className="text-sm">Failed</span>
                </div>
                <span className="font-medium text-destructive">{recentFailed}</span>
              </div>
              {recentJobs.length > 0 && recentJobs[0] && (
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <span>Last Job</span>
                    <span>{formatRelativeTime(new Date(recentJobs[0].created_at))}</span>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Job Status Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Job Status Distribution
          </CardTitle>
          <CardDescription>Breakdown of all jobs by status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              {
                label: 'Completed',
                value: stats.completed,
                color: 'bg-success',
                textColor: 'text-success',
              },
              {
                label: 'Running',
                value: stats.running,
                color: 'bg-accent',
                textColor: 'text-accent',
              },
              {
                label: 'Pending',
                value: stats.pending,
                color: 'bg-warning',
                textColor: 'text-warning',
              },
              {
                label: 'Failed',
                value: stats.failed,
                color: 'bg-destructive',
                textColor: 'text-destructive',
              },
              {
                label: 'Cancelled',
                value: stats.cancelled,
                color: 'bg-muted',
                textColor: 'text-muted-foreground',
              },
            ].map((item) => {
              const percentage =
                stats.total_jobs > 0 ? (item.value / stats.total_jobs) * 100 : 0;
              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span>{item.label}</span>
                    <span className={`font-medium ${item.textColor}`}>
                      {item.value} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full ${item.color} transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Jobs by Type */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Jobs by Type
          </CardTitle>
          <CardDescription>Distribution of job types</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(jobsByType).map(([type, count]) => {
              const percentage = stats.total_jobs > 0 ? (count / stats.total_jobs) * 100 : 0;
              return (
                <div key={type} className="p-4 rounded-lg bg-muted/50 text-center">
                  <div className="text-2xl font-bold">{count}</div>
                  <div className="text-sm text-muted-foreground capitalize">
                    {type.replace('_', ' ')}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">{percentage.toFixed(1)}%</div>
                </div>
              );
            })}
            {Object.keys(jobsByType).length === 0 && (
              <div className="col-span-full text-center text-muted-foreground py-8">
                No jobs processed yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
