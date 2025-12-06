import { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { RefreshCw, Download, Trash2, Search, Pause, Play } from 'lucide-react';
import { jobsApi } from '@/lib/api';
import type { Job } from '@/types';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge, statusToVariant } from '@/components/ui/Badge';

interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
}

const levelColors: Record<string, string> = {
  DEBUG: 'text-muted-foreground',
  INFO: 'text-foreground',
  WARNING: 'text-warning',
  ERROR: 'text-destructive',
};

export default function LogsPage() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [filterLevel, setFilterLevel] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Fetch active/recent jobs for filter
  const { data: jobsData } = useQuery({
    queryKey: ['jobs', 'recent'],
    queryFn: () => jobsApi.list({ page: 1, page_size: 20 }),
  });

  const jobs: Job[] = jobsData?.jobs || [];

  // Fetch logs for selected job
  const {
    data: logsData,
    isLoading: logsLoading,
    refetch: refetchLogs,
  } = useQuery({
    queryKey: ['job', selectedJobId, 'logs'],
    queryFn: () => (selectedJobId ? jobsApi.getLogs(selectedJobId) : null),
    enabled: !!selectedJobId,
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Parse logs from job logs response
  const parseLogs = (logsText: string | null): LogEntry[] => {
    if (!logsText) return [];

    const lines = logsText.split('\n').filter((line) => line.trim());
    return lines.map((line) => {
      // Try to parse structured log: [timestamp] [LEVEL] message
      const match = line.match(/^\[(.*?)\]\s*\[(.*?)\]\s*(.*)$/);
      if (match) {
        return {
          timestamp: match[1],
          level: match[2] as LogEntry['level'],
          message: match[3],
        };
      }
      // Fallback: plain text
      return {
        timestamp: new Date().toISOString(),
        level: 'INFO' as const,
        message: line,
      };
    });
  };

  const logs = parseLogs(logsData?.logs || null);

  // Filter logs
  const filteredLogs = logs.filter((log) => {
    if (filterLevel !== 'all' && log.level !== filterLevel) return false;
    if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase()))
      return false;
    return true;
  });

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, autoScroll]);

  const handleDownload = () => {
    const logText = filteredLogs
      .map((log) => `[${log.timestamp}] [${log.level}] ${log.message}`)
      .join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${selectedJobId || 'all'}-${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClear = () => {
    setSearchQuery('');
    setFilterLevel('all');
  };

  // Find selected job for display
  const selectedJob = jobs.find((j) => j.id === selectedJobId);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Logs</h1>
          <p className="text-muted-foreground mt-1">View real-time job logs</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => refetchLogs()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownload}
            disabled={filteredLogs.length === 0}
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium mb-1">Job</label>
              <select
                value={selectedJobId || ''}
                onChange={(e) => setSelectedJobId(e.target.value || null)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="">Select a job...</option>
                {jobs.map((job) => (
                  <option key={job.id} value={job.id}>
                    {job.id.slice(0, 8)} - {job.input_path.split('/').pop()} ({job.status})
                  </option>
                ))}
              </select>
            </div>
            <div className="w-[150px]">
              <label className="block text-sm font-medium mb-1">Level</label>
              <select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="all">All Levels</option>
                <option value="DEBUG">Debug</option>
                <option value="INFO">Info</option>
                <option value="WARNING">Warning</option>
                <option value="ERROR">Error</option>
              </select>
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium mb-1">Search</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search logs..."
                  className="pl-9"
                />
              </div>
            </div>
            <div className="flex items-end gap-2">
              <Button variant="ghost" size="sm" onClick={handleClear}>
                <Trash2 className="h-4 w-4 mr-2" />
                Clear Filters
              </Button>
              <Button
                variant={autoScroll ? 'default' : 'outline'}
                size="sm"
                onClick={() => setAutoScroll(!autoScroll)}
              >
                {autoScroll ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Auto-scroll On
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Auto-scroll Off
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Selected Job Info */}
      {selectedJob && (
        <Card>
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Job: {selectedJob.id.slice(0, 8)}</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant={statusToVariant(selectedJob.status)}>{selectedJob.status}</Badge>
                <span className="text-sm text-muted-foreground">
                  {selectedJob.input_path.split('/').pop()}
                </span>
              </div>
            </div>
          </CardHeader>
        </Card>
      )}

      {/* Logs Viewer */}
      <Card className="h-[500px] flex flex-col">
        <CardHeader className="py-3 border-b flex-shrink-0">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              Logs ({filteredLogs.length})
            </CardTitle>
            {logsLoading && <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />}
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-auto p-0">
          {!selectedJobId ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Select a job to view logs
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              {logs.length === 0 ? 'No logs available' : 'No logs match your filters'}
            </div>
          ) : (
            <div className="font-mono text-xs">
              {filteredLogs.map((log, index) => (
                <div
                  key={`${log.timestamp}-${index}`}
                  className="flex border-b border-border/50 hover:bg-muted/50"
                >
                  <div className="w-48 flex-shrink-0 px-3 py-1.5 text-muted-foreground border-r border-border/50">
                    {log.timestamp}
                  </div>
                  <div
                    className={`w-20 flex-shrink-0 px-3 py-1.5 font-semibold border-r border-border/50 ${levelColors[log.level]}`}
                  >
                    {log.level}
                  </div>
                  <div className="flex-1 px-3 py-1.5 break-all whitespace-pre-wrap">
                    {log.message}
                  </div>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
