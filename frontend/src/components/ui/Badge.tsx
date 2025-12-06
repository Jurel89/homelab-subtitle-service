import { cn } from '@/lib/utils';
import type { JobStatus } from '@/types';

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'destructive' | 'outline' | 'secondary';
  className?: string;
  children: React.ReactNode;
}

export function Badge({ variant = 'default', className, children }: BadgeProps) {
  const variants = {
    default: 'bg-primary text-primary-foreground',
    success: 'bg-success text-white',
    warning: 'bg-warning text-white',
    destructive: 'bg-destructive text-destructive-foreground',
    outline: 'border border-input bg-transparent',
    secondary: 'bg-secondary text-secondary-foreground',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  );
}

interface StatusBadgeProps {
  status: JobStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config: Record<JobStatus, { variant: BadgeProps['variant']; label: string }> = {
    pending: { variant: 'secondary', label: 'Pending' },
    queued: { variant: 'secondary', label: 'Queued' },
    running: { variant: 'warning', label: 'Running' },
    done: { variant: 'success', label: 'Completed' },
    failed: { variant: 'destructive', label: 'Failed' },
    cancelled: { variant: 'outline', label: 'Cancelled' },
  };

  const { variant, label } = config[status] || { variant: 'default', label: status };

  return (
    <Badge variant={variant} className={className}>
      {label}
    </Badge>
  );
}

// Helper function to get variant from status
export function statusToVariant(status: JobStatus): BadgeProps['variant'] {
  const mapping: Record<JobStatus, BadgeProps['variant']> = {
    pending: 'secondary',
    queued: 'secondary',
    running: 'warning',
    done: 'success',
    failed: 'destructive',
    cancelled: 'outline',
  };
  return mapping[status] || 'default';
}
