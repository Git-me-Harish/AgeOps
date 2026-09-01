import clsx from 'clsx';

export type Severity = 'ok' | 'warn' | 'crit' | 'unknown' | 'accent';

const LABELS: Record<Severity, string> = {
  ok: 'Healthy',
  warn: 'Warning',
  crit: 'Critical',
  unknown: 'Unknown',
  accent: 'Info',
};

export function StatusPill({ severity, label }: { severity: Severity; label?: string }) {
  return (
    <span className={clsx('badge', `badge-${severity}`)}>
      <span className="badge-dot" />
      {label ?? LABELS[severity]}
    </span>
  );
}

/** Maps the real backend alert_level / status strings onto a Severity — no fabricated in-between states. */
export function severityFromAlertLevel(level: string | null | undefined): Severity {
  switch (level) {
    case 'critical':
    case 'error':
      return 'crit';
    case 'warning':
      return 'warn';
    case 'none':
      return 'ok';
    default:
      return 'unknown';
  }
}

export function severityFromPodStatus(status: string | null | undefined): Severity {
  switch (status) {
    case 'online':
      return 'ok';
    case 'degraded':
      return 'warn';
    default:
      return 'unknown';
  }
}
