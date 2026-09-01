'use client';

import { useEventStream } from '@/lib/ws';
import { StatusPill } from '@/components/ui/status-pill';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { UserMenu } from '@/components/layout/user-menu';

export function TopBar({ title }: { title: string }) {
  const { connected } = useEventStream(() => {});
  return (
    <div
      className="row"
      style={{
        justifyContent: 'space-between',
        padding: '14px 24px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-elevated)',
      }}
    >
      <h1 style={{ fontSize: 17 }}>{title}</h1>
      <div className="row" style={{ gap: 14 }}>
        <StatusPill severity={connected ? 'ok' : 'unknown'} label={connected ? 'Live' : 'Reconnecting…'} />
        <ThemeToggle />
        <UserMenu />
      </div>
    </div>
  );
}
