'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';

const NAV = [
  { href: '/command-center', label: 'Command Center' },
  { href: '/pipeline-builder', label: 'Pipeline Builder' },
  { href: '/experiment-lab', label: 'Experiment Lab' },
  { href: '/model-registry', label: 'Model Registry' },
  { href: '/live-serving', label: 'Live Serving' },
  { href: '/drift-health', label: 'Drift & Health' },
  { href: '/agent-intelligence', label: 'Agent Intelligence' },
  { href: '/security', label: 'Security & Compliance' },
];

export function Sidebar({ isAdmin = false }: { isAdmin?: boolean }) {
  const pathname = usePathname();
  const nav = isAdmin ? [...NAV, { href: '/admin/users', label: 'Admin — Users' }] : NAV;
  return (
    <nav
      style={{
        width: 232,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        background: 'var(--bg-elevated)',
        padding: '16px 10px',
      }}
    >
      <div style={{ padding: '0 10px 18px' }}>
        <div style={{ fontWeight: 700, fontSize: 15 }}>Multi-Agent MLOps</div>
        <div className="text-faint" style={{ fontSize: 11.5 }}>
          Platform v2
        </div>
      </div>
      <div className="stack" style={{ gap: 2 }}>
        {nav.map((item) => {
          const active = pathname?.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              style={{
                padding: '8px 10px',
                borderRadius: 8,
                fontSize: 13,
                fontWeight: active ? 600 : 500,
                color: active ? 'var(--accent-strong)' : 'var(--text-muted)',
                background: active ? 'var(--accent-soft)' : 'transparent',
                textDecoration: 'none',
              }}
              className={clsx(!active && 'nav-link')}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
