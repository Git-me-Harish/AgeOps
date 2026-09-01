'use client';

import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { signIn, signOut, useSession } from 'next-auth/react';
import Link from 'next/link';
import { LogOut, User as UserIcon } from 'lucide-react';
import { StatusPill } from '@/components/ui/status-pill';
import { Button } from '@/components/ui/button';
import type { Role } from '@/lib/rbac-client';

const ROLE_SEVERITY: Record<Role, 'unknown' | 'accent' | 'warn' | 'crit'> = {
  viewer: 'unknown', operator: 'accent', approver: 'warn', admin: 'crit',
};

/**
 * Real signed-in identity + sign-out. Works whether or not GitHub OAuth is
 * configured: with no provider set up (dev mode, already surfaced by
 * DevModeBanner), this just shows a "Sign in" button that — if
 * clicked — hits NextAuth's own "no providers configured" response,
 * which is an honest reflection of the actual state rather than hiding
 * the control.
 */
export function UserMenu() {
  const { data: session, status } = useSession();

  if (status === 'loading') return <div style={{ width: 32, height: 32 }} />;

  if (!session?.user) {
    return (
      <Button size="sm" variant="secondary" onClick={() => signIn('github')}>
        Sign in
      </Button>
    );
  }

  const role = ((session.user as { role?: Role }).role ?? 'viewer') as Role;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          className="row"
          style={{ gap: 8, background: 'none', border: 'none', cursor: 'pointer', padding: 2, borderRadius: 999 }}
          aria-label="Account menu"
        >
          {session.user.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={session.user.image}
              alt=""
              width={28}
              height={28}
              style={{ borderRadius: '50%', border: '1px solid var(--border-strong)' }}
            />
          ) : (
            <div
              className="row"
              style={{
                width: 28, height: 28, borderRadius: '50%', background: 'var(--accent-soft)',
                color: 'var(--accent-strong)', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <UserIcon size={14} />
            </div>
          )}
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={8}
          style={{
            minWidth: 220, background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-md)', padding: 8, zIndex: 20,
          }}
        >
          <div style={{ padding: '6px 8px 10px' }}>
            <div style={{ fontWeight: 600, fontSize: 13 }}>{session.user.name ?? 'Signed in'}</div>
            <div className="text-faint mono" style={{ fontSize: 11 }}>{session.user.email}</div>
            <div style={{ marginTop: 6 }}>
              <StatusPill severity={ROLE_SEVERITY[role]} label={role} />
            </div>
          </div>
          <DropdownMenu.Separator style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />
          <DropdownMenu.Item asChild>
            <Link
              href="/profile"
              className="row"
              style={{ gap: 8, padding: '8px', borderRadius: 6, fontSize: 13, textDecoration: 'none', color: 'var(--text)' }}
            >
              <UserIcon size={14} /> My profile
            </Link>
          </DropdownMenu.Item>
          <DropdownMenu.Item asChild>
            <button
              onClick={() => signOut({ callbackUrl: '/' })}
              className="row"
              style={{
                gap: 8, padding: '8px', borderRadius: 6, fontSize: 13, width: '100%',
                background: 'none', border: 'none', cursor: 'pointer', color: 'var(--crit)',
              }}
            >
              <LogOut size={14} /> Sign out
            </button>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
