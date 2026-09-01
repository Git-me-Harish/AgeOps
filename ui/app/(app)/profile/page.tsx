import { auth, authIsConfigured } from '@/auth';
import { redirect } from 'next/navigation';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { StatusPill } from '@/components/ui/status-pill';
import { SignOutButton } from '@/components/layout/sign-out-button';
import type { Role } from '@/lib/rbac-client';

const ROLE_SEVERITY: Record<Role, 'unknown' | 'accent' | 'warn' | 'crit'> = {
  viewer: 'unknown', operator: 'accent', approver: 'warn', admin: 'crit',
};

const ROLE_GRANTS: Record<Role, string[]> = {
  viewer: ['Read every page — dashboards, runs, models, drift reports, audit log'],
  operator: ['Everything viewer can, plus: start workflows, trigger retraining, accept/reject RL recommendations'],
  approver: ['Everything operator can, plus: approve/reject workflows, promote models, roll back deployments'],
  admin: ['Everything approver can, plus: edit the OPA policy, manage other users’ roles'],
};

/** Real identity + real role, read directly from the session — no client round-trip needed for the initial render. */
export default async function ProfilePage() {
  if (!authIsConfigured) {
    return (
      <>
        <TopBar title="My Profile" />
        <div style={{ padding: 24 }}>
          <Card>
            <CardBody>
              <p className="text-faint" style={{ fontSize: 13 }}>
                GitHub OAuth isn&apos;t configured in this environment — the app is running unauthenticated
                (dev mode), so there&apos;s no session to show a profile for.
              </p>
            </CardBody>
          </Card>
        </div>
      </>
    );
  }

  const session = await auth();
  if (!session?.user) redirect('/api/auth/signin?callbackUrl=/profile');

  const role = ((session.user as { role?: Role }).role ?? 'viewer') as Role;

  return (
    <>
      <TopBar title="My Profile" />
      <div className="stack" style={{ padding: 24, gap: 20, maxWidth: 560 }}>
        <Card>
          <CardBody className="row" style={{ gap: 16, alignItems: 'center' }}>
            {session.user.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={session.user.image}
                alt=""
                width={56}
                height={56}
                style={{ borderRadius: '50%', border: '1px solid var(--border-strong)' }}
              />
            ) : (
              <div style={{ width: 56, height: 56, borderRadius: '50%', background: 'var(--accent-soft)' }} />
            )}
            <div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>{session.user.name}</div>
              <div className="text-muted mono" style={{ fontSize: 12.5 }}>{session.user.email}</div>
              <div style={{ marginTop: 8 }}>
                <StatusPill severity={ROLE_SEVERITY[role]} label={role} />
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title={`What '${role}' can do`} />
          <CardBody className="stack" style={{ gap: 6 }}>
            {ROLE_GRANTS[role].map((g, i) => (
              <p key={i} style={{ fontSize: 12.5, margin: 0 }}>{g}</p>
            ))}
            {role !== 'admin' && (
              <p className="text-faint" style={{ fontSize: 11.5, marginTop: 6 }}>
                Need a different role? Ask an admin — roles are changed from Admin → Users, never
                self-service, so nobody can grant themselves elevated access.
              </p>
            )}
          </CardBody>
        </Card>

        <div>
          <SignOutButton />
        </div>
      </div>
    </>
  );
}
