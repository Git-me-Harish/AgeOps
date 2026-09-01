'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { StatusPill } from '@/components/ui/status-pill';

interface UserRow {
  user_id: number;
  name: string | null;
  email: string | null;
  role: 'viewer' | 'operator' | 'approver' | 'admin';
  updated_at: string | null;
}

const ROLES: UserRow['role'][] = ['viewer', 'operator', 'approver', 'admin'];
const ROLE_SEVERITY: Record<UserRow['role'], 'unknown' | 'accent' | 'warn' | 'crit'> = {
  viewer: 'unknown',
  operator: 'accent',
  approver: 'warn',
  admin: 'crit',
};

/**
 * Real CRUD against the real `user_roles` table (scripts/migrations/
 * versions/0007_auth_rbac.py) — admin-only, enforced server-side by
 * app/api/admin/users/route.ts, not just hidden client-side.
 */
export default function AdminUsersPage() {
  const qc = useQueryClient();
  const usersQuery = useQuery({
    queryKey: ['admin-users'],
    queryFn: async () => {
      const res = await fetch('/api/admin/users');
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Failed to load users');
      return (await res.json()) as { users: UserRow[] };
    },
  });

  const mutation = useMutation({
    mutationFn: async ({ userId, role }: { userId: number; role: UserRow['role'] }) => {
      const res = await fetch('/api/admin/users', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ user_id: userId, role }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Failed to update role');
      return res.json();
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin-users'] }),
  });

  return (
    <>
      <TopBar title="Admin — Users & Roles" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <Card>
          <CardHeader title="Role matrix" />
          <CardBody className="row" style={{ gap: 16, flexWrap: 'wrap', fontSize: 12.5 }}>
            <span><b>viewer</b> — read-only</span>
            <span><b>operator</b> — + start workflows, trigger retraining, decide RL recs</span>
            <span><b>approver</b> — + approve workflows, promote models, rollback</span>
            <span><b>admin</b> — + edit OPA policy, manage roles</span>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Users" />
          <CardBody className="scroll-x">
            {usersQuery.isError && (
              <p style={{ color: 'var(--crit)', fontSize: 12.5 }}>{(usersQuery.error as Error).message}</p>
            )}
            <table className="data-table">
              <thead><tr><th>User</th><th>Email</th><th>Role</th><th>Updated</th></tr></thead>
              <tbody>
                {(usersQuery.data?.users ?? []).map((u) => (
                  <tr key={u.user_id}>
                    <td>{u.name ?? `user #${u.user_id}`}</td>
                    <td className="mono text-muted">{u.email ?? '—'}</td>
                    <td>
                      <div className="row" style={{ gap: 8 }}>
                        <StatusPill severity={ROLE_SEVERITY[u.role]} label={u.role} />
                        <select
                          value={u.role}
                          onChange={(e) =>
                            mutation.mutate({ userId: u.user_id, role: e.target.value as UserRow['role'] })
                          }
                          style={{
                            padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border-strong)',
                            background: 'var(--bg)', color: 'var(--text)', fontSize: 12,
                          }}
                        >
                          {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                        </select>
                      </div>
                    </td>
                    <td className="text-faint">{u.updated_at ? new Date(u.updated_at).toLocaleString() : '—'}</td>
                  </tr>
                ))}
                {(usersQuery.data?.users ?? []).length === 0 && !usersQuery.isLoading && (
                  <tr><td colSpan={4} className="text-faint">No users have signed in yet.</td></tr>
                )}
              </tbody>
            </table>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
