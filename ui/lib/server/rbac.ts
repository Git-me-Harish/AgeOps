import { getPgPool } from './db';

export type Role = 'viewer' | 'operator' | 'approver' | 'admin';

const ROLE_RANK: Record<Role, number> = { viewer: 0, operator: 1, approver: 2, admin: 3 };

export function roleAtLeast(role: Role | null | undefined, required: Role): boolean {
  if (!role) return false;
  return ROLE_RANK[role] >= ROLE_RANK[required];
}

/**
 * Called from auth.ts's `events.createUser` — fires exactly once, the
 * moment @auth/pg-adapter inserts a brand-new row into `users`. Whoever's
 * email matches INITIAL_ADMIN_EMAIL becomes admin; everyone else starts
 * as viewer until an existing admin promotes them from Admin > Users.
 * (Bootstrap is keyed on email, not GitHub login/username, because the
 * standard Auth.js adapter schema this project uses only captures
 * name/email/image/emailVerified — no username column — and email is
 * something the project owner already knows to put in an env var.)
 */
export async function bootstrapRoleForNewUser(userId: number, email: string | null | undefined): Promise<void> {
  const pool = getPgPool();
  if (!pool) return;
  const bootstrapEmail = process.env.INITIAL_ADMIN_EMAIL?.toLowerCase().trim();
  const isBootstrapAdmin = Boolean(bootstrapEmail) && email?.toLowerCase().trim() === bootstrapEmail;
  await pool.query(
    `INSERT INTO user_roles (user_id, role) VALUES ($1, $2)
     ON CONFLICT (user_id) DO NOTHING`,
    [userId, isBootstrapAdmin ? 'admin' : 'viewer'],
  );
}

export async function getRoleForUserId(userId: number | string): Promise<Role> {
  const pool = getPgPool();
  if (!pool) return 'viewer';
  const result = await pool.query<{ role: Role }>(
    'SELECT role FROM user_roles WHERE user_id = $1',
    [Number(userId)],
  );
  return result.rows[0]?.role ?? 'viewer';
}

export async function setRoleForUserId(userId: number, role: Role, grantedBy: number): Promise<void> {
  const pool = getPgPool();
  if (!pool) throw new Error('No database configured — cannot manage roles');
  await pool.query(
    `INSERT INTO user_roles (user_id, role, granted_by, updated_at)
     VALUES ($1, $2, $3, NOW())
     ON CONFLICT (user_id) DO UPDATE SET role = $2, granted_by = $3, updated_at = NOW()`,
    [userId, role, grantedBy],
  );
}

export interface UserRoleRow {
  user_id: number;
  name: string | null;
  email: string | null;
  role: Role;
  updated_at: string;
}

export async function listUsersWithRoles(): Promise<UserRoleRow[]> {
  const pool = getPgPool();
  if (!pool) return [];
  const result = await pool.query<UserRoleRow>(
    `SELECT u.id AS user_id, u.name, u.email, COALESCE(r.role, 'viewer') AS role, r.updated_at
     FROM users u
     LEFT JOIN user_roles r ON r.user_id = u.id
     ORDER BY u.id ASC`,
  );
  return result.rows;
}
