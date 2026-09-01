/**
 * Client-safe copy of the Role type — lib/server/rbac.ts can't be
 * imported from client components (it pulls in the `pg` driver, a
 * Node-only module that breaks bundling for the browser/edge, the same
 * class of issue that broke middleware.ts earlier). Keep in sync with
 * lib/server/rbac.ts's Role/ROLE_RANK by hand; it's a 4-value union that
 * only changes if the role matrix itself changes.
 */
export type Role = 'viewer' | 'operator' | 'approver' | 'admin';

const ROLE_RANK: Record<Role, number> = { viewer: 0, operator: 1, approver: 2, admin: 3 };

export function roleAtLeast(role: Role | null | undefined, required: Role): boolean {
  if (!role) return false;
  return ROLE_RANK[role] >= ROLE_RANK[required];
}
