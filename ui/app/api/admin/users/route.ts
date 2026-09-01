import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/auth';
import { listUsersWithRoles, roleAtLeast, setRoleForUserId, type Role } from '@/lib/server/rbac';

const VALID_ROLES: Role[] = ['viewer', 'operator', 'approver', 'admin'];

async function requireAdmin() {
  const session = await auth();
  const role = (session?.user as { role?: Role } | undefined)?.role;
  if (!roleAtLeast(role, 'admin')) return null;
  return session;
}

export async function GET() {
  const session = await requireAdmin();
  if (!session) return NextResponse.json({ detail: 'Forbidden — admin only' }, { status: 403 });
  return NextResponse.json({ users: await listUsersWithRoles() });
}

export async function POST(req: NextRequest) {
  const session = await requireAdmin();
  if (!session) return NextResponse.json({ detail: 'Forbidden — admin only' }, { status: 403 });

  const body = await req.json();
  const userId = Number(body.user_id);
  const role = body.role as Role;
  if (!userId || !VALID_ROLES.includes(role)) {
    return NextResponse.json({ detail: 'user_id and a valid role are required' }, { status: 422 });
  }
  const grantedBy = Number((session.user as { id?: string }).id);
  await setRoleForUserId(userId, role, grantedBy);
  return NextResponse.json({ user_id: userId, role });
}
