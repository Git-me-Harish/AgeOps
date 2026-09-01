/**
 * NextAuth.js v5, GitHub OAuth provider (multi-agent-mlops-v2-plan.md
 * §6.1) — now backed by real, revocable database sessions
 * (@auth/pg-adapter against the same Postgres the Python side uses;
 * schema is scripts/migrations/versions/0007_auth_rbac.py) instead of a
 * bare JWT. Deleting a row from the real `sessions` table force-logs-out
 * that browser on its next request — a JWT-only session can't be killed
 * before it expires.
 *
 * Deliberately still degrades gracefully in two independent ways:
 *   - No GITHUB_CLIENT_ID/SECRET → no provider, dev-mode banner (unchanged
 *     from the original Phase 6 build).
 *   - No DATABASE_URL → no adapter, falls back to JWT sessions (so a
 *     from-scratch local dev environment with no Postgres configured yet
 *     doesn't crash on import) — see lib/server/db.ts.
 */
import NextAuth from 'next-auth';
import GitHub from 'next-auth/providers/github';
import PostgresAdapter from '@auth/pg-adapter';
import { getPgPool } from './lib/server/db';
import { bootstrapRoleForNewUser, getRoleForUserId, type Role } from './lib/server/rbac';

export const authIsConfigured = Boolean(process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET);

const pgPool = getPgPool();

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: pgPool ? PostgresAdapter(pgPool) : undefined,
  providers: authIsConfigured
    ? [
        GitHub({
          clientId: process.env.GITHUB_CLIENT_ID,
          clientSecret: process.env.GITHUB_CLIENT_SECRET,
        }),
      ]
    : [],
  secret: process.env.NEXTAUTH_SECRET,
  pages: { signIn: '/login' },
  // Database strategy requires the adapter (real Postgres); falls back to
  // jwt only when no DATABASE_URL is configured at all.
  session: { strategy: pgPool ? 'database' : 'jwt' },
  events: {
    async createUser({ user }) {
      if (user.id) await bootstrapRoleForNewUser(Number(user.id), user.email);
    },
  },
  callbacks: {
    // database-strategy session callback receives { session, user }
    async session({ session, user, token }) {
      if (session.user) {
        const userId = user?.id ?? (token?.sub as string | undefined);
        (session.user as { role?: Role; id?: string }).role = userId ? await getRoleForUserId(userId) : 'viewer';
        (session.user as { role?: Role; id?: string }).id = userId;
      }
      return session;
    },
  },
});
