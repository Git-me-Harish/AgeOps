import { Pool } from 'pg';

/**
 * Shared Postgres pool for the Next.js server — real database sessions
 * (@auth/pg-adapter, auth.ts) and RBAC (lib/server/rbac.ts) both use this.
 * Same Postgres instance the Python side's DATABASE_URL points at; this is
 * the Node-side connection string (plain postgres://, no +asyncpg suffix).
 * Returns null when unset so local dev without a Postgres degrades to
 * JWT-only sessions (see auth.ts) instead of crashing on import.
 */
let pool: Pool | null | undefined;

export function getPgPool(): Pool | null {
  if (pool !== undefined) return pool;
  if (!process.env.DATABASE_URL) {
    pool = null;
    return pool;
  }
  pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 10 });
  return pool;
}
