/**
 * Generic backend-forwarding proxy (BFF gateway) — the browser only ever
 * talks to this Next.js server, never the 4 internal FastAPI services
 * directly. `service` picks the internal origin (see
 * lib/server/gateway-config.ts); everything after it is forwarded
 * unchanged onto that origin's real route — e.g.
 *   GET /api/gateway/registry/registry/v1/lineage/fraud-model/3
 *   -> GET {REGISTRY_SERVER_INTERNAL_URL}/registry/v1/lineage/fraud-model/3
 * so every path here maps 1:1 onto a route that genuinely exists in
 * mcp_servers/*.py — no invented endpoints.
 *
 * Three real, server-side-enforced controls sit in front of every
 * forwarded call (never trust a disabled button as the only defense):
 *   1. Auth — once GitHub OAuth is configured, every non-GET requires a
 *      real session (auth.ts, database-backed via @auth/pg-adapter).
 *   2. RBAC — the caller's role (lib/server/rbac.ts, from the real
 *      user_roles table) must meet the route's minimum
 *      (lib/server/authorization.ts) or the request is rejected with 403.
 *   3. Rate limiting — Redis-backed, correct across the mlops-ui
 *      Deployment's multiple replicas (lib/server/rate-limit.ts).
 *
 * With no GitHub OAuth app configured, the whole product runs in the
 * explicit dev-mode banner state described in
 * multi-agent-mlops-v2-plan.md's Phase 6 auth scoping — auth and RBAC
 * checks are skipped (nothing to check against), but rate limiting still
 * applies (keyed by IP), since that control has nothing to do with identity.
 */
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/auth';
import { GATEWAY_SERVICES, isGatewayService } from '@/lib/server/gateway-config';
import { requiredRoleFor } from '@/lib/server/authorization';
import { roleAtLeast, type Role } from '@/lib/server/rbac';
import { checkRateLimit, MUTATE_LIMIT, READ_LIMIT } from '@/lib/server/rate-limit';

const AUTH_CONFIGURED = Boolean(process.env.GITHUB_CLIENT_ID);

function clientIp(req: NextRequest): string {
  return req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? 'unknown';
}

async function forward(req: NextRequest, params: { service: string; path: string[] }) {
  const { service, path } = params;
  if (!isGatewayService(service)) {
    return NextResponse.json({ detail: `Unknown gateway service '${service}'` }, { status: 404 });
  }

  if (service === 'prometheus' && req.method !== 'GET') {
    return NextResponse.json({ detail: 'prometheus gateway is read-only' }, { status: 405 });
  }

  const isMutation = req.method !== 'GET' && req.method !== 'HEAD';
  let rateLimitKey = `ip:${clientIp(req)}`;

  if (AUTH_CONFIGURED) {
    const session = await auth();
    if (isMutation && !session) {
      return NextResponse.json({ detail: 'Authentication required' }, { status: 401 });
    }
    if (session?.user) {
      const userId = (session.user as { id?: string }).id;
      if (userId) rateLimitKey = `user:${userId}`;
      if (isMutation) {
        const callerRole = (session.user as { role?: Role }).role ?? 'viewer';
        const { role: needed, reason } = requiredRoleFor(req.method, req.nextUrl.pathname);
        if (!roleAtLeast(callerRole, needed)) {
          return NextResponse.json(
            { detail: `Forbidden — ${reason} (you have '${callerRole}', need '${needed}' or higher)` },
            { status: 403 },
          );
        }
      }
    }
  }

  const { count, windowSeconds } = isMutation ? MUTATE_LIMIT : READ_LIMIT;
  const rl = await checkRateLimit(rateLimitKey, isMutation ? 'mutate' : 'read', count, windowSeconds);
  if (!rl.allowed) {
    return NextResponse.json(
      { detail: `Rate limit exceeded (${rl.limit} requests / ${windowSeconds}s)` },
      { status: 429, headers: { 'Retry-After': String(windowSeconds) } },
    );
  }

  const base = GATEWAY_SERVICES[service];
  const target = new URL(`${base}/${path.join('/')}`);
  target.search = req.nextUrl.search;

  const init: RequestInit = {
    method: req.method,
    headers: { 'content-type': req.headers.get('content-type') ?? 'application/json' },
  };
  if (isMutation) {
    init.body = await req.text();
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (exc) {
    return NextResponse.json(
      { detail: `Upstream service '${service}' unreachable: ${String(exc)}` },
      { status: 502 },
    );
  }

  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { 'content-type': upstream.headers.get('content-type') ?? 'application/json' },
  });
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ service: string; path: string[] }> }) {
  return forward(req, await ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ service: string; path: string[] }> }) {
  return forward(req, await ctx.params);
}
export async function PUT(req: NextRequest, ctx: { params: Promise<{ service: string; path: string[] }> }) {
  return forward(req, await ctx.params);
}
export async function DELETE(req: NextRequest, ctx: { params: Promise<{ service: string; path: string[] }> }) {
  return forward(req, await ctx.params);
}
