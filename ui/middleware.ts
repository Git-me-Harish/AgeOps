/**
 * Route protection (Phase 6 §6.1 auth). Deliberately does NOT import
 * auth.ts here — Next.js middleware runs in the Edge runtime, which
 * cannot load the `pg` driver auth.ts now pulls in for real database
 * sessions (@auth/pg-adapter). Importing it broke the app outright
 * ("The edge runtime does not support Node.js 'crypto' module") the
 * first time this was wired up — found by actually running it, not by
 * inspection.
 *
 * The fix is the pattern Auth.js itself recommends for this exact
 * conflict: middleware does a cheap, Edge-safe check for the presence of
 * the session cookie only (no DB round-trip, no cryptographic
 * verification) purely to redirect an obviously-signed-out browser to
 * sign-in. The real, authoritative check — a real Postgres lookup of the
 * session row, plus the RBAC role lookup — happens server-side in Node.js
 * runtime code that CAN import `pg`: the gateway route
 * (app/api/gateway/[service]/[...path]/route.ts) and
 * app/(app)/layout.tsx both call auth() for real. A forged or stale
 * cookie that passes this middleware check still gets rejected there.
 */
import { NextResponse, type NextRequest } from 'next/server';

const AUTH_CONFIGURED = Boolean(process.env.GITHUB_CLIENT_ID);
const PUBLIC_PATHS = ['/api/auth', '/api/healthz', '/login'];
const SESSION_COOKIE_NAMES = ['authjs.session-token', '__Secure-authjs.session-token'];

export default function middleware(req: NextRequest) {
  if (!AUTH_CONFIGURED) return NextResponse.next();
  if (PUBLIC_PATHS.some((p) => req.nextUrl.pathname.startsWith(p))) return NextResponse.next();

  const hasSessionCookie = SESSION_COOKIE_NAMES.some((name) => req.cookies.has(name));
  if (!hasSessionCookie) {
    const signInUrl = new URL('/login', req.url);
    signInUrl.searchParams.set('callbackUrl', req.nextUrl.pathname);
    return NextResponse.redirect(signInUrl);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
