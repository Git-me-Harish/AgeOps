/**
 * Renders only when GitHub OAuth is genuinely unconfigured (see
 * auth.ts's authIsConfigured) — makes the unauthenticated state visible
 * instead of silently pretending a session exists.
 */
export function DevModeBanner({ authConfigured }: { authConfigured: boolean }) {
  if (authConfigured) return null;
  return (
    <div className="dev-banner">
      Dev mode — GitHub OAuth not configured (GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET unset). Running
      unauthenticated; every mutating action below hits the real backend regardless of identity.
    </div>
  );
}
