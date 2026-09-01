import { redirect } from 'next/navigation';
import { auth, authIsConfigured } from '@/auth';
import { SignInButton } from './sign-in-button';
import { FeatureGrid } from '@/components/landing/feature-grid';
import { ThemeToggle } from '@/components/ui/theme-toggle';

// Real technologies this platform actually runs on (MLflow tracking/registry,
// Kubernetes/KServe deployment, Prometheus/Loki monitoring, Redis event bus +
// rate limiting, Postgres for lineage/RBAC/sessions, GitHub OAuth) — plain
// text badges, not fabricated logo images.
const TECH_STACK = ['MLflow', 'Kubernetes', 'Prometheus', 'Redis', 'PostgreSQL', 'GitHub'];

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ callbackUrl?: string; error?: string }>;
}) {
  const session = await auth();
  if (session?.user) redirect('/command-center');

  const { callbackUrl, error } = await searchParams;

  return (
    <div
      className="stack"
      style={{ height: '100vh', width: '100vw', overflow: 'hidden', background: 'var(--bg)' }}
    >
      {/* Nav */}
      <div
        className="row"
        style={{
          justifyContent: 'space-between', padding: '16px 40px', flexShrink: 0,
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div className="row" style={{ gap: 8 }}>
          <div style={{ fontWeight: 700, fontSize: 15 }}>Multi-Agent MLOps</div>
          <span className="text-faint" style={{ fontSize: 11.5, marginTop: 1 }}>Platform v2</span>
        </div>
        <ThemeToggle />
      </div>

      {/* Hero — fills the rest of the viewport, no scroll. `alignItems: stretch`
          overrides the shared .row class's default `align-items: center`. */}
      <div
        className="row"
        style={{ flex: 1, minHeight: 0, alignItems: 'stretch', padding: '0 40px' }}
      >
        {/* Copy + sign-in + (for now, empty) product preview */}
        <div
          className="stack"
          style={{ flex: '1 1 0', minWidth: 320, justifyContent: 'flex-start', gap: 16, paddingRight: 40, paddingTop: 36, paddingBottom: 24 }}
        >
          <div className="row" style={{ gap: 6, alignSelf: 'flex-start' }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)' }} />
            <span className="text-muted" style={{ fontSize: 12, fontWeight: 500 }}>
              real-time multi-agent MLOps platform
            </span>
          </div>

          <h1 style={{ fontSize: 34, lineHeight: 1.15, maxWidth: 480, margin: 0 }}>
            Your ML pipeline,<br />
            <span className="text-muted" style={{ fontWeight: 500 }}>orchestrated end to end.</span>
          </h1>

          <p className="text-muted" style={{ fontSize: 13.5, maxWidth: 420, lineHeight: 1.55, margin: 0 }}>
            Data ingestion, training, evaluation, deployment, and drift monitoring — six real agents
            coordinated by one orchestrator, with a closed retraining loop that fires on real drift.
          </p>

          <div className="stack" style={{ gap: 8, maxWidth: 320 }}>
            {!authIsConfigured ? (
              <p className="text-faint" style={{ fontSize: 12.5, margin: 0 }}>
                GitHub OAuth isn&apos;t configured in this environment — the app runs unauthenticated
                (dev mode) until GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET are set.
              </p>
            ) : (
              <>
                {error && (
                  <p style={{ color: 'var(--crit)', fontSize: 12.5, margin: 0 }}>
                    {error === 'AccessDenied'
                      ? 'Access denied — that GitHub account could not sign in.'
                      : 'Sign-in failed — please try again.'}
                  </p>
                )}
                <SignInButton callbackUrl={callbackUrl ?? '/command-center'} />
                <p className="text-faint" style={{ fontSize: 11.5, margin: 0 }}>
                  First time signing in creates your account automatically as read-only (viewer). An
                  admin promotes you from there.
                </p>
              </>
            )}
          </div>

          <div className="stack" style={{ gap: 8, marginTop: 4 }}>
            <span className="text-faint" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Built on
            </span>
            <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
              {TECH_STACK.map((t) => (
                <span
                  key={t}
                  className="text-muted mono"
                  style={{
                    fontSize: 11.5, padding: '4px 10px', borderRadius: 999,
                    border: '1px solid var(--border)', background: 'var(--bg-inset)',
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Product preview — intentionally empty for now; real screenshots
              go here later. */}
          <div className="landing-mockup-frame" style={{ flex: 1, minHeight: 90, display: 'flex', flexDirection: 'column', marginTop: 4 }}>
            <div className="landing-mockup-chrome">
              <span className="landing-mockup-dot" />
              <span className="landing-mockup-dot" />
              <span className="landing-mockup-dot" />
              <span className="text-faint mono" style={{ fontSize: 11, marginLeft: 8 }}>
                command-center — pipeline overview
              </span>
            </div>
            <div style={{ flex: 1 }} />
          </div>
        </div>

        {/* Feature bento grid */}
        <div className="stack" style={{ flex: '1.15 1 0', minWidth: 0, justifyContent: 'center' }}>
          <div style={{ height: 'min(560px, 100%)', padding: '24px 0' }}>
            <FeatureGrid />
          </div>
        </div>
      </div>
    </div>
  );
}
