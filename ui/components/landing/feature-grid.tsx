import { Activity, GitBranch, ShieldCheck, Workflow, type LucideIcon } from 'lucide-react';

interface Feature {
  Icon: LucideIcon;
  title: string;
  description: string;
}

// Real capabilities this build actually has, not marketing filler — each
// maps to a page in the app: Live Serving/Drift & Health (monitoring),
// Admin > Users + the OPA policy editor (governance), Model Registry's
// lineage graph + packaging (lineage), Pipeline Builder + the orchestrator
// graph (orchestration).
const FEATURES: Feature[] = [
  {
    Icon: Activity,
    title: 'Real-time drift detection',
    description:
      'Live Prometheus + Loki-backed monitoring with a closed retraining loop that fires automatically on real drift, not a scheduled guess.',
  },
  {
    Icon: ShieldCheck,
    title: 'Governed by design',
    description:
      'Role-based access from viewer through admin, OPA-evaluated promotion gates, and an immutable audit log of every policy decision.',
  },
  {
    Icon: GitBranch,
    title: 'Full model lineage',
    description:
      'Every model traces back through training, evaluation, and its signed, SBOM-scanned image — dataset to deployment, one real graph.',
  },
  {
    Icon: Workflow,
    title: 'Multi-agent orchestration',
    description:
      'Six specialized agents coordinated by one orchestrator, with human-in-the-loop approval gates before anything reaches production.',
  },
];

/** Abstract icon tiles are monochrome-on-accent by design — distinct colors per tile would read as status semantics (ok/warn/crit) on a page that isn't showing system state. */
export function FeatureGrid() {
  return (
    <div
      style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr',
        gap: 14, height: '100%', width: '100%',
      }}
    >
      {FEATURES.map((f) => (
        <div key={f.title} className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%' }}>
            <div
              style={{
                width: 40, height: 40, borderRadius: 10, background: 'var(--accent-soft)',
                color: 'var(--accent-strong)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <f.Icon size={20} strokeWidth={2} />
            </div>
            <div style={{ fontWeight: 600, fontSize: 13.5 }}>{f.title}</div>
            <p className="text-muted" style={{ fontSize: 12, lineHeight: 1.5, margin: 0 }}>
              {f.description}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
