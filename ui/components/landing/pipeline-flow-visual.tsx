'use client';

import { useEffect, useState } from 'react';
import { Activity, Database, Rocket, ShieldCheck, Sparkles, Workflow } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface FlowNode {
  id: string;
  label: string;
  Icon: LucideIcon;
  x: number;
  y: number;
}

const NODES: FlowNode[] = [
  { id: 'orchestrator', label: 'Orchestrator', Icon: Workflow, x: 360, y: 60 },
  { id: 'data', label: 'Data', Icon: Database, x: 70, y: 260 },
  { id: 'training', label: 'Training', Icon: Sparkles, x: 245, y: 260 },
  { id: 'evaluation', label: 'Evaluation', Icon: ShieldCheck, x: 420, y: 260 },
  { id: 'deployment', label: 'Deployment', Icon: Rocket, x: 595, y: 260 },
  { id: 'monitoring', label: 'Monitoring', Icon: Activity, x: 650, y: 400 },
];

const NODE_BY_ID = new Map(NODES.map((n) => [n.id, n]));
const R = 26;

function curve(fromId: string, toId: string, dip = 34): string {
  const a = NODE_BY_ID.get(fromId)!;
  const b = NODE_BY_ID.get(toId)!;
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2 + dip;
  return `M ${a.x} ${a.y} Q ${mx} ${my} ${b.x} ${b.y}`;
}

// Forward pipeline: data flowing left to right through the real agent stages.
const FORWARD_EDGES = [
  { id: 'e-data-training', d: curve('data', 'training') },
  { id: 'e-training-eval', d: curve('training', 'evaluation') },
  { id: 'e-eval-deploy', d: curve('evaluation', 'deployment') },
  { id: 'e-deploy-monitor', d: curve('deployment', 'monitoring', 60) },
];

// The closed retraining loop — monitoring feeds back into training on drift,
// the real path agents/orchestrator.py's process_pending_triggers() runs.
const FEEDBACK_EDGE = {
  id: 'e-monitor-training',
  d: 'M 650 428 C 650 470 245 470 245 288',
};

// Orchestrator coordinates every stage — thin static spokes, not part of the flow.
const ORCHESTRATOR_EDGES = NODES.filter((n) => n.id !== 'orchestrator').map((n) => ({
  id: `e-orch-${n.id}`,
  d: `M 360 86 L ${n.x} ${n.y - R - 4}`,
}));

/**
 * Purely decorative hero visual for the sign-in page — the real agent
 * roles (same icon mapping as components/pipeline/agent-node.tsx)
 * connected in the real pipeline order, with packets animated along the
 * real closed retraining loop (monitoring -> training) the orchestrator
 * actually runs. Fully theme-aware (every color is a CSS custom property
 * resolved via inline `style`, not a raw SVG attribute, so it tracks the
 * app's own light/dark/system toggle rather than being locked to one
 * look). Respects prefers-reduced-motion by rendering statically, no
 * SMIL animation, when the user has that preference set.
 */
export function PipelineFlowVisual() {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mql.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  return (
    <svg viewBox="0 0 720 460" width="100%" height="100%" role="img" aria-label="Animated diagram of the agent pipeline">
      <defs>
        <radialGradient id="glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.16" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="360" cy="230" r="300" fill="url(#glow)" />

      {ORCHESTRATOR_EDGES.map((e) => (
        <path key={e.id} id={e.id} d={e.d} style={{ stroke: 'var(--border)' }} strokeWidth={1.5} fill="none" />
      ))}
      {FORWARD_EDGES.map((e) => (
        <path key={e.id} id={e.id} d={e.d} style={{ stroke: 'var(--border-strong)' }} strokeWidth={2} fill="none" />
      ))}
      <path
        id={FEEDBACK_EDGE.id} d={FEEDBACK_EDGE.d}
        style={{ stroke: 'var(--warn)', opacity: 0.5 }} strokeWidth={2} fill="none" strokeDasharray="3 5"
      />

      {!reducedMotion && (
        <>
          {FORWARD_EDGES.map((e, i) => (
            <circle key={`p-${e.id}`} r={4} style={{ fill: 'var(--accent)' }}>
              <animateMotion dur="2.6s" begin={`${i * 0.5}s`} repeatCount="indefinite">
                <mpath href={`#${e.id}`} />
              </animateMotion>
            </circle>
          ))}
          <circle r={4} style={{ fill: 'var(--warn)' }}>
            <animateMotion dur="4s" repeatCount="indefinite">
              <mpath href={`#${FEEDBACK_EDGE.id}`} />
            </animateMotion>
          </circle>
        </>
      )}

      {NODES.map((n) => (
        <g key={n.id} className="flow-node" transform={`translate(${n.x}, ${n.y})`}>
          <circle r={R} style={{ fill: 'var(--bg-inset)', stroke: 'var(--border-strong)' }} strokeWidth={1.5} />
          <foreignObject x={-13} y={-13} width={26} height={26}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-strong)' }}>
              <n.Icon size={16} strokeWidth={2} />
            </div>
          </foreignObject>
          <text y={R + 18} textAnchor="middle" style={{ fill: 'var(--text-faint)' }} fontSize={11} fontFamily="var(--font-sans)">
            {n.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
