'use client';

import { useQuery } from '@tanstack/react-query';
import { getRunTrace } from '@/lib/api';
import type { Span } from '@/lib/schemas';

function buildTree(spans: Span[]) {
  const byParent = new Map<string, Span[]>();
  spans.forEach((s) => {
    const key = s.parent_id || '__root__';
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key)!.push(s);
  });
  return byParent;
}

function SpanNode({ span, byParent, depth }: { span: Span; byParent: Map<string, Span[]>; depth: number }) {
  const children = byParent.get(span.span_id) ?? [];
  const durationMs =
    span.start_time_ns != null && span.end_time_ns != null
      ? ((span.end_time_ns - span.start_time_ns) / 1_000_000).toFixed(2)
      : null;
  return (
    <div>
      <div className="row" style={{ gap: 8, paddingLeft: depth * 16, fontSize: 12.5, padding: '4px 0' }}>
        <span style={{ color: 'var(--accent-strong)' }}>▸</span>
        <span className="mono">{span.name}</span>
        {durationMs && <span className="text-faint">{durationMs}ms</span>}
        <span className="text-faint">{span.status}</span>
      </div>
      {children.map((c) => (
        <SpanNode key={c.span_id} span={c} byParent={byParent} depth={depth + 1} />
      ))}
    </div>
  );
}

/** Real expandable span tree for a run — every LLM call / tool invocation / decision the agents traced. */
export function TraceView({ runId }: { runId: string }) {
  const traceQuery = useQuery({ queryKey: ['trace', runId], queryFn: () => getRunTrace(runId) });
  const spans = traceQuery.data?.spans ?? [];
  const byParent = buildTree(spans);
  const roots = byParent.get('__root__') ?? [];

  if (traceQuery.isLoading) return <p className="text-faint" style={{ fontSize: 12.5 }}>Loading trace…</p>;
  if (roots.length === 0) {
    return (
      <p className="text-faint" style={{ fontSize: 12.5 }}>
        No trace spans recorded for this run — it may predate @mlflow.trace instrumentation, or the agents
        it ran didn&apos;t emit any (LLMGateway/agents each add spans on their own tool calls).
      </p>
    );
  }
  return (
    <div className="stack">
      {roots.map((r) => (
        <SpanNode key={r.span_id} span={r} byParent={byParent} depth={0} />
      ))}
    </div>
  );
}
