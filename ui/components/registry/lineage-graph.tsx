'use client';

import ReactFlow, { Background, Controls, type Edge, type Node } from 'reactflow';
import 'reactflow/dist/style.css';
import { useMemo } from 'react';
import type { LineageGraph } from '@/lib/schemas';

const TYPE_COLOR: Record<string, string> = {
  dataset: 'var(--accent)',
  validation: 'var(--ok)',
  drift_report: 'var(--warn)',
  training_run: 'var(--accent-strong)',
  evaluation_run: 'var(--ok)',
  model_version: 'var(--accent-strong)',
  oci_image: 'var(--unknown)',
  inference_service: 'var(--crit)',
};

/** Renders the real DAG from GET /registry/v1/lineage — no synthetic layout data, only real rows. */
export function LineageGraphView({ graph }: { graph: LineageGraph }) {
  const { nodes, edges } = useMemo(() => {
    const rawNodes = graph.nodes as Array<Record<string, unknown>>;
    const rawEdges = graph.edges as Array<Record<string, unknown>>;

    const nodes: Node[] = rawNodes.map((n, i) => ({
      id: String(n.id),
      position: { x: (i % 4) * 220, y: Math.floor(i / 4) * 110 },
      data: { label: String(n.display_label ?? n.external_id ?? n.id) },
      style: {
        border: `2px solid ${TYPE_COLOR[String(n.node_type)] ?? 'var(--border-strong)'}`,
        borderRadius: 8,
        background: 'var(--bg-elevated)',
        color: 'var(--text)',
        fontSize: 11,
        padding: 6,
        width: 190,
      },
    }));

    const edges: Edge[] = rawEdges.map((e) => ({
      id: String(e.id),
      source: String(e.parent_node_id),
      target: String(e.child_node_id),
      label: String(e.relationship ?? ''),
      style: { stroke: 'var(--border-strong)' },
      labelStyle: { fontSize: 10, fill: 'var(--text-muted)' },
    }));

    return { nodes, edges };
  }, [graph]);

  if (nodes.length === 0) {
    return (
      <p className="text-faint" style={{ fontSize: 12.5, padding: 16 }}>
        No lineage recorded yet for {graph.model_name} v{graph.model_version} — nodes are written as the
        real pipeline runs (ingestion → validation → training → evaluation → packaging → deployment).
      </p>
    );
  }

  return (
    <div style={{ height: 360, border: '1px solid var(--border)', borderRadius: 8 }}>
      <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
        <Background color="var(--border)" gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
