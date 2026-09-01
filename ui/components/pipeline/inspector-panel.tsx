'use client';

import type { Edge, Node } from 'reactflow';
import { X } from 'lucide-react';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { EDGE_TYPES, type EdgeType } from '@/lib/pipeline/edge-types';
import type { AgentCard } from '@/lib/schemas';

export function InspectorPanel({
  selectedNode, selectedEdge, agentsById, onClose, onEdgeTypeChange, onDeleteNode, onDeleteEdge,
}: {
  selectedNode: Node | null;
  selectedEdge: Edge | null;
  agentsById: Map<string, AgentCard>;
  onClose: () => void;
  onEdgeTypeChange: (edgeId: string, type: EdgeType) => void;
  onDeleteNode: (nodeId: string) => void;
  onDeleteEdge: (edgeId: string) => void;
}) {
  if (!selectedNode && !selectedEdge) return null;

  return (
    <Card style={{ width: 260, flexShrink: 0 }}>
      <CardHeader
        title={selectedNode ? 'Node' : 'Edge'}
        action={
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close inspector">
            <X size={14} />
          </button>
        }
      />
      <CardBody className="stack" style={{ gap: 10 }}>
        {selectedNode && (
          <>
            <div>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{selectedNode.data?.label}</div>
              <div className="text-faint mono" style={{ fontSize: 11 }}>{selectedNode.data?.agentId}</div>
            </div>
            <div>
              <div className="text-faint" style={{ fontSize: 10.5, textTransform: 'uppercase', marginBottom: 4 }}>Capabilities</div>
              <div className="stack" style={{ gap: 4 }}>
                {(agentsById.get(selectedNode.data?.agentId)?.capabilities ?? []).map((c) => (
                  <span key={c} className="badge badge-accent" style={{ width: 'fit-content' }}>{c}</span>
                ))}
                {(agentsById.get(selectedNode.data?.agentId)?.capabilities ?? []).length === 0 && (
                  <span className="text-faint" style={{ fontSize: 11.5 }}>none declared</span>
                )}
              </div>
            </div>
            <div>
              <div className="text-faint" style={{ fontSize: 10.5, textTransform: 'uppercase', marginBottom: 4 }}>MCP tools</div>
              <div className="stack" style={{ gap: 2 }}>
                {(agentsById.get(selectedNode.data?.agentId)?.mcp_tools ?? []).map((t) => (
                  <span key={t} className="mono text-muted" style={{ fontSize: 11 }}>{t}</span>
                ))}
              </div>
            </div>
            <Button variant="danger" size="sm" onClick={() => onDeleteNode(selectedNode.id)}>Remove node</Button>
          </>
        )}
        {selectedEdge && (
          <>
            <div className="text-muted mono" style={{ fontSize: 11 }}>
              {selectedEdge.source} → {selectedEdge.target}
            </div>
            <div>
              <div className="text-faint" style={{ fontSize: 10.5, textTransform: 'uppercase', marginBottom: 6 }}>Edge type</div>
              <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
                {EDGE_TYPES.map((t) => (
                  <button
                    key={t}
                    className="btn btn-sm"
                    style={{
                      background: selectedEdge.data?.edgeType === t ? 'var(--accent-soft)' : 'var(--bg-elevated)',
                      border: '1px solid var(--border-strong)',
                      color: selectedEdge.data?.edgeType === t ? 'var(--accent-strong)' : 'var(--text)',
                    }}
                    onClick={() => onEdgeTypeChange(selectedEdge.id, t)}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <Button variant="danger" size="sm" onClick={() => onDeleteEdge(selectedEdge.id)}>Remove edge</Button>
          </>
        )}
      </CardBody>
    </Card>
  );
}
