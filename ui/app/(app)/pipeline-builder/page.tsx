'use client';

import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import ReactFlow, {
  addEdge, applyEdgeChanges, applyNodeChanges, Background, BackgroundVariant, MarkerType, MiniMap,
  ReactFlowProvider, type Connection, type Edge, type EdgeChange, type Node, type NodeChange,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { StartPipelineDialog } from '@/components/workflow/start-pipeline-dialog';
import { PIPELINE_NODE_TYPES } from '@/components/pipeline/agent-node';
import { CanvasToolbar } from '@/components/pipeline/canvas-toolbar';
import { InspectorPanel } from '@/components/pipeline/inspector-panel';
import { autoLayout } from '@/lib/pipeline/layout';
import { EDGE_TYPE_STYLE, type EdgeType } from '@/lib/pipeline/edge-types';
import { listAgents } from '@/lib/api';
import { exportAsJson, exportAsPython, exportAsYaml, validatePipeline, type ValidationIssue } from '@/lib/pipeline/validate';

let nodeSeq = 0;

function edgeStyleFor(type: EdgeType) {
  const s = EDGE_TYPE_STYLE[type];
  return {
    style: { stroke: s.stroke, strokeWidth: 1.75, strokeDasharray: s.dash },
    markerEnd: { type: MarkerType.ArrowClosed, color: s.stroke, width: 16, height: 16 },
    label: type,
    data: { edgeType: type },
  };
}

export default function PipelineBuilderPage() {
  const agentsQuery = useQuery({ queryKey: ['agents'], queryFn: listAgents });
  const agents = agentsQuery.data?.agents ?? [];
  const agentsById = useMemo(
    () => new Map(agents.map((a) => [a.agent_id, {
      ...a,
      capabilities: a.capabilities ?? [],
      mcp_tools: a.mcp_tools ?? [],
      requires_human_approval_for: a.requires_human_approval_for ?? [],
    }])),
    [agents],
  );

  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [issues, setIssues] = useState<ValidationIssue[] | null>(null);
  const [exportText, setExportText] = useState<{ format: string; text: string } | null>(null);

  const onNodesChange = useCallback((changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);
  const onConnect = useCallback(
    (conn: Connection) => setEdges((eds) => addEdge({ ...conn, ...edgeStyleFor('sequential') }, eds)),
    [],
  );

  const addNode = (agentId: string, label: string) => {
    nodeSeq += 1;
    const id = `${agentId}-${nodeSeq}`;
    setNodes((nds) => [
      ...nds,
      { id, type: 'agentNode', position: { x: 60 + (nodeSeq % 5) * 60, y: 40 + (nodeSeq % 4) * 90 }, data: { label, agentId } },
    ]);
  };

  const runAutoLayout = () => setNodes((nds) => autoLayout(nds, edges));

  const setEdgeType = (edgeId: string, type: EdgeType) =>
    setEdges((eds) => eds.map((e) => (e.id === edgeId ? { ...e, ...edgeStyleFor(type) } : e)));

  const deleteNode = (nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setSelectedNodeId(null);
  };
  const deleteEdge = (edgeId: string) => {
    setEdges((eds) => eds.filter((e) => e.id !== edgeId));
    setSelectedEdgeId(null);
  };

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? null;
  const selectedEdge = edges.find((e) => e.id === selectedEdgeId) ?? null;

  return (
    <>
      <TopBar title="Pipeline Builder" />
      <div className="row" style={{ padding: 24, gap: 16, alignItems: 'flex-start' }}>
        <div className="stack" style={{ width: 220, gap: 16, flexShrink: 0 }}>
          <Card>
            <CardHeader title="Real agents (A2A registry)" />
            <CardBody className="stack" style={{ gap: 6 }}>
              {agents.map((a) => (
                <button
                  key={a.agent_id}
                  onClick={() => addNode(a.agent_id, a.name)}
                  className="btn btn-secondary btn-sm"
                  style={{ justifyContent: 'flex-start', textAlign: 'left' }}
                  title={(a.capabilities ?? []).join(', ')}
                >
                  + {a.name}
                </button>
              ))}
              {agents.length === 0 && !agentsQuery.isLoading && (
                <p className="text-faint" style={{ fontSize: 12 }}>No agents registered.</p>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Actions" />
            <CardBody className="stack" style={{ gap: 8 }}>
              <Button size="sm" variant="secondary" onClick={() => setIssues(validatePipeline(nodes, edges))}>
                Validate
              </Button>
              <div title="No real duration/cost estimator is wired up yet (PlannerAgent has no dry-run method) — left disabled rather than showing fabricated numbers.">
                <Button size="sm" variant="ghost" disabled style={{ width: '100%' }}>Dry run (unavailable)</Button>
              </div>
              <Button size="sm" variant="secondary" onClick={() => setExportText({ format: 'JSON', text: exportAsJson(nodes, edges) })}>
                Export JSON
              </Button>
              <Button size="sm" variant="secondary" onClick={() => setExportText({ format: 'YAML', text: exportAsYaml(nodes, edges) })}>
                Export YAML
              </Button>
              <Button size="sm" variant="secondary" onClick={() => setExportText({ format: 'Python', text: exportAsPython(nodes, edges) })}>
                Export Python
              </Button>
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                <p className="text-faint" style={{ fontSize: 11, marginBottom: 8 }}>
                  Executing always runs the real, fixed orchestrator graph (agents/orchestrator.py) — this
                  canvas is a design/export tool, not yet a dynamic graph compiler.
                </p>
                <StartPipelineDialog />
              </div>
            </CardBody>
          </Card>

          {issues && (
            <Card>
              <CardHeader title={`Validation (${issues.length === 0 ? 'passed' : `${issues.length} issue(s)`})`} />
              <CardBody className="stack" style={{ gap: 6 }}>
                {issues.length === 0 && <p style={{ color: 'var(--ok)', fontSize: 12.5 }}>No structural issues found.</p>}
                {issues.map((v, i) => (
                  <p key={i} style={{ color: v.severity === 'error' ? 'var(--crit)' : 'var(--warn)', fontSize: 12 }}>
                    {v.message}
                  </p>
                ))}
              </CardBody>
            </Card>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <Card>
            <CardBody style={{ padding: 0 }}>
              <ReactFlowProvider>
                <div style={{ height: 560, position: 'relative' }} className="pipeline-canvas">
                  <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={PIPELINE_NODE_TYPES}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={(_, n) => { setSelectedNodeId(n.id); setSelectedEdgeId(null); }}
                    onEdgeClick={(_, e) => { setSelectedEdgeId(e.id); setSelectedNodeId(null); }}
                    onPaneClick={() => { setSelectedNodeId(null); setSelectedEdgeId(null); }}
                    fitView
                    proOptions={{ hideAttribution: true }}
                    defaultEdgeOptions={{ type: 'smoothstep' }}
                  >
                    <Background variant={BackgroundVariant.Dots} color="var(--border)" gap={18} size={1} />
                    <MiniMap
                      pannable zoomable
                      nodeColor="var(--accent-soft)"
                      nodeStrokeColor="var(--accent)"
                      maskColor="rgba(0,0,0,0.06)"
                    />
                    <div style={{ position: 'absolute', top: 12, left: 12, zIndex: 5 }}>
                      <CanvasToolbar onAutoLayout={runAutoLayout} />
                    </div>
                  </ReactFlow>
                </div>
              </ReactFlowProvider>
            </CardBody>
          </Card>

          {exportText && (
            <Card style={{ marginTop: 16 }}>
              <CardHeader
                title={`${exportText.format} export`}
                action={<Button size="sm" variant="ghost" onClick={() => setExportText(null)}>Close</Button>}
              />
              <CardBody>
                <pre className="mono scroll-x" style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{exportText.text}</pre>
              </CardBody>
            </Card>
          )}
        </div>

        <InspectorPanel
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          agentsById={agentsById}
          onClose={() => { setSelectedNodeId(null); setSelectedEdgeId(null); }}
          onEdgeTypeChange={setEdgeType}
          onDeleteNode={deleteNode}
          onDeleteEdge={deleteEdge}
        />
      </div>
    </>
  );
}
