import type { Edge, Node } from 'reactflow';

export interface ValidationIssue {
  severity: 'error' | 'warning';
  message: string;
}

/** Real graph-structure validation on whatever the user has actually drawn — no canned pass/fail. */
export function validatePipeline(nodes: Node[], edges: Edge[]): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (nodes.length === 0) {
    return [{ severity: 'error', message: 'Canvas is empty — add at least one agent node.' }];
  }

  const REQUIRED_AGENTS = ['data_agent', 'training_agent', 'evaluation_agent'];
  const presentAgentIds = new Set(nodes.map((n) => n.data?.agentId as string));
  for (const req of REQUIRED_AGENTS) {
    if (!presentAgentIds.has(req)) {
      issues.push({ severity: 'warning', message: `Pipeline has no '${req}' node — the real orchestrator graph runs this stage regardless of what's drawn here.` });
    }
  }

  const connected = new Set<string>();
  edges.forEach((e) => { connected.add(e.source); connected.add(e.target); });
  const disconnected = nodes.filter((n) => nodes.length > 1 && !connected.has(n.id));
  disconnected.forEach((n) => issues.push({ severity: 'warning', message: `Node '${n.data?.label}' is disconnected from the rest of the graph.` }));

  // Cycle detection via DFS
  const adjacency = new Map<string, string[]>();
  edges.forEach((e) => {
    if (!adjacency.has(e.source)) adjacency.set(e.source, []);
    adjacency.get(e.source)!.push(e.target);
  });
  const visiting = new Set<string>();
  const visited = new Set<string>();
  let hasCycle = false;
  function dfs(id: string) {
    if (hasCycle) return;
    visiting.add(id);
    for (const next of adjacency.get(id) ?? []) {
      if (visiting.has(next)) { hasCycle = true; return; }
      if (!visited.has(next)) dfs(next);
    }
    visiting.delete(id);
    visited.add(id);
  }
  nodes.forEach((n) => { if (!visited.has(n.id)) dfs(n.id); });
  if (hasCycle) issues.push({ severity: 'error', message: 'Graph contains a cycle — the real LangGraph executor cannot run a cyclic pipeline.' });

  return issues;
}

export function exportAsJson(nodes: Node[], edges: Edge[]): string {
  return JSON.stringify(
    {
      nodes: nodes.map((n) => ({ id: n.id, agent_id: n.data?.agentId, label: n.data?.label, position: n.position })),
      edges: edges.map((e) => ({ source: e.source, target: e.target, type: e.data?.edgeType ?? 'sequential' })),
    },
    null, 2,
  );
}

export function exportAsYaml(nodes: Node[], edges: Edge[]): string {
  const lines: string[] = ['nodes:'];
  nodes.forEach((n) => lines.push(`  - id: ${n.id}\n    agent_id: ${n.data?.agentId}\n    label: "${n.data?.label}"`));
  lines.push('edges:');
  edges.forEach((e) => lines.push(`  - source: ${e.source}\n    target: ${e.target}\n    type: ${e.data?.edgeType ?? 'sequential'}`));
  return lines.join('\n');
}

export function exportAsPython(nodes: Node[], edges: Edge[]): string {
  const nodeLines = nodes.map((n) => `    "${n.id}",  # ${n.data?.agentId}`).join('\n');
  const edgeLines = edges.map((e) => `    ("${e.source}", "${e.target}"),  # ${e.data?.edgeType ?? 'sequential'}`).join('\n');
  return [
    '"""Generated from the Pipeline Builder canvas — a design artifact, not',
    'an executable substitute for agents/orchestrator.py\'s real LangGraph',
    '(which is defined in Python and fixed, not dynamically driven by this',
    'export). Use this as a reference / starting point for a custom script."""',
    '',
    'NODES = [',
    nodeLines,
    ']',
    '',
    'EDGES = [',
    edgeLines,
    ']',
  ].join('\n');
}
