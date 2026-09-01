import type { Edge, Node } from 'reactflow';

const COLUMN_WIDTH = 240;
const ROW_HEIGHT = 120;

/**
 * Layered left-to-right auto-layout: each node's column is
 * 1 + max(column of its predecessors) — a plain BFS/topological layering,
 * the same idea dagre uses internally, without pulling in the dependency.
 * Nodes with no incoming edge start at column 0; disconnected nodes are
 * placed after the main graph rather than overlapping it.
 */
export function autoLayout(nodes: Node[], edges: Edge[]): Node[] {
  const incoming = new Map<string, string[]>();
  nodes.forEach((n) => incoming.set(n.id, []));
  edges.forEach((e) => {
    if (incoming.has(e.target)) incoming.get(e.target)!.push(e.source);
  });

  const column = new Map<string, number>();
  const visiting = new Set<string>();

  function resolve(id: string): number {
    if (column.has(id)) return column.get(id)!;
    if (visiting.has(id)) return 0; // cycle guard — treat as root rather than recursing forever
    visiting.add(id);
    const preds = incoming.get(id) ?? [];
    const col = preds.length === 0 ? 0 : Math.max(...preds.map(resolve)) + 1;
    column.set(id, col);
    visiting.delete(id);
    return col;
  }

  nodes.forEach((n) => resolve(n.id));

  const rowsPerColumn = new Map<number, number>();
  return nodes.map((n) => {
    const col = column.get(n.id) ?? 0;
    const row = rowsPerColumn.get(col) ?? 0;
    rowsPerColumn.set(col, row + 1);
    return { ...n, position: { x: col * COLUMN_WIDTH, y: row * ROW_HEIGHT } };
  });
}
