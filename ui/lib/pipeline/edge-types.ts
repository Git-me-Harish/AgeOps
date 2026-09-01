export type EdgeType = 'sequential' | 'parallel' | 'conditional' | 'fallback';

export const EDGE_TYPES: EdgeType[] = ['sequential', 'parallel', 'conditional', 'fallback'];

export const EDGE_TYPE_STYLE: Record<EdgeType, { stroke: string; dash?: string }> = {
  sequential: { stroke: 'var(--border-strong)' },
  parallel: { stroke: 'var(--accent)' },
  conditional: { stroke: 'var(--warn)', dash: '6 3' },
  fallback: { stroke: 'var(--crit)', dash: '2 3' },
};
