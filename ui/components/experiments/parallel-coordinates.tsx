'use client';

import * as Plot from '@observablehq/plot';
import { useEffect, useRef } from 'react';
import type { ExperimentRun } from '@/lib/schemas';

/** Real Observable Plot parallel-coordinates view over each run's numeric metrics. */
export function ParallelCoordinates({ runs }: { runs: ExperimentRun[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const axes = Array.from(new Set(runs.flatMap((r) => Object.keys(r.metrics)))).sort();
    if (axes.length === 0 || runs.length === 0) {
      ref.current.replaceChildren();
      return;
    }
    const ranges = new Map(axes.map((a) => {
      const vals = runs.map((r) => r.metrics[a]).filter((v): v is number => v != null);
      return [a, [Math.min(...vals), Math.max(...vals)] as [number, number]];
    }));

    const rows = runs.flatMap((r) =>
      axes
        .filter((a) => r.metrics[a] != null)
        .map((a) => {
          const [lo, hi] = ranges.get(a)!;
          const raw = r.metrics[a] as number;
          const norm = hi === lo ? 0.5 : (raw - lo) / (hi - lo);
          return { run: r.run_name ?? r.run_id, axis: a, norm, raw };
        }),
    );

    const plot = Plot.plot({
      height: 220,
      width: ref.current.clientWidth || 640,
      marginLeft: 80,
      marginBottom: 30,
      style: { background: 'transparent', color: 'var(--text-muted)', fontSize: '11px' },
      x: { domain: axes, label: null },
      y: { domain: [0, 1], ticks: [], label: null },
      color: { legend: true, scheme: 'tableau10' },
      marks: [
        Plot.line(rows, { x: 'axis', y: 'norm', stroke: 'run', strokeWidth: 2, curve: 'catmull-rom', title: (d: any) => `${d.run}\n${d.axis}=${d.raw}` }),
        Plot.dot(rows, { x: 'axis', y: 'norm', fill: 'run', r: 3 }),
      ],
    });
    ref.current.replaceChildren(plot);
    return () => plot.remove();
  }, [runs]);

  if (runs.length === 0) {
    return <p className="text-faint" style={{ fontSize: 12.5 }}>No runs to plot.</p>;
  }
  return <div ref={ref} className="scroll-x" />;
}
