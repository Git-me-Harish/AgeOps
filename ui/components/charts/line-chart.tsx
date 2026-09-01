'use client';

import * as Plot from '@observablehq/plot';
import { useEffect, useRef } from 'react';

export interface Series {
  label: string;
  points: { ts: number; value: number }[];
}

/** Real Observable Plot line chart (plan §6.1) — no hardcoded data ever reaches this component. */
export function LineChart({
  series,
  yLabel,
  height = 220,
  formatY,
}: {
  series: Series[];
  yLabel: string;
  height?: number;
  formatY?: (v: number) => string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const data = series.flatMap((s) => s.points.map((p) => ({ ...p, series: s.label })));

    if (data.length === 0) {
      containerRef.current.replaceChildren();
      return;
    }

    const plot = Plot.plot({
      height,
      width: containerRef.current.clientWidth || 640,
      marginLeft: 52,
      style: { background: 'transparent', color: 'var(--text-muted)', fontSize: '11px' },
      x: { type: 'time', label: null },
      y: { label: yLabel, grid: true, tickFormat: formatY },
      color: { legend: series.length > 1, scheme: 'tableau10' },
      marks: [
        Plot.gridY({ stroke: 'var(--border)' }),
        Plot.line(data, { x: 'ts', y: 'value', stroke: 'series', strokeWidth: 2 }),
        Plot.dot(
          data.filter((_, i, arr) => i === arr.length - 1),
          { x: 'ts', y: 'value', fill: 'series', r: 3 },
        ),
        Plot.ruleY([0]),
      ],
    });
    containerRef.current.replaceChildren(plot);
    return () => plot.remove();
  }, [series, yLabel, height, formatY]);

  return <div ref={containerRef} className="scroll-x" />;
}
