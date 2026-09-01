/**
 * Thin client for Prometheus's HTTP API, routed through the read-only
 * /api/gateway/prometheus proxy. Real PromQL, the same expressions
 * configs/monitoring/grafana-dashboard.json's panels use — never a
 * hardcoded time series.
 */
import { z } from 'zod';

const InstantResultSchema = z.object({
  status: z.string(),
  data: z.object({
    resultType: z.string(),
    result: z.array(
      z.object({
        metric: z.record(z.string()),
        value: z.tuple([z.number(), z.string()]),
      }),
    ),
  }),
});

const RangeResultSchema = z.object({
  status: z.string(),
  data: z.object({
    resultType: z.string(),
    result: z.array(
      z.object({
        metric: z.record(z.string()),
        values: z.array(z.tuple([z.number(), z.string()])),
      }),
    ),
  }),
});

export interface Point {
  ts: number;
  value: number;
  labels: Record<string, string>;
}

export async function promQuery(promql: string): Promise<Point[]> {
  const url = `/api/gateway/prometheus/api/v1/query?query=${encodeURIComponent(promql)}`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const parsed = InstantResultSchema.safeParse(await res.json());
  if (!parsed.success || parsed.data.status !== 'success') return [];
  return parsed.data.data.result.map((r) => ({
    ts: r.value[0],
    value: Number(r.value[1]),
    labels: r.metric,
  }));
}

export async function promQueryRange(promql: string, startSec: number, endSec: number, stepSec: number) {
  const params = new URLSearchParams({
    query: promql,
    start: String(startSec),
    end: String(endSec),
    step: String(stepSec),
  });
  const res = await fetch(`/api/gateway/prometheus/api/v1/query_range?${params}`);
  if (!res.ok) return [];
  const parsed = RangeResultSchema.safeParse(await res.json());
  if (!parsed.success || parsed.data.status !== 'success') return [];
  return parsed.data.data.result.map((series) => ({
    labels: series.metric,
    points: series.values.map(([ts, v]) => ({ ts: ts * 1000, value: Number(v) })),
  }));
}
