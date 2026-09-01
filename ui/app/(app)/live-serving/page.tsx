'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { LineChart } from '@/components/charts/line-chart';
import { RollbackDialog } from '@/components/serving/rollback-dialog';
import { listModels } from '@/lib/api';
import { promQuery, promQueryRange } from '@/lib/prometheus';
import { useEventStream } from '@/lib/ws';
import type { WsEvent } from '@/lib/schemas';

export default function LiveServingPage() {
  const [deployEvents, setDeployEvents] = useState<WsEvent[]>([]);
  useEventStream((e) => {
    if (e.channel?.includes('.deployment_status_changed')) {
      setDeployEvents((prev) => [e, ...prev].slice(0, 30));
    }
  });

  const modelsQuery = useQuery({ queryKey: ['models', 'Production'], queryFn: () => listModels('Production') });
  const models = modelsQuery.data?.models ?? [];

  return (
    <>
      <TopBar title="Live Serving Dashboard" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        {models.length === 0 && !modelsQuery.isLoading && (
          <Card>
            <CardBody>
              <p className="text-faint" style={{ fontSize: 13 }}>
                No models currently in the Production stage — nothing is being served.
              </p>
            </CardBody>
          </Card>
        )}
        {models.map((m) => (
          <ModelServingCard key={`${m.name}-${m.version}`} modelName={m.name} modelVersion={m.version} />
        ))}

        <Card>
          <CardHeader title="Deployment / canary events (live)" />
          <CardBody className="stack" style={{ gap: 6 }}>
            {deployEvents.length === 0 && <p className="text-faint" style={{ fontSize: 12.5 }}>No events yet.</p>}
            {deployEvents.map((e, i) => (
              <div key={i} className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
                <span className="mono">{String(e.model_name)} v{String(e.model_version ?? '?')}</span>
                <span className="text-muted">{String(e.phase)} — {String(e.status)} ({String(e.traffic_pct ?? '?')}%)</span>
              </div>
            ))}
          </CardBody>
        </Card>
      </div>
    </>
  );
}

function ModelServingCard({ modelName, modelVersion }: { modelName: string; modelVersion: string }) {
  const svcHint = `mlops-model-${modelName}`;
  const selector = `model_name="${modelName}",model_version="${modelVersion}"`;

  const latencyQuery = useQuery({
    queryKey: ['prom', 'latency', modelName, modelVersion],
    queryFn: async () => {
      const now = Math.floor(Date.now() / 1000);
      const quantile = (q: number) =>
        promQueryRange(
          `histogram_quantile(${q}, sum(rate(mlops_prediction_duration_seconds_bucket{${selector}}[5m])) by (le)) * 1000`,
          now - 1800, now, 30,
        );
      const [p50, p95, p99] = await Promise.all([quantile(0.5), quantile(0.95), quantile(0.99)]);
      return { p50, p95, p99 };
    },
    refetchInterval: 30_000,
  });

  const rateQuery = useQuery({
    queryKey: ['prom', 'rates', modelName, modelVersion],
    queryFn: async () => {
      const [errorRate, throughput] = await Promise.all([
        promQuery(`sum(rate(mlops_predictions_total{${selector},status="error"}[5m])) / sum(rate(mlops_predictions_total{${selector}}[5m]))`),
        promQuery(`sum(rate(mlops_predictions_total{${selector}}[5m]))`),
      ]);
      return {
        errorRate: errorRate[0]?.value ?? null,
        throughput: throughput[0]?.value ?? null,
      };
    },
    refetchInterval: 15_000,
  });

  const series = [
    { label: 'p50', points: (latencyQuery.data?.p50[0]?.points ?? []).map((p) => ({ ts: p.ts, value: p.value })) },
    { label: 'p95', points: (latencyQuery.data?.p95[0]?.points ?? []).map((p) => ({ ts: p.ts, value: p.value })) },
    { label: 'p99', points: (latencyQuery.data?.p99[0]?.points ?? []).map((p) => ({ ts: p.ts, value: p.value })) },
  ];
  const hasData = series.some((s) => s.points.length > 0);

  return (
    <Card>
      <CardHeader
        title={`${modelName} v${modelVersion}`}
        action={<RollbackDialog serviceName={svcHint} modelName={modelName} modelVersion={modelVersion} />}
      />
      <CardBody>
        <div className="grid-cols" style={{ gridTemplateColumns: '2fr 1fr 1fr', alignItems: 'start' }}>
          <div>
            {hasData ? (
              <LineChart series={series} yLabel="ms" height={180} />
            ) : (
              <p className="text-faint" style={{ fontSize: 12.5 }}>
                No traffic in the last 30 minutes for this model/version pair — Prometheus has nothing to
                report, so no chart is shown rather than a synthetic one.
              </p>
            )}
          </div>
          <Metric
            label="Error rate"
            value={rateQuery.data?.errorRate != null ? `${(rateQuery.data.errorRate * 100).toFixed(2)}%` : '—'}
          />
          <Metric
            label="Throughput"
            value={rateQuery.data?.throughput != null ? `${rateQuery.data.throughput.toFixed(2)} req/s` : '—'}
          />
        </div>
        <div className="row" style={{ marginTop: 14, gap: 8, alignItems: 'center' }}>
          <input type="checkbox" disabled />
          <span className="text-faint" style={{ fontSize: 12 }}>
            Shadow mode — not wired to real traffic-splitting infrastructure yet (deployment_agent.py has no
            shadow-only routing path); left disabled rather than faking the control.
          </span>
        </div>
      </CardBody>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-faint" style={{ fontSize: 11, textTransform: 'uppercase' }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 700, marginTop: 4 }}>{value}</div>
    </div>
  );
}
