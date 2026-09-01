'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { StatusPill, severityFromAlertLevel } from '@/components/ui/status-pill';
import { LineChart } from '@/components/charts/line-chart';
import { Pagination } from '@/components/ui/pagination';
import { listDriftReports, listModels, listMonitoringEvents, triggerRetraining } from '@/lib/api';
import { usePagedSlice } from '@/lib/use-paged-slice';

const REPORTS_PAGE_SIZE = 20;

export default function DriftHealthPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: () => listModels() });
  const models = modelsQuery.data?.models ?? [];
  const uniqueNames = Array.from(new Set(models.map((m) => m.name)));

  useEffect(() => {
    if (!selectedModel && uniqueNames.length > 0) setSelectedModel(uniqueNames[0] ?? null);
  }, [uniqueNames, selectedModel]);

  return (
    <>
      <TopBar title="Drift & Health Center" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <div className="row" style={{ gap: 8 }}>
          {uniqueNames.map((name) => (
            <Button key={name} size="sm" variant={selectedModel === name ? 'primary' : 'secondary'} onClick={() => setSelectedModel(name)}>
              {name}
            </Button>
          ))}
          {uniqueNames.length === 0 && !modelsQuery.isLoading && (
            <p className="text-faint" style={{ fontSize: 13 }}>No registered models to monitor yet.</p>
          )}
        </div>
        {selectedModel && <ModelDriftPanel modelName={selectedModel} />}
      </div>
    </>
  );
}

function ModelDriftPanel({ modelName }: { modelName: string }) {
  const qc = useQueryClient();
  const eventsQuery = useQuery({ queryKey: ['monitoring-events', modelName], queryFn: () => listMonitoringEvents(modelName) });
  const reportsQuery = useQuery({ queryKey: ['drift-reports', modelName], queryFn: () => listDriftReports(modelName) });

  const retrainMutation = useMutation({
    mutationFn: () => triggerRetraining(modelName, events[0]?.model_version ?? 'latest', 'Manual trigger from Drift & Health Center'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['triggers'] }),
  });

  const events = eventsQuery.data ?? [];
  const driftSeries = [...events].reverse().filter((e) => e.drift_score != null).map((e) => ({
    ts: new Date(e.created_at).getTime(),
    value: e.drift_score as number,
  }));
  const accuracySeries = [...events].reverse().filter((e) => e.accuracy != null).map((e) => ({
    ts: new Date(e.created_at).getTime(),
    value: e.accuracy as number,
  }));

  const latest = events[0];
  const reportsPaged = usePagedSlice(reportsQuery.data ?? [], REPORTS_PAGE_SIZE);

  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="row" style={{ gap: 10 }}>
          <StatusPill severity={severityFromAlertLevel(latest?.alert_level)} label={latest?.alert_level ?? 'no data'} />
          <span className="text-muted" style={{ fontSize: 12.5 }}>
            {latest ? `as of ${new Date(latest.created_at).toLocaleString()}` : 'no monitoring events recorded yet'}
          </span>
        </div>
        <Button variant="secondary" size="sm" onClick={() => retrainMutation.mutate()} disabled={retrainMutation.isPending}>
          {retrainMutation.isPending ? 'Triggering…' : 'Trigger retraining'}
        </Button>
      </div>

      <div className="grid-cols" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <Card>
          <CardHeader title="Drift score over time (Evidently)" />
          <CardBody>
            {driftSeries.length > 0 ? (
              <LineChart series={[{ label: 'drift_score', points: driftSeries }]} yLabel="score" />
            ) : (
              <p className="text-faint" style={{ fontSize: 12.5 }}>No drift checks recorded yet for this model.</p>
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Accuracy vs. baseline" />
          <CardBody>
            {accuracySeries.length > 0 ? (
              <LineChart series={[{ label: 'accuracy', points: accuracySeries }]} yLabel="accuracy" />
            ) : (
              <p className="text-faint" style={{ fontSize: 12.5 }}>
                No accuracy data — MonitoringAgent reports None when no ground_truth_labels exist yet
                rather than fabricating a number.
              </p>
            )}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Historical drift reports" />
        <CardBody className="scroll-x">
          <table className="data-table">
            <thead>
              <tr><th>Version</th><th>Samples</th><th>Drift score</th><th>Level</th><th>Report</th><th>Date</th></tr>
            </thead>
            <tbody>
              {reportsPaged.pageItems.map((r) => (
                <tr key={r.id}>
                  <td className="mono">{r.model_version ?? '—'}</td>
                  <td>{r.sample_count ?? '—'}</td>
                  <td className="mono">{r.drift_score?.toFixed(3) ?? '—'}</td>
                  <td><StatusPill severity={severityFromAlertLevel(r.alert_level)} label={r.alert_level} /></td>
                  <td>
                    {r.r2_report_path ? (
                      <a href={r.r2_report_path} target="_blank" rel="noreferrer" style={{ color: 'var(--accent-strong)' }}>
                        Open HTML report
                      </a>
                    ) : (r.check_error ?? '—')}
                  </td>
                  <td className="text-faint">{new Date(r.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
              {(reportsQuery.data ?? []).length === 0 && (
                <tr><td colSpan={6} className="text-faint">No drift reports stored yet.</td></tr>
              )}
            </tbody>
          </table>
          <Pagination
            page={reportsPaged.page}
            pageSize={REPORTS_PAGE_SIZE}
            totalCount={reportsPaged.totalCount}
            onPageChange={reportsPaged.setPage}
          />
        </CardBody>
      </Card>
    </div>
  );
}
