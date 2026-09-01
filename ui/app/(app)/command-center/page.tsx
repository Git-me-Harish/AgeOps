'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { StatusPill, severityFromAlertLevel, severityFromPodStatus } from '@/components/ui/status-pill';
import { LineChart } from '@/components/charts/line-chart';
import { Pagination } from '@/components/ui/pagination';
import { StartPipelineDialog } from '@/components/workflow/start-pipeline-dialog';
import { listAgentStatus, listTriggers, listWorkflows } from '@/lib/api';
import { promQueryRange } from '@/lib/prometheus';
import { useEventStream } from '@/lib/ws';
import type { WsEvent } from '@/lib/schemas';

const WORKFLOWS_PAGE_SIZE = 20;

export default function CommandCenterPage() {
  const qc = useQueryClient();
  const [liveEvents, setLiveEvents] = useState<WsEvent[]>([]);
  const [workflowPage, setWorkflowPage] = useState(0);

  const { connected } = useEventStream((event) => {
    setLiveEvents((prev) => [event, ...prev].slice(0, 25));
    if (event.channel?.startsWith('workflow.')) {
      qc.invalidateQueries({ queryKey: ['workflows'] });
    }
  });

  const workflowsQuery = useQuery({
    queryKey: ['workflows', workflowPage],
    queryFn: () => listWorkflows(WORKFLOWS_PAGE_SIZE, workflowPage * WORKFLOWS_PAGE_SIZE),
    refetchInterval: connected ? false : 10_000, // WS keeps this fresh when live; poll only as fallback
  });

  const agentStatusQuery = useQuery({
    queryKey: ['agent-status'],
    queryFn: listAgentStatus,
    refetchInterval: 20_000,
  });

  const triggersQuery = useQuery({
    queryKey: ['triggers'],
    queryFn: listTriggers,
    refetchInterval: 30_000,
  });

  const throughputQuery = useQuery({
    queryKey: ['prom-throughput'],
    queryFn: async () => {
      const now = Math.floor(Date.now() / 1000);
      const series = await promQueryRange(
        'sum(rate(mlops_predictions_total[5m]))',
        now - 3600,
        now,
        60,
      );
      return series;
    },
    refetchInterval: 60_000,
  });

  const workflows = workflowsQuery.data?.workflows ?? [];
  // Real global aggregates from the backend (COUNT(*) FILTER), not derived
  // from the current 20-row page — deriving from the page alone would
  // under-report once there's more than one page of workflows.
  const running = workflowsQuery.data?.running_count ?? 0;
  const awaitingApproval = workflowsQuery.data?.awaiting_approval_count ?? 0;
  const agents = agentStatusQuery.data?.agents ?? [];
  const onlineAgents = agents.filter((a) => a.status === 'online').length;
  const triggers = triggersQuery.data ?? [];
  const pendingTriggers = triggers.filter((t) => t.status === 'pending').length;

  return (
    <>
      <TopBar title="Command Center" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="grid-cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))', flex: 1, marginRight: 20 }}>
            <StatCard label="Workflows running" value={running} />
            <StatCard label="Awaiting approval" value={awaitingApproval} accent={awaitingApproval > 0 ? 'warn' : undefined} />
            <StatCard label="Agents online" value={`${onlineAgents} / ${agents.length}`} />
            <StatCard label="Pending triggers" value={pendingTriggers} accent={pendingTriggers > 0 ? 'accent' : undefined} />
          </div>
          <StartPipelineDialog />
        </div>

        <div className="grid-cols" style={{ gridTemplateColumns: '1.4fr 1fr' }}>
          <Card>
            <CardHeader title="Serving throughput — sum(rate(mlops_predictions_total[5m]))" />
            <CardBody>
              {throughputQuery.isLoading ? (
                <SkeletonBlock height={220} />
              ) : (
                <LineChart
                  series={[
                    {
                      label: 'requests/sec',
                      points: (throughputQuery.data?.[0]?.points ?? []).map((p) => ({ ts: p.ts, value: p.value })),
                    },
                  ]}
                  yLabel="req/s"
                />
              )}
              {throughputQuery.data?.length === 0 && (
                <p className="text-faint" style={{ fontSize: 12, marginTop: 8 }}>
                  No data — either no traffic in the last hour or Prometheus is unreachable from this UI instance.
                </p>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Agent health (real pod status)" />
            <CardBody>
              {agentStatusQuery.isLoading ? (
                <SkeletonBlock height={160} />
              ) : (
                <div className="stack" style={{ gap: 8 }}>
                  {agents.map((a) => (
                    <div key={a.agent_id} className="row" style={{ justifyContent: 'space-between', fontSize: 13 }}>
                      <span className="mono">{a.agent_id}</span>
                      <StatusPill severity={severityFromPodStatus(a.status)} label={a.status} />
                    </div>
                  ))}
                  {agents.length === 0 && <p className="text-faint" style={{ fontSize: 12.5 }}>No agents registered.</p>}
                </div>
              )}
            </CardBody>
          </Card>
        </div>

        <div className="grid-cols" style={{ gridTemplateColumns: '1.4fr 1fr' }}>
          <Card>
            <CardHeader title="Workflow stream" />
            <CardBody className="scroll-x">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Workflow</th>
                    <th>Stage</th>
                    <th>Status</th>
                    <th>Source</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {workflows.map((w) => (
                    <tr key={w.workflow_id}>
                      <td className="mono">{w.workflow_id.slice(0, 12)}</td>
                      <td>{w.current_stage ?? '—'}</td>
                      <td>
                        <StatusPill
                          severity={(w.errors ?? []).length ? 'crit' : w.awaiting_approval ? 'warn' : w.status === 'completed' ? 'ok' : 'accent'}
                          label={w.awaiting_approval ? 'awaiting approval' : w.status}
                        />
                      </td>
                      <td className="text-muted">{w.source ?? 'manual'}</td>
                      <td className="text-faint">{w.updated_at ? new Date(w.updated_at).toLocaleTimeString() : '—'}</td>
                    </tr>
                  ))}
                  {workflows.length === 0 && (
                    <tr>
                      <td colSpan={5} className="text-faint">
                        No workflows yet — launch one above.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              <Pagination
                page={workflowPage}
                pageSize={WORKFLOWS_PAGE_SIZE}
                totalCount={workflowsQuery.data?.total_count ?? 0}
                onPageChange={setWorkflowPage}
              />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Alerts (live + workflow_triggers)" />
            <CardBody>
              <div className="stack" style={{ gap: 10 }}>
                {liveEvents
                  .filter((e) => e.channel?.startsWith('alert.'))
                  .map((e, i) => (
                    <AlertRow
                      key={`live-${i}`}
                      severity={severityFromAlertLevel(String(e.severity ?? 'unknown'))}
                      text={`${e.model_name ?? 'unknown model'} — ${(e.alerts as string[] | undefined)?.join('; ') ?? e.severity}`}
                    />
                  ))}
                {triggers.slice(0, 8).map((t) => (
                  <AlertRow
                    key={t.id}
                    severity={t.status === 'pending' ? 'warn' : 'unknown'}
                    text={`${t.model_name} — ${t.reason ?? t.trigger_type} (${t.status})`}
                  />
                ))}
                {liveEvents.length === 0 && triggers.length === 0 && (
                  <p className="text-faint" style={{ fontSize: 12.5 }}>No alerts.</p>
                )}
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string | number; accent?: 'warn' | 'accent' }) {
  return (
    <Card>
      <CardBody>
        <div className="text-faint" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {label}
        </div>
        <div
          className="mono"
          style={{
            fontSize: 26,
            fontWeight: 700,
            marginTop: 4,
            color: accent === 'warn' ? 'var(--warn)' : accent === 'accent' ? 'var(--accent-strong)' : 'var(--text)',
          }}
        >
          {value}
        </div>
      </CardBody>
    </Card>
  );
}

function AlertRow({ severity, text }: { severity: 'ok' | 'warn' | 'crit' | 'unknown' | 'accent'; text: string }) {
  return (
    <div className="row" style={{ gap: 10, fontSize: 12.5 }}>
      <StatusPill severity={severity} label="" />
      <span style={{ flex: 1 }}>{text}</span>
    </div>
  );
}

function SkeletonBlock({ height }: { height: number }) {
  return <div style={{ height, background: 'var(--bg-inset)', borderRadius: 8 }} />;
}
