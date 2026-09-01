'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { StatusPill, severityFromPodStatus } from '@/components/ui/status-pill';
import { Pagination } from '@/components/ui/pagination';
import { decideRecommendation, getAgentActivity, getAuditLog, listAgents, listAgentStatus, listRecommendations } from '@/lib/api';
import type { Recommendation } from '@/lib/schemas';
import { usePagedSlice } from '@/lib/use-paged-slice';

const AUDIT_LOG_PAGE_SIZE = 20;
const ACTIVITY_PAGE_SIZE = 20;

export default function AgentIntelligencePage() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const agentsQuery = useQuery({ queryKey: ['agents'], queryFn: listAgents });
  const statusQuery = useQuery({ queryKey: ['agent-status'], queryFn: listAgentStatus, refetchInterval: 20_000 });
  const recsQuery = useQuery({ queryKey: ['recommendations'], queryFn: listRecommendations, refetchInterval: 30_000 });
  const auditQuery = useQuery({ queryKey: ['audit-log'], queryFn: () => getAuditLog(30) });

  const agents = agentsQuery.data?.agents ?? [];
  const statusByAgent = new Map((statusQuery.data?.agents ?? []).map((a) => [a.agent_id, a]));
  const auditPaged = usePagedSlice(auditQuery.data?.decisions ?? [], AUDIT_LOG_PAGE_SIZE);

  return (
    <>
      <TopBar title="Agent Intelligence" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <div className="grid-cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))' }}>
          {agents.map((a) => {
            const status = statusByAgent.get(a.agent_id);
            return (
              <Card key={a.agent_id} className="stack" style={{ cursor: 'pointer' }}>
                <CardBody onClick={() => setSelectedAgent(a.agent_id)}>
                  <div className="row" style={{ justifyContent: 'space-between' }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{a.name}</span>
                    <StatusPill severity={severityFromPodStatus(status?.status)} label={status?.status ?? 'unknown'} />
                  </div>
                  <p className="text-faint" style={{ fontSize: 11.5, marginTop: 6, marginBottom: 0 }}>
                    {(a.capabilities ?? []).slice(0, 3).join(', ') || 'no declared capabilities'}
                  </p>
                </CardBody>
              </Card>
            );
          })}
        </div>

        {selectedAgent && <AgentActivityPanel agentRole={selectedAgent} />}

        <div className="grid-cols" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <Card>
            <CardHeader title="RL recommendations" />
            <CardBody className="stack" style={{ gap: 10 }}>
              {(recsQuery.data ?? []).map((r) => (
                <RecommendationRow key={r.id} rec={r} />
              ))}
              {(recsQuery.data ?? []).length === 0 && (
                <p className="text-faint" style={{ fontSize: 12.5 }}>No pending recommendations.</p>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="OPA audit log" />
            <CardBody className="scroll-x">
              <table className="data-table">
                <thead><tr><th>Policy</th><th>Agent</th><th>Decision</th><th>Sidecar</th><th>When</th></tr></thead>
                <tbody>
                  {auditPaged.pageItems.map((d, i) => (
                    <tr key={i}>
                      <td className="mono">{d.policy_name}</td>
                      <td>{d.agent_role ?? '—'}</td>
                      <td>
                        <StatusPill severity={d.decision ? 'ok' : 'crit'} label={d.decision ? 'Allow' : 'Deny'} />
                      </td>
                      <td>{d.was_sidecar ? 'yes' : 'no (network)'}</td>
                      <td className="text-faint">{d.created_at ? new Date(d.created_at).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                  {auditPaged.pageItems.length === 0 && (
                    <tr><td colSpan={5} className="text-faint">No policy decisions recorded yet.</td></tr>
                  )}
                </tbody>
              </table>
              <Pagination
                page={auditPaged.page}
                pageSize={AUDIT_LOG_PAGE_SIZE}
                totalCount={auditPaged.totalCount}
                onPageChange={auditPaged.setPage}
              />
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}

function AgentActivityPanel({ agentRole }: { agentRole: string }) {
  const activityQuery = useQuery({ queryKey: ['agent-activity', agentRole], queryFn: () => getAgentActivity(agentRole) });
  const s = activityQuery.data?.summary;
  const callsPaged = usePagedSlice(activityQuery.data?.recent_calls ?? [], ACTIVITY_PAGE_SIZE);

  return (
    <Card>
      <CardHeader title={`Activity — ${agentRole}`} />
      <CardBody>
        <div className="grid-cols" style={{ gridTemplateColumns: 'repeat(5, minmax(0,1fr))', marginBottom: 16 }}>
          <Metric label="Calls" value={s?.call_count ?? '—'} />
          <Metric label="Success rate" value={s?.success_rate != null ? `${(s.success_rate * 100).toFixed(0)}%` : '—'} />
          <Metric label="Avg latency" value={s?.avg_latency_ms != null ? `${s.avg_latency_ms.toFixed(0)}ms` : '—'} />
          <Metric label="Total tokens" value={s?.total_tokens ?? '—'} />
          <Metric label="Total cost" value={s?.total_cost_usd != null ? `$${s.total_cost_usd.toFixed(4)}` : '—'} />
        </div>
        <table className="data-table">
          <thead><tr><th>Workflow</th><th>Model</th><th>Tokens</th><th>Cost</th><th>Latency</th><th>When</th></tr></thead>
          <tbody>
            {callsPaged.pageItems.map((c, i) => (
              <tr key={i}>
                <td className="mono">{c.workflow_id?.slice(0, 10) ?? '—'}</td>
                <td>{c.model}{c.was_fallback ? ' (fallback)' : ''}{c.was_cache_hit ? ' (cached)' : ''}</td>
                <td>{c.total_tokens ?? '—'}</td>
                <td>{c.cost_usd != null ? `$${c.cost_usd.toFixed(5)}` : '—'}</td>
                <td>{c.latency_ms ?? '—'}ms</td>
                <td className="text-faint">{c.created_at ? new Date(c.created_at).toLocaleString() : '—'}</td>
              </tr>
            ))}
            {callsPaged.pageItems.length === 0 && (
              <tr><td colSpan={6} className="text-faint">No LLM calls recorded for this agent role yet.</td></tr>
            )}
          </tbody>
        </table>
        <Pagination
          page={callsPaged.page}
          pageSize={ACTIVITY_PAGE_SIZE}
          totalCount={callsPaged.totalCount}
          onPageChange={callsPaged.setPage}
        />
      </CardBody>
    </Card>
  );
}

function RecommendationRow({ rec }: { rec: Recommendation }) {
  const qc = useQueryClient();
  const mutation = useMutation({
    mutationFn: (decision: 'accepted' | 'rejected') => decideRecommendation(rec.id, decision, 'ui-user'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recommendations'] }),
  });

  return (
    <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <span className="mono" style={{ fontSize: 12.5 }}>{rec.agent_role} — {rec.status}</span>
        {rec.confidence != null && <span className="text-faint" style={{ fontSize: 11.5 }}>confidence {(rec.confidence * 100).toFixed(0)}%</span>}
      </div>
      <pre className="mono scroll-x" style={{ fontSize: 11, background: 'var(--bg-inset)', padding: 8, borderRadius: 6, marginTop: 6 }}>
        {JSON.stringify(rec.recommendation, null, 2)}
      </pre>
      {rec.status === 'pending' && (
        <div className="row" style={{ gap: 6, marginTop: 6 }}>
          <Button size="sm" onClick={() => mutation.mutate('accepted')} disabled={mutation.isPending}>Accept</Button>
          <Button size="sm" variant="ghost" onClick={() => mutation.mutate('rejected')} disabled={mutation.isPending}>Reject</Button>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-faint" style={{ fontSize: 11, textTransform: 'uppercase' }}>{label}</div>
      <div className="mono" style={{ fontSize: 18, fontWeight: 700, marginTop: 4 }}>{value}</div>
    </div>
  );
}
