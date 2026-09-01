'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Pagination } from '@/components/ui/pagination';
import { ParallelCoordinates } from '@/components/experiments/parallel-coordinates';
import { TraceView } from '@/components/experiments/trace-view';
import { listExperimentRuns } from '@/lib/api';
import type { ExperimentRun } from '@/lib/schemas';

const RUNS_PAGE_SIZE = 20;

export default function ExperimentLabPage() {
  const [runsPage, setRunsPage] = useState(0);
  const runsQuery = useQuery({
    queryKey: ['experiment-runs', runsPage],
    queryFn: () => listExperimentRuns(RUNS_PAGE_SIZE, runsPage * RUNS_PAGE_SIZE),
  });
  const runs: ExperimentRun[] = (runsQuery.data?.runs ?? []).map((r) => ({
    ...r, metrics: r.metrics ?? {}, params: r.params ?? {},
  }));
  const [selected, setSelected] = useState<string[]>([]);
  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);

  const metricKeys = Array.from(new Set(runs.flatMap((r) => Object.keys(r.metrics)))).sort();
  const selectedRuns = runs.filter((r) => selected.includes(r.run_id));

  return (
    <>
      <TopBar title="Experiment Lab" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <Card>
          <CardHeader title="Runs — real mlflow.search_runs()" />
          <CardBody className="scroll-x">
            <table className="data-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Run</th>
                  <th>Status</th>
                  {metricKeys.map((k) => <th key={k}>{k}</th>)}
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <RunRow
                    key={r.run_id}
                    run={r}
                    metricKeys={metricKeys}
                    checked={selected.includes(r.run_id)}
                    onToggle={(checked) =>
                      setSelected((prev) => (checked ? [...prev, r.run_id] : prev.filter((id) => id !== r.run_id)))
                    }
                    expanded={expandedTrace === r.run_id}
                    onToggleExpand={() => setExpandedTrace((prev) => (prev === r.run_id ? null : r.run_id))}
                  />
                ))}
                {runs.length === 0 && !runsQuery.isLoading && (
                  <tr><td colSpan={4 + metricKeys.length} className="text-faint">No runs recorded in this experiment yet.</td></tr>
                )}
              </tbody>
            </table>
            <Pagination
              page={runsPage}
              pageSize={RUNS_PAGE_SIZE}
              totalCount={runsQuery.data?.total_count ?? 0}
              onPageChange={setRunsPage}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Metric comparison — parallel coordinates" />
          <CardBody>
            <ParallelCoordinates runs={selectedRuns.length > 0 ? selectedRuns : runs.slice(0, 10)} />
          </CardBody>
        </Card>

        {selectedRuns.length === 2 && <RunDiff a={selectedRuns[0]!} b={selectedRuns[1]!} />}
      </div>
    </>
  );
}

function RunRow({ run, metricKeys, checked, onToggle, expanded, onToggleExpand }: {
  run: ExperimentRun; metricKeys: string[]; checked: boolean; onToggle: (c: boolean) => void;
  expanded: boolean; onToggleExpand: () => void;
}) {
  return (
    <>
      <tr onClick={onToggleExpand} style={{ cursor: 'pointer' }}>
        <td onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={checked} onChange={(e) => onToggle(e.target.checked)} />
        </td>
        <td className="mono">{run.run_name ?? run.run_id.slice(0, 10)}</td>
        <td>{run.status}</td>
        {metricKeys.map((k) => <td key={k} className="mono">{run.metrics[k]?.toFixed(3) ?? '—'}</td>)}
        <td className="text-faint">{run.start_time ? new Date(run.start_time).toLocaleString() : '—'}</td>
      </tr>
      {expanded && (
        <tr>
          <td></td>
          <td colSpan={2 + metricKeys.length}>
            <div style={{ padding: '8px 0' }}>
              <TraceView runId={run.run_id} />
            </div>
          </td>
          <td></td>
        </tr>
      )}
    </>
  );
}

function RunDiff({ a, b }: { a: ExperimentRun; b: ExperimentRun }) {
  const keys = Array.from(new Set([...Object.keys(a.params), ...Object.keys(b.params)])).sort();
  const metricKeys = Array.from(new Set([...Object.keys(a.metrics), ...Object.keys(b.metrics)])).sort();
  return (
    <Card>
      <CardHeader title={`Diff: ${a.run_name ?? a.run_id.slice(0, 8)} vs ${b.run_name ?? b.run_id.slice(0, 8)}`} />
      <CardBody className="scroll-x">
        <table className="data-table">
          <thead><tr><th>Field</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
          <tbody>
            {keys.map((k) => (
              <tr key={`p-${k}`}>
                <td>param.{k}</td>
                <td className="mono">{a.params[k] ?? '—'}</td>
                <td className="mono">{b.params[k] ?? '—'}</td>
                <td className="mono" style={{ color: a.params[k] !== b.params[k] ? 'var(--warn)' : undefined }}>
                  {a.params[k] !== b.params[k] ? 'changed' : '—'}
                </td>
              </tr>
            ))}
            {metricKeys.map((k) => {
              const av = a.metrics[k];
              const bv = b.metrics[k];
              const delta = av != null && bv != null ? bv - av : null;
              return (
                <tr key={`m-${k}`}>
                  <td>metric.{k}</td>
                  <td className="mono">{av?.toFixed(4) ?? '—'}</td>
                  <td className="mono">{bv?.toFixed(4) ?? '—'}</td>
                  <td className="mono" style={{ color: delta && delta > 0 ? 'var(--ok)' : delta && delta < 0 ? 'var(--crit)' : undefined }}>
                    {delta != null ? (delta > 0 ? '+' : '') + delta.toFixed(4) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardBody>
    </Card>
  );
}
