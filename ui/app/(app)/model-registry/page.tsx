'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { StatusPill } from '@/components/ui/status-pill';
import { LineageGraphView } from '@/components/registry/lineage-graph';
import { Pagination } from '@/components/ui/pagination';
import {
  ApiError, compareModels, getLineage, getOciImageOrNull, getSbomOrNull, getTrivyScanOrNull,
  listModels, promoteModel,
} from '@/lib/api';
import type { RegisteredModel } from '@/lib/schemas';
import { usePagedSlice } from '@/lib/use-paged-slice';

const STAGES = ['Staging', 'Production', 'Archived'];
const MODELS_PAGE_SIZE = 20;

export default function ModelRegistryPage() {
  const [stageFilter, setStageFilter] = useState<string | undefined>(undefined);
  const [selected, setSelected] = useState<RegisteredModel | null>(null);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);

  const modelsQuery = useQuery({
    queryKey: ['models', stageFilter],
    queryFn: () => listModels(stageFilter),
  });

  const models = modelsQuery.data?.models ?? [];
  const modelsPaged = usePagedSlice(models, MODELS_PAGE_SIZE);
  useEffect(() => { modelsPaged.setPage(0); }, [stageFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <TopBar title="Model Registry" />
      <div className="row" style={{ padding: 24, gap: 20, alignItems: 'flex-start' }}>
        <div className="stack" style={{ gap: 12, width: 340, flexShrink: 0 }}>
          <div className="row" style={{ gap: 6 }}>
            {[undefined, ...STAGES].map((s) => (
              <Button
                key={s ?? 'all'}
                variant={stageFilter === s ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setStageFilter(s)}
              >
                {s ?? 'All'}
              </Button>
            ))}
          </div>
          <Card>
            <CardBody className="stack" style={{ gap: 4, padding: 8 }}>
              {modelsQuery.isLoading && <SkeletonRows />}
              {models.length === 0 && !modelsQuery.isLoading && (
                <p className="text-faint" style={{ fontSize: 12.5, padding: 8 }}>
                  No models registered{stageFilter ? ` in ${stageFilter}` : ''} yet.
                </p>
              )}
              {modelsPaged.pageItems.map((m) => {
                const key = `${m.name}::${m.version}`;
                const isSelected = selected?.name === m.name && selected.version === m.version;
                return (
                  <div
                    key={key}
                    onClick={() => setSelected({ ...m, tags: m.tags ?? {} })}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 8,
                      cursor: 'pointer',
                      background: isSelected ? 'var(--accent-soft)' : 'transparent',
                    }}
                  >
                    <div className="row" style={{ justifyContent: 'space-between' }}>
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</span>
                      <span className="text-faint mono" style={{ fontSize: 11.5 }}>
                        v{m.version}
                      </span>
                    </div>
                    <div className="row" style={{ gap: 8, marginTop: 4 }}>
                      {m.stage && <StatusPill severity="accent" label={m.stage} />}
                      <label
                        className="row"
                        style={{ gap: 4, fontSize: 11 }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={compareSelection.includes(m.version)}
                          onChange={(e) =>
                            setCompareSelection((prev) =>
                              e.target.checked ? [...prev, m.version] : prev.filter((v) => v !== m.version),
                            )
                          }
                        />
                        compare
                      </label>
                    </div>
                  </div>
                );
              })}
              <Pagination
                page={modelsPaged.page}
                pageSize={MODELS_PAGE_SIZE}
                totalCount={modelsPaged.totalCount}
                onPageChange={modelsPaged.setPage}
              />
            </CardBody>
          </Card>
          {compareSelection.length >= 2 && selected && (
            <CompareCard modelName={selected.name} versions={compareSelection} />
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          {selected ? (
            <ModelDetail model={selected} />
          ) : (
            <Card>
              <CardBody>
                <p className="text-faint" style={{ fontSize: 13 }}>Select a model version to view its detail.</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

function ModelDetail({ model }: { model: RegisteredModel }) {
  const qc = useQueryClient();
  const [targetStage, setTargetStage] = useState('Production');
  const [triggeredBy, setTriggeredBy] = useState('');

  const lineageQuery = useQuery({
    queryKey: ['lineage', model.name, model.version],
    queryFn: () => getLineage(model.name, model.version),
  });

  const ociQuery = useQuery({
    queryKey: ['oci-image', model.name, model.version],
    queryFn: () => getOciImageOrNull(model.name, model.version),
  });

  const sbomQuery = useQuery({
    queryKey: ['sbom', model.name, model.version],
    queryFn: () => getSbomOrNull(model.name, model.version),
  });

  const trivyQuery = useQuery({
    queryKey: ['trivy', ociQuery.data?.image_digest],
    queryFn: () => getTrivyScanOrNull(ociQuery.data!.image_digest),
    enabled: Boolean(ociQuery.data?.image_digest),
  });

  const promoteMutation = useMutation({
    mutationFn: () => promoteModel(model.name, model.version, targetStage, triggeredBy || 'ui-user'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }),
  });

  return (
    <div className="stack" style={{ gap: 20 }}>
      <Card>
        <CardHeader
          title={`${model.name} v${model.version}`}
          action={<StatusPill severity="accent" label={model.stage ?? 'None'} />}
        />
        <CardBody className="row" style={{ gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <label className="stack" style={{ gap: 4 }}>
            <span className="text-faint" style={{ fontSize: 11 }}>Target stage</span>
            <select value={targetStage} onChange={(e) => setTargetStage(e.target.value)} style={selectStyle}>
              {STAGES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <label className="stack" style={{ gap: 4 }}>
            <span className="text-faint" style={{ fontSize: 11 }}>Triggered by</span>
            <input
              value={triggeredBy}
              onChange={(e) => setTriggeredBy(e.target.value)}
              placeholder="github-username"
              style={selectStyle}
            />
          </label>
          <Button onClick={() => promoteMutation.mutate()} disabled={promoteMutation.isPending}>
            {promoteMutation.isPending ? 'Promoting…' : `Promote to ${targetStage}`}
          </Button>
          {promoteMutation.isSuccess && !promoteMutation.data.success && (
            <span style={{ color: 'var(--crit)', fontSize: 12.5 }}>
              Rejected — gates failed: {promoteMutation.data.gates_failed.join(', ')}
            </span>
          )}
          {promoteMutation.isSuccess && promoteMutation.data.success && (
            <span style={{ color: 'var(--ok)', fontSize: 12.5 }}>Promoted successfully.</span>
          )}
          {promoteMutation.isError && (
            <span style={{ color: 'var(--crit)', fontSize: 12.5 }}>
              {promoteMutation.error instanceof ApiError ? promoteMutation.error.detail : 'Promotion failed'}
            </span>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Lineage — dataset → model → deployment" />
        <CardBody>
          {lineageQuery.isLoading ? (
            <SkeletonBlock height={200} />
          ) : lineageQuery.data ? (
            <LineageGraphView graph={lineageQuery.data} />
          ) : (
            <p className="text-faint" style={{ fontSize: 12.5 }}>Lineage query failed.</p>
          )}
        </CardBody>
      </Card>

      <div className="grid-cols" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <Card>
          <CardHeader title="OCI Image" />
          <CardBody>
            {ociQuery.isLoading ? (
              <SkeletonBlock height={80} />
            ) : ociQuery.data ? (
              <div className="stack" style={{ gap: 8, fontSize: 12.5 }}>
                <Row label="Pull command">
                  <code className="mono">docker pull {ociQuery.data.image_uri}</code>
                </Row>
                <Row label="Digest"><span className="mono">{ociQuery.data.image_digest.slice(0, 24)}…</span></Row>
                <Row label="Registry">{ociQuery.data.registry_host}</Row>
                <Row label="Pushed">{ociQuery.data.pushed_at ?? '—'}</Row>
              </div>
            ) : (
              <p className="text-faint" style={{ fontSize: 12.5 }}>No OCI image built for this version yet.</p>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Security (Trivy + SBOM)" />
          <CardBody>
            {trivyQuery.data ? (
              <div className="stack" style={{ gap: 6, fontSize: 12.5 }}>
                <Row label="Scan result">
                  <StatusPill severity={trivyQuery.data.passed ? 'ok' : 'crit'} label={trivyQuery.data.passed ? 'Passed' : 'Failed'} />
                </Row>
                <Row label="Critical / High">
                  {trivyQuery.data.critical_count} / {trivyQuery.data.high_count}
                </Row>
                <Row label="Total findings">{trivyQuery.data.total_count}</Row>
              </div>
            ) : (
              <p className="text-faint" style={{ fontSize: 12.5, marginBottom: 10 }}>
                {ociQuery.data ? 'No Trivy scan recorded for this image.' : 'No image to scan yet.'}
              </p>
            )}
            {sbomQuery.data ? (
              <Row label="SBOM">
                {sbomQuery.data.package_count} packages ({sbomQuery.data.format})
              </Row>
            ) : (
              <p className="text-faint" style={{ fontSize: 12 }}>No SBOM generated yet.</p>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function CompareCard({ modelName, versions }: { modelName: string; versions: string[] }) {
  const compareMutation = useMutation({
    mutationFn: () => compareModels(modelName, versions.slice(0, 4), 'ui-user'),
  });

  return (
    <Card>
      <CardHeader title={`Compare ${versions.length} versions`} />
      <CardBody className="stack" style={{ gap: 8 }}>
        <Button size="sm" onClick={() => compareMutation.mutate()} disabled={compareMutation.isPending}>
          {compareMutation.isPending ? 'Comparing…' : 'Run comparison'}
        </Button>
        {compareMutation.data && (
          <div className="stack" style={{ gap: 4, fontSize: 12 }}>
            <span className="text-muted">
              Recommended: <b className="mono">{compareMutation.data.recommended_version ?? '—'}</b>
            </span>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="row" style={{ justifyContent: 'space-between', gap: 12 }}>
      <span className="text-faint">{label}</span>
      <span style={{ textAlign: 'right', wordBreak: 'break-all' }}>{children}</span>
    </div>
  );
}

function SkeletonRows() {
  return (
    <div className="stack" style={{ gap: 6, padding: 8 }}>
      {[0, 1, 2].map((i) => (
        <div key={i} style={{ height: 44, background: 'var(--bg-inset)', borderRadius: 8 }} />
      ))}
    </div>
  );
}

function SkeletonBlock({ height }: { height: number }) {
  return <div style={{ height, background: 'var(--bg-inset)', borderRadius: 8 }} />;
}

const selectStyle: React.CSSProperties = {
  padding: '7px 10px',
  borderRadius: 6,
  border: '1px solid var(--border-strong)',
  background: 'var(--bg)',
  color: 'var(--text)',
  fontSize: 13,
};
