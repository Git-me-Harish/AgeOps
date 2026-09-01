'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { TopBar } from '@/components/layout/top-bar';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { StatusPill } from '@/components/ui/status-pill';
import { Pagination } from '@/components/ui/pagination';
import { OpaPolicyEditor } from '@/components/security/opa-policy-editor';
import {
  getAuditLog, getOciImageOrNull, getPromotionHistory, getSbomOrNull, getTrivyScanOrNull, listModels,
} from '@/lib/api';
import { usePagedSlice } from '@/lib/use-paged-slice';

const DENIALS_PAGE_SIZE = 20;

export default function SecurityPage() {
  const [selected, setSelected] = useState<{ name: string; version: string } | null>(null);
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: () => listModels() });
  const models = modelsQuery.data?.models ?? [];

  useEffect(() => {
    if (!selected && models.length > 0) setSelected({ name: models[0]!.name, version: models[0]!.version });
  }, [models, selected]);

  const auditQuery = useQuery({ queryKey: ['audit-log', 'security'], queryFn: () => getAuditLog(200) });
  const denies = (auditQuery.data?.decisions ?? []).filter((d) => !d.decision);
  const deniesPaged = usePagedSlice(denies, DENIALS_PAGE_SIZE);

  return (
    <>
      <TopBar title="Security & Compliance" />
      <div className="stack" style={{ padding: 24, gap: 20 }}>
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          {models.map((m) => (
            <button
              key={`${m.name}-${m.version}`}
              className="btn btn-secondary btn-sm"
              style={{
                background: selected?.name === m.name && selected.version === m.version ? 'var(--accent-soft)' : undefined,
              }}
              onClick={() => setSelected({ name: m.name, version: m.version })}
            >
              {m.name} v{m.version}
            </button>
          ))}
        </div>

        {selected && <ImageSecurityCard modelName={selected.name} modelVersion={selected.version} />}

        <div className="grid-cols" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <Card>
            <CardHeader title="Recent policy denials" />
            <CardBody className="stack" style={{ gap: 8 }}>
              {deniesPaged.pageItems.map((d, i) => (
                <div key={i} style={{ fontSize: 12.5 }}>
                  <div className="row" style={{ justifyContent: 'space-between' }}>
                    <span className="mono">{d.policy_name}</span>
                    <span className="text-faint">{d.created_at ? new Date(d.created_at).toLocaleString() : ''}</span>
                  </div>
                  <p className="text-muted" style={{ margin: '2px 0 0' }}>{(d.deny_reasons ?? []).join('; ') || 'no reason recorded'}</p>
                </div>
              ))}
              {deniesPaged.pageItems.length === 0 && <p className="text-faint" style={{ fontSize: 12.5 }}>No policy denials recorded.</p>}
              <Pagination
                page={deniesPaged.page}
                pageSize={DENIALS_PAGE_SIZE}
                totalCount={deniesPaged.totalCount}
                onPageChange={deniesPaged.setPage}
              />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Secret scanner" />
            <CardBody>
              <p className="text-faint" style={{ fontSize: 12.5, marginBottom: 10 }}>
                No secret-scanning tool (trufflehog/gitleaks) is wired into this repo&apos;s build pipeline yet
                — left disabled rather than faking a scan result.
              </p>
              <button className="btn btn-secondary btn-sm" disabled>Scan R2 buckets (unavailable)</button>
            </CardBody>
          </Card>
        </div>

        <Card>
          <CardHeader title="OPA policy editor — mlops-opa-policy ConfigMap" />
          <CardBody>
            <OpaPolicyEditor />
          </CardBody>
        </Card>
      </div>
    </>
  );
}

function ImageSecurityCard({ modelName, modelVersion }: { modelName: string; modelVersion: string }) {
  const ociQuery = useQuery({ queryKey: ['oci-image', modelName, modelVersion], queryFn: () => getOciImageOrNull(modelName, modelVersion) });
  const trivyQuery = useQuery({
    queryKey: ['trivy', ociQuery.data?.image_digest],
    queryFn: () => getTrivyScanOrNull(ociQuery.data!.image_digest),
    enabled: Boolean(ociQuery.data?.image_digest),
  });
  const sbomQuery = useQuery({ queryKey: ['sbom', modelName, modelVersion], queryFn: () => getSbomOrNull(modelName, modelVersion) });
  const historyQuery = useQuery({ queryKey: ['promotion-history', modelName], queryFn: () => getPromotionHistory(modelName) });

  return (
    <Card>
      <CardHeader title={`${modelName} v${modelVersion} — compliance summary`} />
      <CardBody>
        <div className="grid-cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))' }}>
          <Metric label="Trivy result" value={
            trivyQuery.data ? <StatusPill severity={trivyQuery.data.passed ? 'ok' : 'crit'} label={trivyQuery.data.passed ? 'Passed' : 'Failed'} /> : '—'
          } />
          <Metric label="Critical / High CVEs" value={trivyQuery.data ? `${trivyQuery.data.critical_count} / ${trivyQuery.data.high_count}` : '—'} />
          <Metric label="SBOM packages" value={sbomQuery.data ? String(sbomQuery.data.package_count) : '—'} />
          <Metric label="Image signed" value={ociQuery.data ? 'OCI image present' : 'not packaged'} />
        </div>
        <p className="text-faint" style={{ fontSize: 11.5, marginTop: 14 }}>
          Compiled directly from packaging_server.py&apos;s real Trivy/SBOM records and registry_server.py&apos;s
          promotion history ({(historyQuery.data as any)?.promotions?.length ?? 0} promotion event(s) on file) —
          this is real data assembled into a summary, not a generated PDF against fabricated GDPR/SOC2
          checklist items with no backing data.
        </p>
      </CardBody>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-faint" style={{ fontSize: 11, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>{value}</div>
    </div>
  );
}
