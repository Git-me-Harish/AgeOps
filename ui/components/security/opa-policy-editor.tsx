'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { ApiError, getOpaPolicy, putOpaPolicy } from '@/lib/api';

/** Real read/validate/write against the mlops-opa-policy ConfigMap (agents/opa_admin.py). */
export function OpaPolicyEditor() {
  const policyQuery = useQuery({ queryKey: ['opa-policy'], queryFn: getOpaPolicy });
  const [draft, setDraft] = useState<string | null>(null);

  useEffect(() => {
    if (policyQuery.data && draft === null) setDraft(policyQuery.data.rego);
  }, [policyQuery.data, draft]);

  const saveMutation = useMutation({
    mutationFn: () => putOpaPolicy(draft ?? ''),
  });

  const dirty = draft !== null && draft !== policyQuery.data?.rego;

  if (policyQuery.isLoading) return <p className="text-faint" style={{ fontSize: 12.5 }}>Loading policy…</p>;
  if (policyQuery.isError) {
    return (
      <p style={{ color: 'var(--crit)', fontSize: 12.5 }}>
        Could not read the policy ConfigMap: {policyQuery.error instanceof ApiError ? policyQuery.error.detail : 'unknown error'}.
        Likely means no Kubernetes cluster is reachable from this environment (expected in local dev).
      </p>
    );
  }

  return (
    <div className="stack" style={{ gap: 10 }}>
      <textarea
        value={draft ?? ''}
        onChange={(e) => setDraft(e.target.value)}
        rows={18}
        spellCheck={false}
        style={{
          width: '100%', fontFamily: 'var(--font-mono)', fontSize: 12.5, lineHeight: 1.5,
          padding: 12, borderRadius: 8, border: '1px solid var(--border-strong)',
          background: 'var(--bg-inset)', color: 'var(--text)', resize: 'vertical',
        }}
      />
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          {saveMutation.isSuccess && (
            <span style={{ color: 'var(--ok)', fontSize: 12.5 }}>
              Saved — checked with {saveMutation.data.checked_with}. Sidecar picks it up on its own within ~60s (--watch).
            </span>
          )}
          {saveMutation.isError && (
            <span style={{ color: 'var(--crit)', fontSize: 12.5 }}>
              {saveMutation.error instanceof ApiError ? saveMutation.error.detail : 'Save failed'}
            </span>
          )}
        </div>
        <div className="row" style={{ gap: 8 }}>
          <Button variant="ghost" size="sm" disabled={!dirty} onClick={() => setDraft(policyQuery.data?.rego ?? '')}>
            Revert
          </Button>
          <Button size="sm" disabled={!dirty || saveMutation.isPending} onClick={() => saveMutation.mutate()}>
            {saveMutation.isPending ? 'Validating & saving…' : 'Save policy'}
          </Button>
        </div>
      </div>
    </div>
  );
}
