'use client';

import * as Dialog from '@radix-ui/react-dialog';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { startWorkflow } from '@/lib/api';

type Step = 'source' | 'review';

/** Real multi-step wizard (plan §Page 1) — the only "single text input" left is the final confirm. */
export function StartPipelineDialog() {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState<Step>('source');
  const [datasetUri, setDatasetUri] = useState('');
  const [modelUri, setModelUri] = useState('');
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => startWorkflow(datasetUri, modelUri || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workflows'] });
      setOpen(false);
      setStep('source');
      setDatasetUri('');
      setModelUri('');
    },
  });

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <Button>Start Pipeline</Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)' }} />
        <Dialog.Content
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 460,
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-md)',
            padding: 24,
          }}
        >
          <Dialog.Title style={{ fontSize: 16, marginBottom: 4 }}>Start Pipeline</Dialog.Title>
          <p className="text-muted" style={{ fontSize: 12.5, marginTop: 0, marginBottom: 18 }}>
            Step {step === 'source' ? '1' : '2'} of 2 — {step === 'source' ? 'Data source' : 'Review & launch'}
          </p>

          {step === 'source' && (
            <div className="stack" style={{ gap: 12 }}>
              <label className="stack" style={{ gap: 5 }}>
                <span style={{ fontSize: 12.5, fontWeight: 500 }}>Dataset URI (required)</span>
                <input
                  className="mono"
                  value={datasetUri}
                  onChange={(e) => setDatasetUri(e.target.value)}
                  placeholder="s3://mlflow-artifacts/datasets/latest.csv"
                  style={inputStyle}
                />
              </label>
              <label className="stack" style={{ gap: 5 }}>
                <span style={{ fontSize: 12.5, fontWeight: 500 }}>Model URI (optional — skip to train fresh)</span>
                <input
                  className="mono"
                  value={modelUri}
                  onChange={(e) => setModelUri(e.target.value)}
                  placeholder="runs:/abc123/model"
                  style={inputStyle}
                />
              </label>
              <div className="row" style={{ justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                <Dialog.Close asChild>
                  <Button variant="ghost">Cancel</Button>
                </Dialog.Close>
                <Button disabled={!datasetUri} onClick={() => setStep('review')}>
                  Next
                </Button>
              </div>
            </div>
          )}

          {step === 'review' && (
            <div className="stack" style={{ gap: 10 }}>
              <ReviewRow label="Dataset" value={datasetUri} />
              <ReviewRow label="Model" value={modelUri || '(train fresh)'} />
              {mutation.isError && (
                <p style={{ color: 'var(--crit)', fontSize: 12.5 }}>{(mutation.error as Error).message}</p>
              )}
              <div className="row" style={{ justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                <Button variant="ghost" onClick={() => setStep('source')}>
                  Back
                </Button>
                <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
                  {mutation.isPending ? 'Launching…' : 'Launch workflow'}
                </Button>
              </div>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ReviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="row" style={{ justifyContent: 'space-between', fontSize: 13 }}>
      <span className="text-muted">{label}</span>
      <span className="mono">{value}</span>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: '8px 10px',
  borderRadius: 6,
  border: '1px solid var(--border-strong)',
  background: 'var(--bg)',
  color: 'var(--text)',
  fontSize: 13,
};
