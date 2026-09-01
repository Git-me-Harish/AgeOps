'use client';

import * as Dialog from '@radix-ui/react-dialog';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { rollbackDeployment } from '@/lib/api';

export function RollbackDialog({ serviceName, modelName, modelVersion }: {
  serviceName: string; modelName: string; modelVersion: string;
}) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => rollbackDeployment(serviceName, reason, modelName, modelVersion),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['prom'] });
      setOpen(false);
      setReason('');
    },
  });

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <Button variant="danger" size="sm">Rollback</Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)' }} />
        <Dialog.Content
          style={{
            position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
            width: 420, background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-md)', padding: 22,
          }}
        >
          <Dialog.Title style={{ fontSize: 15 }}>Roll back {serviceName}</Dialog.Title>
          <p className="text-muted" style={{ fontSize: 12.5, marginTop: 6 }}>
            Immediately sets canary traffic to 0% via the real InferenceService patch. A reason is
            required for the audit trail.
          </p>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="e.g. elevated 5xx rate reported by on-call"
            rows={3}
            style={{
              width: '100%', marginTop: 10, padding: 8, borderRadius: 6,
              border: '1px solid var(--border-strong)', background: 'var(--bg)', color: 'var(--text)', fontSize: 13,
            }}
          />
          {mutation.isError && (
            <p style={{ color: 'var(--crit)', fontSize: 12.5, marginTop: 6 }}>{(mutation.error as Error).message}</p>
          )}
          <div className="row" style={{ justifyContent: 'flex-end', gap: 8, marginTop: 14 }}>
            <Dialog.Close asChild>
              <Button variant="ghost">Cancel</Button>
            </Dialog.Close>
            <Button variant="danger" disabled={!reason.trim() || mutation.isPending} onClick={() => mutation.mutate()}>
              {mutation.isPending ? 'Rolling back…' : 'Confirm rollback'}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
