'use client';

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from './button';

/**
 * Real Previous/Next pagination control. Two flavors:
 *  - server: caller passes the true total_count from a real COUNT(*)/DataFrame
 *    length (Command Center workflows, Experiment Lab runs).
 *  - client: caller passes the length of an already-fetched batch and this
 *    just windows into it (used for the smaller lists — drift reports,
 *    agent activity, audit log — where a full backend rewrite for
 *    offset pagination wasn't worth it for lists that don't realistically
 *    grow past a couple hundred rows).
 * Either way, Next disables at the real last page — never a fixed count of
 * pages guessed from a single fetched chunk.
 */
export function Pagination({
  page, pageSize, totalCount, onPageChange,
}: {
  page: number;
  pageSize: number;
  totalCount: number;
  onPageChange: (page: number) => void;
}) {
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
  const from = totalCount === 0 ? 0 : page * pageSize + 1;
  const to = Math.min(totalCount, (page + 1) * pageSize);

  if (totalCount <= pageSize) return null;

  return (
    <div className="row" style={{ justifyContent: 'space-between', paddingTop: 10, marginTop: 4, borderTop: '1px solid var(--border)' }}>
      <span className="text-faint" style={{ fontSize: 11.5 }}>
        {from}–{to} of {totalCount}
      </span>
      <div className="row" style={{ gap: 6 }}>
        <Button size="sm" variant="secondary" disabled={page <= 0} onClick={() => onPageChange(page - 1)}>
          <ChevronLeft size={13} /> Previous
        </Button>
        <span className="text-faint mono" style={{ fontSize: 11.5, alignSelf: 'center', padding: '0 4px' }}>
          {page + 1} / {totalPages}
        </span>
        <Button size="sm" variant="secondary" disabled={page + 1 >= totalPages} onClick={() => onPageChange(page + 1)}>
          Next <ChevronRight size={13} />
        </Button>
      </div>
    </div>
  );
}
