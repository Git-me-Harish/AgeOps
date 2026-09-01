import { useMemo, useState } from 'react';

/**
 * Client-side pagination over an already-fetched array — used for the
 * smaller lists (drift reports, agent activity, audit log, policy
 * denials) where a full backend offset-pagination rewrite wasn't worth it
 * for tables that don't realistically grow past a couple hundred rows.
 * Real data throughout — this only windows into what was actually
 * fetched, it never fabricates rows.
 */
export function usePagedSlice<T>(items: T[], pageSize: number) {
  const [page, setPage] = useState(0);
  const totalCount = items.length;
  const pageItems = useMemo(
    () => items.slice(page * pageSize, page * pageSize + pageSize),
    [items, page, pageSize],
  );
  return { page, setPage, pageItems, totalCount };
}
