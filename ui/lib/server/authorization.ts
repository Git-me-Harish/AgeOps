import type { Role } from './rbac';

interface Rule {
  method: string;
  test: RegExp;
  role: Role;
  reason: string;
}

/**
 * Route -> minimum-role map for the BFF gateway (real matrix confirmed
 * with the project owner): viewer (read-only) < operator (start
 * workflows, trigger retraining, decide RL recs) < approver (approve
 * workflows, promote models, rollback) < admin (edit OPA policy, manage
 * roles). Checked in order — first match wins. Any GET request only ever
 * needs an authenticated session (checked separately, not via this list).
 */
const RULES: Rule[] = [
  { method: 'PUT', test: /\/api\/opa\/policy$/, role: 'admin', reason: 'editing the OPA policy requires admin' },
  {
    method: 'POST',
    test: /\/api\/workflows\/[^/]+\/approve$/,
    role: 'approver',
    reason: 'approving/rejecting a workflow requires approver',
  },
  {
    method: 'POST',
    test: /\/registry\/v1\/promote_model$/,
    role: 'approver',
    reason: 'promoting a model requires approver',
  },
  {
    method: 'POST',
    test: /\/api\/deployments\/[^/]+\/rollback$/,
    role: 'approver',
    reason: 'rolling back a deployment requires approver',
  },
];

const DEFAULT_MUTATION_ROLE: Role = 'operator';

export function requiredRoleFor(method: string, pathname: string): { role: Role; reason: string } {
  if (method === 'GET' || method === 'HEAD') return { role: 'viewer', reason: 'read access' };
  const rule = RULES.find((r) => r.method === method && r.test.test(pathname));
  if (rule) return { role: rule.role, reason: rule.reason };
  return { role: DEFAULT_MUTATION_ROLE, reason: 'mutating action requires operator' };
}
