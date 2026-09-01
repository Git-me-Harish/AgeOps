/**
 * Client-side data layer. Every function here calls the real /api/gateway
 * proxy (never a mock array) and parses the response through the Zod
 * schemas in lib/schemas.ts — a shape mismatch throws immediately instead
 * of silently rendering `undefined` the way V1's untyped fetches could.
 */
import { z } from 'zod';
import {
  AgentActivitySchema,
  AgentCardSchema,
  AgentPodStatusSchema,
  AuditDecisionSchema,
  BuildJobSchema,
  CompareModelsResponseSchema,
  DriftReportSchema,
  ExperimentRunSchema,
  SpanSchema,
  LineageGraphSchema,
  MetricsSummarySchema,
  MonitoringEventSchema,
  OciImageSchema,
  OpaPolicySchema,
  OpaValidationResultSchema,
  PromoteModelResponseSchema,
  RecommendationSchema,
  RegisteredModelSchema,
  SbomRecordSchema,
  TriggerSchema,
  TrivyScanSchema,
  Workflow,
  WorkflowListItemSchema,
  WorkflowSchema,
} from './schemas';

class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail);
  }
}

async function request<T>(path: string, schema: z.ZodType<T>, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'content-type': 'application/json', ...init?.headers } });
  const text = await res.text();
  const body = text ? JSON.parse(text) : {};
  if (!res.ok) {
    throw new ApiError(res.status, typeof body?.detail === 'string' ? body.detail : JSON.stringify(body));
  }
  return schema.parse(body);
}

// ── orchestrator ───────────────────────────────────────────────────────────
export const startWorkflow = (datasetUri: string, modelUri?: string) =>
  request('/api/gateway/orchestrator/api/workflows', WorkflowSchema, {
    method: 'POST',
    body: JSON.stringify({ dataset_uri: datasetUri, model_uri: modelUri }),
  });

export const listWorkflows = (limit = 20, offset = 0) =>
  request(
    `/api/gateway/orchestrator/api/workflows?limit=${limit}&offset=${offset}`,
    z.object({
      workflows: z.array(WorkflowListItemSchema),
      total_count: z.number(),
      running_count: z.number(),
      awaiting_approval_count: z.number(),
    }),
  );

export const getWorkflow = (id: string) =>
  request(`/api/gateway/orchestrator/api/workflows/${id}`, WorkflowSchema);

export const approveWorkflow = (id: string, approved: boolean, reviewer: string, reason: string) =>
  request(`/api/gateway/orchestrator/api/workflows/${id}/approve`, WorkflowSchema, {
    method: 'POST',
    body: JSON.stringify({ approved, reviewer, reason }),
  });

export const listAgents = () =>
  request('/api/gateway/orchestrator/api/agents', z.object({ agents: z.array(AgentCardSchema) }));

export const getAgentActivity = (agentRole: string, limit = 100) =>
  request(`/api/gateway/orchestrator/api/agents/${agentRole}/activity?limit=${limit}`, AgentActivitySchema);

export const getAuditLog = (limit = 200) =>
  request(`/api/gateway/orchestrator/api/governance/audit-log?limit=${limit}`, z.object({ decisions: z.array(AuditDecisionSchema) }));

export const listAgentStatus = () =>
  request('/api/gateway/orchestrator/api/agents/status', z.object({ agents: z.array(AgentPodStatusSchema) }));

export const listModels = (stage?: string) =>
  request(
    `/api/gateway/orchestrator/api/models${stage ? `?stage=${stage}` : ''}`,
    z.object({ models: z.array(RegisteredModelSchema) }),
  );

export const listExperimentRuns = (limit = 20, offset = 0) =>
  request(
    `/api/gateway/orchestrator/api/experiments/runs?limit=${limit}&offset=${offset}`,
    z.object({ runs: z.array(ExperimentRunSchema), total_count: z.number() }),
  );

export const getRunTrace = (runId: string) =>
  request(`/api/gateway/orchestrator/api/experiments/runs/${runId}/trace`, z.object({ spans: z.array(SpanSchema) }));

export const getMetricsSummary = () =>
  request('/api/gateway/orchestrator/api/metrics/summary', z.object({ summary: MetricsSummarySchema, total_runs: z.number().optional() }));

// ── registry ───────────────────────────────────────────────────────────────
export const getLineage = (modelName: string, modelVersion: string) =>
  request(`/api/gateway/registry/registry/v1/lineage/${modelName}/${modelVersion}`, LineageGraphSchema);

export const getPromotionHistory = (modelName: string) =>
  request(`/api/gateway/registry/registry/v1/promotion_history/${modelName}`, z.record(z.unknown()));

export const promoteModel = (
  modelName: string, modelVersion: string, targetStage: string, triggeredBy: string,
) =>
  request('/api/gateway/registry/registry/v1/promote_model', PromoteModelResponseSchema, {
    method: 'POST',
    body: JSON.stringify({
      model_name: modelName, model_version: modelVersion,
      target_stage: targetStage, triggered_by: triggeredBy, trigger_type: 'human',
    }),
  });

export const compareModels = (modelName: string, versions: string[], initiatedBy: string) =>
  request('/api/gateway/registry/registry/v1/compare_models', CompareModelsResponseSchema, {
    method: 'POST',
    body: JSON.stringify({ model_name: modelName, versions, initiated_by: initiatedBy }),
  });

// ── packaging ──────────────────────────────────────────────────────────────
export const getOciImage = (modelName: string, modelVersion: string) =>
  request(`/api/gateway/packaging/packaging/v1/image/${modelName}/${modelVersion}`, OciImageSchema);

export const getTrivyScan = (imageDigest: string) =>
  request(`/api/gateway/packaging/packaging/v1/scan/${encodeURIComponent(imageDigest)}`, TrivyScanSchema);

export const getSbom = (modelName: string, modelVersion: string) =>
  request(`/api/gateway/packaging/packaging/v1/sbom/${modelName}/${modelVersion}`, SbomRecordSchema);

/** 404 ("not packaged yet") is a real, expected state — not an error to surface as one. */
async function orNull<T>(fn: () => Promise<T>): Promise<T | null> {
  try {
    return await fn();
  } catch (exc) {
    if (exc instanceof ApiError && exc.status === 404) return null;
    throw exc;
  }
}
export const getOciImageOrNull = (modelName: string, modelVersion: string) =>
  orNull(() => getOciImage(modelName, modelVersion));
export const getSbomOrNull = (modelName: string, modelVersion: string) =>
  orNull(() => getSbom(modelName, modelVersion));
export const getTrivyScanOrNull = (imageDigest: string) => orNull(() => getTrivyScan(imageDigest));

export const listBuildJobs = (modelName: string) =>
  request(
    `/api/gateway/packaging/packaging/v1/jobs/${modelName}`,
    z.object({ model_name: z.string(), jobs: z.array(BuildJobSchema) }),
  );

// ── monitoring ─────────────────────────────────────────────────────────────
export const listMonitoringEvents = (modelName: string) =>
  request(`/api/gateway/monitoring/monitoring/v1/events/${modelName}`, z.array(MonitoringEventSchema));

export const listDriftReports = (modelName: string) =>
  request(`/api/gateway/monitoring/monitoring/v1/drift_reports/${modelName}`, z.array(DriftReportSchema));

export const listTriggers = () =>
  request('/api/gateway/monitoring/monitoring/v1/triggers', z.array(TriggerSchema));

export const triggerRetraining = (modelName: string, modelVersion: string, reason: string) =>
  request('/api/gateway/monitoring/monitoring/v1/trigger_retraining', TriggerSchema, {
    method: 'POST',
    body: JSON.stringify({ model_name: modelName, model_version: modelVersion, reason }),
  });

export const listRecommendations = () =>
  request('/api/gateway/monitoring/monitoring/v1/recommendations', z.array(RecommendationSchema));

export const decideRecommendation = (id: number, decision: 'accepted' | 'rejected', reviewedBy: string) =>
  request(`/api/gateway/monitoring/monitoring/v1/recommendations/${id}/decide`, RecommendationSchema, {
    method: 'POST',
    body: JSON.stringify({ decision, reviewed_by: reviewedBy }),
  });

// ── live serving ───────────────────────────────────────────────────────────
export const rollbackDeployment = (serviceName: string, reason: string, modelName?: string, modelVersion?: string) =>
  request(`/api/gateway/orchestrator/api/deployments/${serviceName}/rollback`, z.object({
    service_name: z.string(), status: z.string(), reason: z.string(),
  }), {
    method: 'POST',
    body: JSON.stringify({ reason, model_name: modelName ?? '', model_version: modelVersion ?? '' }),
  });

// ── OPA policy editor ────────────────────────────────────────────────────
export const getOpaPolicy = () => request('/api/gateway/orchestrator/api/opa/policy', OpaPolicySchema);
export const putOpaPolicy = (rego: string) =>
  request('/api/gateway/orchestrator/api/opa/policy', OpaValidationResultSchema, {
    method: 'PUT',
    body: JSON.stringify({ rego }),
  });

export type { Workflow };
export { ApiError };
