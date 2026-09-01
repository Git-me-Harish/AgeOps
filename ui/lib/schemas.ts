/**
 * Zod schemas mirroring the real backend Pydantic response models
 * (mcp_servers/api_server.py, registry_server.py, packaging_server.py,
 * monitoring_server.py). Every field here corresponds to a field that
 * genuinely exists in a backend response — this file is the contract that
 * keeps the UI from silently drifting back into V1's mock-shaped types.
 */
import { z } from 'zod';

// ── orchestrator (api_server.py) ──────────────────────────────────────────
export const WorkflowSchema = z.object({
  workflow_id: z.string(),
  status: z.string(),
  current_stage: z.string().nullable().optional(),
  metrics: z.record(z.unknown()).default({}),
  errors: z.array(z.string()).default([]),
  awaiting_approval: z.boolean().default(false),
});
export type Workflow = z.infer<typeof WorkflowSchema>;

export const WorkflowListItemSchema = z.object({
  workflow_id: z.string(),
  status: z.string(),
  current_stage: z.string().nullable(),
  dataset_uri: z.string().nullable(),
  model_uri: z.string().nullable(),
  metrics: z.record(z.unknown()).default({}),
  errors: z.array(z.string()).default([]),
  awaiting_approval: z.boolean(),
  source: z.string().nullable(),
  trigger_type: z.string().nullable(),
  created_at: z.string().nullable(),
  updated_at: z.string().nullable(),
});
export type WorkflowListItem = z.infer<typeof WorkflowListItemSchema>;

export const AgentCardSchema = z.object({
  agent_id: z.string(),
  name: z.string(),
  version: z.string(),
  description: z.string(),
  capabilities: z.array(z.string()).default([]),
  mcp_tools: z.array(z.string()).default([]),
  health_endpoint: z.string().nullable().optional(),
  max_concurrent_tasks: z.number().optional(),
  timeout_seconds: z.number().optional(),
  requires_human_approval_for: z.array(z.string()).default([]),
});
export type AgentCard = z.infer<typeof AgentCardSchema>;

export const AgentPodStatusSchema = z.object({
  agent_id: z.string(),
  status: z.enum(['online', 'degraded', 'unknown']),
  reason: z.string().optional(),
  pod_count: z.number().optional(),
  phases: z.array(z.string()).optional(),
  restarts: z.number().optional(),
});
export type AgentPodStatus = z.infer<typeof AgentPodStatusSchema>;

export const RegisteredModelSchema = z.object({
  name: z.string(),
  version: z.string(),
  stage: z.string().nullable(),
  run_id: z.string(),
  description: z.string().nullable().optional(),
  tags: z.record(z.string()).default({}),
});
export type RegisteredModel = z.infer<typeof RegisteredModelSchema>;

const MetricStatSchema = z.object({ min: z.number(), max: z.number(), mean: z.number() }).nullable();
export const MetricsSummarySchema = z.object({
  accuracy: MetricStatSchema.optional(),
  f1_score: MetricStatSchema.optional(),
  drift_score: MetricStatSchema.optional(),
  validation_score: MetricStatSchema.optional(),
});
export type MetricsSummary = z.infer<typeof MetricsSummarySchema>;

// ── registry_server.py ────────────────────────────────────────────────────
export const PromoteModelResponseSchema = z.object({
  model_name: z.string(),
  model_version: z.string(),
  from_stage: z.string(),
  to_stage: z.string(),
  gates_passed: z.array(z.string()),
  gates_failed: z.array(z.string()),
  success: z.boolean(),
  promotion_db_id: z.number().nullable().optional(),
  rejection_reason: z.string().nullable().optional(),
});
export type PromoteModelResponse = z.infer<typeof PromoteModelResponseSchema>;

export const LineageGraphSchema = z.object({
  model_name: z.string(),
  model_version: z.string(),
  nodes: z.array(z.record(z.unknown())),
  edges: z.array(z.record(z.unknown())),
});
export type LineageGraph = z.infer<typeof LineageGraphSchema>;

export const CompareModelsResponseSchema = z.object({
  comparison_id: z.union([z.string(), z.number()]),
  model_name: z.string(),
  recommended_version: z.string().nullable(),
  versions: z.array(z.record(z.unknown())),
  metric_diffs: z.array(z.record(z.unknown())),
});
export type CompareModelsResponse = z.infer<typeof CompareModelsResponseSchema>;

// ── packaging_server.py ───────────────────────────────────────────────────
export const OciImageSchema = z.object({
  id: z.number(),
  model_name: z.string(),
  model_version: z.string(),
  image_tag: z.string(),
  image_digest: z.string(),
  image_uri: z.string(),
  registry_host: z.string(),
  base_image: z.string().nullable().optional(),
  labels: z.record(z.unknown()).default({}),
  pushed_at: z.string().nullable().optional(),
});
export type OciImage = z.infer<typeof OciImageSchema>;

export const TrivyScanSchema = z.object({
  image_digest: z.string(),
  image_tag: z.string(),
  model_name: z.string(),
  model_version: z.string(),
  critical_count: z.number(),
  high_count: z.number(),
  medium_count: z.number(),
  low_count: z.number(),
  total_count: z.number(),
  passed: z.boolean(),
  scan_duration_ms: z.number().nullable().optional(),
  scanned_at: z.string().nullable().optional(),
});
export type TrivyScan = z.infer<typeof TrivyScanSchema>;

export const SbomRecordSchema = z.object({
  image_digest: z.string(),
  model_name: z.string(),
  model_version: z.string(),
  format: z.string(),
  r2_uri: z.string(),
  package_count: z.number(),
  syft_version: z.string().nullable().optional(),
  generated_at: z.string().nullable().optional(),
});
export type SbomRecord = z.infer<typeof SbomRecordSchema>;

export const BuildJobSchema = z.object({
  id: z.number(),
  model_version: z.string(),
  image_tag: z.string(),
  status: z.string(),
  duration_seconds: z.number().nullable().optional(),
  error_message: z.string().nullable().optional(),
  created_at: z.string().nullable().optional(),
});
export type BuildJob = z.infer<typeof BuildJobSchema>;

// ── monitoring_server.py ──────────────────────────────────────────────────
export const MonitoringEventSchema = z.object({
  id: z.number(),
  model_name: z.string(),
  model_version: z.string().nullable().optional(),
  drift_score: z.number().nullable(),
  accuracy: z.number().nullable(),
  baseline_accuracy: z.number().nullable(),
  p99_latency_ms: z.number().nullable(),
  error_rate: z.number().nullable(),
  alert_level: z.string(),
  alerts: z.array(z.string()).default([]),
  created_at: z.string(),
});
export type MonitoringEvent = z.infer<typeof MonitoringEventSchema>;

export const DriftReportSchema = z.object({
  id: z.number(),
  model_name: z.string(),
  model_version: z.string().nullable().optional(),
  sample_count: z.number().nullable(),
  drift_score: z.number().nullable(),
  alert_level: z.string(),
  check_error: z.string().nullable().optional(),
  r2_report_path: z.string().nullable().optional(),
  created_at: z.string(),
});
export type DriftReport = z.infer<typeof DriftReportSchema>;

export const TriggerSchema = z.object({
  id: z.number(),
  trigger_type: z.string(),
  source: z.string(),
  model_name: z.string(),
  model_version: z.string().nullable().optional(),
  status: z.string(),
  reason: z.string().nullable().optional(),
  launched_workflow_id: z.string().nullable().optional(),
  created_at: z.string(),
});
export type Trigger = z.infer<typeof TriggerSchema>;

export const RecommendationSchema = z.object({
  id: z.number(),
  workflow_id: z.string().nullable().optional(),
  agent_role: z.string(),
  recommendation: z.record(z.unknown()),
  confidence: z.number().nullable().optional(),
  status: z.string(),
  reviewed_by: z.string().nullable().optional(),
  created_at: z.string(),
});
export type Recommendation = z.infer<typeof RecommendationSchema>;

// ── real-time events (agents/events.py channels) ──────────────────────────
export const WsEventSchema = z.object({
  channel: z.string(),
}).passthrough();
export type WsEvent = z.infer<typeof WsEventSchema>;

export const ExperimentRunSchema = z.object({
  run_id: z.string(),
  run_name: z.string().nullable(),
  status: z.string().nullable(),
  start_time: z.string().nullable(),
  end_time: z.string().nullable(),
  metrics: z.record(z.number()).default({}),
  params: z.record(z.string()).default({}),
});
export type ExperimentRun = z.infer<typeof ExperimentRunSchema>;

export const SpanSchema = z.object({
  span_id: z.string(),
  name: z.string(),
  start_time_ns: z.number().nullable(),
  end_time_ns: z.number().nullable(),
  parent_id: z.string(),
  status: z.string(),
});
export type Span = z.infer<typeof SpanSchema>;

export const AgentActivitySchema = z.object({
  agent_role: z.string(),
  recent_calls: z.array(z.object({
    workflow_id: z.string().nullable(),
    model: z.string(),
    was_fallback: z.boolean(),
    was_cache_hit: z.boolean(),
    total_tokens: z.number().nullable(),
    cost_usd: z.number().nullable(),
    latency_ms: z.number().nullable(),
    created_at: z.string().nullable(),
  })),
  summary: z.object({
    call_count: z.number(),
    avg_latency_ms: z.number().nullable(),
    total_tokens: z.number().nullable(),
    total_cost_usd: z.number().nullable(),
    cache_hit_rate: z.number().nullable(),
    success_rate: z.number().nullable(),
  }).nullable(),
});
export type AgentActivity = z.infer<typeof AgentActivitySchema>;

export const AuditDecisionSchema = z.object({
  workflow_id: z.string().nullable(),
  policy_name: z.string(),
  policy_package: z.string().nullable(),
  rego_policy_version: z.string().nullable(),
  decision: z.boolean(),
  deny_reasons: z.array(z.string()).default([]),
  agent_role: z.string().nullable(),
  was_sidecar: z.boolean(),
  evaluation_time_ms: z.number().nullable(),
  created_at: z.string().nullable(),
});
export type AuditDecision = z.infer<typeof AuditDecisionSchema>;

export const OpaPolicySchema = z.object({ rego: z.string() });
export const OpaValidationResultSchema = z.object({
  valid: z.boolean(),
  checked_with: z.string(),
  errors: z.array(z.string()).default([]),
});
export type OpaValidationResult = z.infer<typeof OpaValidationResultSchema>;
