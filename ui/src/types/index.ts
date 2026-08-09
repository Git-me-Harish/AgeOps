// ─── Workflow ──────────────────────────────────────────────────────────────────
export type WorkflowStage =
  | 'idle' | 'data' | 'training' | 'evaluation'
  | 'deployment' | 'monitoring' | 'done' | 'error'

export type WorkflowStatus = 'running' | 'completed' | 'failed' | 'pending'

export interface Workflow {
  workflow_id: string
  status: WorkflowStatus
  current_stage: WorkflowStage
  metrics: Record<string, number>
  errors: string[]
  awaiting_approval: boolean
}

// ─── Agent ────────────────────────────────────────────────────────────────────
export interface AgentCard {
  agent_id: string
  name: string
  version: string
  description: string
  capabilities: string[]
  mcp_tools: string[]
  health_endpoint: string | null
  max_concurrent_tasks: number
  timeout_seconds: number
  requires_human_approval_for: string[]
}

export type AgentStatus = 'online' | 'offline' | 'busy' | 'error'

// ─── Model Registry ───────────────────────────────────────────────────────────
export type ModelStage = 'Staging' | 'Production' | 'Archived' | 'None'

export interface RegisteredModel {
  name: string
  version: string
  stage: ModelStage
  run_id: string
  description: string
  tags: Record<string, string>
}

// ─── Metrics ─────────────────────────────────────────────────────────────────
export interface MetricStat {
  min: number
  max: number
  mean: number
}

export interface MetricsSummary {
  accuracy?: MetricStat
  f1_score?: MetricStat
  drift_score?: MetricStat
  validation_score?: MetricStat
}

// ─── Security ─────────────────────────────────────────────────────────────────
export interface SecurityReport {
  critical_cves: number
  high_cves: number
  policy_violations: number
  last_scan: string
  overall_score: number
}
