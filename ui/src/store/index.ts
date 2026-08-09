import { create } from 'zustand'
import type { AgentCard, Workflow, RegisteredModel, MetricsSummary } from '../types'

const API_BASE = '/api'

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`API ${path} failed: ${res.statusText}`)
  return res.json() as Promise<T>
}

// ─── Store ────────────────────────────────────────────────────────────────────
interface AppState {
  // Workflows
  workflows: Record<string, Workflow>
  activeWorkflowId: string | null
  loadingWorkflow: boolean
  startWorkflow: (datasetUri: string) => Promise<string>
  pollWorkflow: (id: string) => Promise<void>
  approveWorkflow: (id: string, approved: boolean, reviewer: string) => Promise<void>

  // Agents
  agents: AgentCard[]
  fetchAgents: () => Promise<void>

  // Models
  models: RegisteredModel[]
  fetchModels: (stage?: string) => Promise<void>

  // Metrics
  metricsSummary: MetricsSummary | null
  fetchMetrics: () => Promise<void>

  // UI state
  sidebarOpen: boolean
  toggleSidebar: () => void
}

export const useStore = create<AppState>((set, get) => ({
  // ── Workflows ───────────────────────────────────────────────────────────────
  workflows: {},
  activeWorkflowId: null,
  loadingWorkflow: false,

  startWorkflow: async (datasetUri: string) => {
    set({ loadingWorkflow: true })
    const data = await apiFetch<Workflow>('/workflows', {
      method: 'POST',
      body: JSON.stringify({ dataset_uri: datasetUri }),
    })
    set(s => ({
      workflows: { ...s.workflows, [data.workflow_id]: data },
      activeWorkflowId: data.workflow_id,
      loadingWorkflow: false,
    }))
    return data.workflow_id
  },

  pollWorkflow: async (id: string) => {
    const data = await apiFetch<Workflow>(`/workflows/${id}`)
    set(s => ({ workflows: { ...s.workflows, [id]: data } }))
  },

  approveWorkflow: async (id: string, approved: boolean, reviewer: string) => {
    const data = await apiFetch<Workflow>(`/workflows/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ approved, reviewer }),
    })
    set(s => ({ workflows: { ...s.workflows, [id]: data } }))
  },

  // ── Agents ─────────────────────────────────────────────────────────────────
  agents: [],
  fetchAgents: async () => {
    const data = await apiFetch<{ agents: AgentCard[] }>('/agents')
    set({ agents: data.agents })
  },

  // ── Models ─────────────────────────────────────────────────────────────────
  models: [],
  fetchModels: async (stage?: string) => {
    const qs = stage ? `?stage=${stage}` : ''
    const data = await apiFetch<{ models: RegisteredModel[] }>(`/models${qs}`)
    set({ models: data.models })
  },

  // ── Metrics ────────────────────────────────────────────────────────────────
  metricsSummary: null,
  fetchMetrics: async () => {
    const data = await apiFetch<{ summary: MetricsSummary }>('/metrics/summary')
    set({ metricsSummary: data.summary })
  },

  // ── UI ─────────────────────────────────────────────────────────────────────
  sidebarOpen: true,
  toggleSidebar: () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
}))
