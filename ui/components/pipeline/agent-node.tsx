'use client';

import { Handle, Position, type NodeProps } from 'reactflow';
import {
  Activity, Bot, Database, GitBranch, Rocket, ShieldAlert, ShieldCheck, Sparkles, Workflow,
  type LucideIcon,
} from 'lucide-react';

const ICON_BY_AGENT: Record<string, LucideIcon> = {
  orchestrator: Workflow,
  planner: GitBranch,
  data_agent: Database,
  training_agent: Sparkles,
  evaluation_agent: ShieldCheck,
  deployment_agent: Rocket,
  monitoring_agent: Activity,
  governance_agent: ShieldCheck,
  security_agent: ShieldAlert,
};

export interface AgentNodeData {
  label: string;
  agentId: string;
  capabilities?: string[];
}

/**
 * Custom React Flow node styled after real pipeline-graph tools (Airflow's
 * Graph View, Kubeflow Pipelines' run graph) — an icon tied to the agent's
 * real role, a monospace id line, and small typed connection handles,
 * rather than a plain bordered <div>.
 */
export function AgentNode({ data, selected }: NodeProps<AgentNodeData>) {
  const Icon = ICON_BY_AGENT[data.agentId] ?? Bot;
  return (
    <div className={`pipeline-node${selected ? ' pipeline-node-selected' : ''}`}>
      <Handle type="target" position={Position.Left} className="pipeline-handle" />
      <div className="pipeline-node-icon">
        <Icon size={15} />
      </div>
      <div className="pipeline-node-body">
        <div className="pipeline-node-label">{data.label}</div>
        <div className="pipeline-node-id mono">{data.agentId}</div>
      </div>
      <Handle type="source" position={Position.Right} className="pipeline-handle" />
    </div>
  );
}

export const PIPELINE_NODE_TYPES = { agentNode: AgentNode };
