import { memo, useCallback, useMemo, useState } from 'react'
import { Stack, Group, Title, Text, Card, Button, Badge, Select, ActionIcon, Tooltip, Box } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  ReactFlow, Background, Controls, MiniMap, Handle, Position, MarkerType,
  addEdge, useNodesState, useEdgesState,
  type Node, type Edge, type Connection, type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { IconPlayerPlay, IconDownload, IconPlus } from '@tabler/icons-react'
import { useStore } from '../store'

type AgentNodeData = {
  name: string
  role: string
  system: string
  status: 'Ready' | 'Gate' | 'Observe'
}

const STATUS_COLOR: Record<AgentNodeData['status'], string> = {
  Ready: 'gray',
  Gate: 'indigo',
  Observe: 'teal',
}

const AgentNode = memo(({ data, selected }: NodeProps<AgentNodeData>) => (
  <Box
    style={{
      width: 220,
      border: selected ? '1px solid #2563eb' : '1px solid #d8dee8',
      borderRadius: 8,
      background: '#ffffff',
      boxShadow: selected
        ? '0 10px 28px rgba(37, 99, 235, 0.14)'
        : '0 8px 22px rgba(15, 23, 42, 0.08)',
      overflow: 'hidden',
    }}
  >
    <Handle
      type="target"
      position={Position.Left}
      style={{ width: 8, height: 8, background: '#64748b', border: '2px solid #fff' }}
    />
    <Box p="sm">
      <Group justify="space-between" align="flex-start" gap="xs" wrap="nowrap">
        <Box>
          <Text size="sm" fw={700} c="gray.9" lh={1.2}>
            {data.name}
          </Text>
          <Text size="xs" c="dimmed" mt={3} lh={1.25}>
            {data.role}
          </Text>
        </Box>
        <Badge size="xs" variant="light" color={STATUS_COLOR[data.status]} radius="sm">
          {data.status}
        </Badge>
      </Group>
    </Box>
    <Box
      px="sm"
      py={7}
      style={{
        borderTop: '1px solid #eef1f5',
        background: '#f8fafc',
      }}
    >
      <Text size="11px" tt="uppercase" c="gray.6" fw={700}>
        {data.system}
      </Text>
    </Box>
    <Handle
      type="source"
      position={Position.Right}
      style={{ width: 8, height: 8, background: '#64748b', border: '2px solid #fff' }}
    />
  </Box>
))

const AGENT_CATALOG: Record<string, AgentNodeData> = {
  data: {
    name: 'Data Agent',
    role: 'Ingestion, validation, and drift checks',
    system: 'Feature Pipeline',
    status: 'Ready',
  },
  training: {
    name: 'Training Agent',
    role: 'Model training jobs and run tracking',
    system: 'MLflow / Kubernetes',
    status: 'Ready',
  },
  evaluation: {
    name: 'Evaluation Agent',
    role: 'Metric gates, bias review, and scorecards',
    system: 'Quality Gate',
    status: 'Gate',
  },
  deployment: {
    name: 'Deployment Agent',
    role: 'Canary rollout and service promotion',
    system: 'KServe',
    status: 'Gate',
  },
  monitoring: {
    name: 'Monitoring Agent',
    role: 'Live accuracy, drift, and SLO telemetry',
    system: 'Prometheus',
    status: 'Observe',
  },
  security: {
    name: 'Security Agent',
    role: 'Policy checks, PII scanning, and CVE review',
    system: 'Guardrails',
    status: 'Gate',
  },
  governance: {
    name: 'Governance Agent',
    role: 'Audit trail, approvals, and compliance rules',
    system: 'OPA Policy',
    status: 'Gate',
  },
  rl: {
    name: 'RL Optimizer',
    role: 'Trace analysis and hyperparameter suggestions',
    system: 'Optimizer',
    status: 'Observe',
  },
}

const EDGE_STYLE = {
  stroke: '#64748b',
  strokeWidth: 1.8,
}

const GATE_EDGE_STYLE = {
  stroke: '#94a3b8',
  strokeWidth: 1.6,
  strokeDasharray: '6 6',
}

// Default pipeline graph
const INITIAL_NODES: Node<AgentNodeData>[] = [
  { id: '1', type: 'agent', position: { x: 40, y: 150 }, data: AGENT_CATALOG.data },
  { id: '2', type: 'agent', position: { x: 320, y: 150 }, data: AGENT_CATALOG.training },
  { id: '3', type: 'agent', position: { x: 600, y: 150 }, data: AGENT_CATALOG.evaluation },
  { id: '4', type: 'agent', position: { x: 880, y: 150 }, data: AGENT_CATALOG.deployment },
  { id: '5', type: 'agent', position: { x: 1160, y: 150 }, data: AGENT_CATALOG.monitoring },
  { id: '6', type: 'agent', position: { x: 600, y: 330 }, data: AGENT_CATALOG.security },
  { id: '7', type: 'agent', position: { x: 880, y: 330 }, data: AGENT_CATALOG.governance },
]

const INITIAL_EDGES: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: EDGE_STYLE },
  { id: 'e2-3', source: '2', target: '3', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: EDGE_STYLE },
  { id: 'e3-4', source: '3', target: '4', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: EDGE_STYLE },
  { id: 'e4-5', source: '4', target: '5', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: EDGE_STYLE },
  { id: 'e3-6', source: '3', target: '6', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: GATE_EDGE_STYLE },
  { id: 'e4-7', source: '4', target: '7', type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, style: GATE_EDGE_STYLE },
]

const AGENT_PALETTE = [
  { value: 'data', label: 'Data Agent' },
  { value: 'training', label: 'Training Agent' },
  { value: 'evaluation', label: 'Evaluation Agent' },
  { value: 'deployment', label: 'Deployment Agent' },
  { value: 'monitoring', label: 'Monitoring Agent' },
  { value: 'security', label: 'Security Agent' },
  { value: 'governance', label: 'Governance Agent' },
  { value: 'rl', label: 'RL Optimizer' },
]

export default function WorkflowDesignerPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState(INITIAL_NODES)
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES)
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const nodeTypes = useMemo(() => ({ agent: AgentNode }), [])
  const startWorkflow = useStore(s => s.startWorkflow)
  const loading = useStore(s => s.loadingWorkflow)

  const onConnect = useCallback(
    (params: Connection) => setEdges(eds => addEdge({
      ...params,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
      style: EDGE_STYLE,
    }, eds)),
    [setEdges]
  )

  const addNode = () => {
    if (!selectedAgent) return
    const newNode: Node<AgentNodeData> = {
      id: `custom-${Date.now()}`,
      type: 'agent',
      position: { x: Math.random() * 420 + 160, y: Math.random() * 220 + 110 },
      data: AGENT_CATALOG[selectedAgent],
    }
    setNodes(nds => [...nds, newNode])
  }

  const exportWorkflow = () => {
    const wf = { nodes: nodes.map(n => ({ id: n.id, agent: n.data.name, system: n.data.system })), edges }
    const blob = new Blob([JSON.stringify(wf, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'workflow.json'; a.click()
    notifications.show({ title: 'Exported', message: 'workflow.json downloaded', color: 'blue' })
  }

  const runWorkflow = async () => {
    try {
      const id = await startWorkflow('s3://mlflow-artifacts/datasets/latest.csv')
      notifications.show({ title: 'Pipeline started', message: `Workflow ${id.slice(0, 8)}`, color: 'green' })
    } catch (err) {
      notifications.show({ title: 'Error', message: String(err), color: 'red' })
    }
  }

  return (
    <Stack gap="md" h="100%">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2} fw={700}>Workflow Designer</Title>
          <Text c="dimmed" size="sm">Author and review the production ML pipeline topology</Text>
        </div>
        <Group gap="xs">
          <Select
            placeholder="Add agent"
            data={AGENT_PALETTE}
            value={selectedAgent}
            onChange={setSelectedAgent}
            w={220}
            size="sm"
          />
          <Tooltip label="Add node">
            <ActionIcon size="lg" variant="default" onClick={addNode} disabled={!selectedAgent}>
              <IconPlus size={16} />
            </ActionIcon>
          </Tooltip>
          <Button leftSection={<IconDownload size={14} />} variant="default" size="sm" onClick={exportWorkflow}>
            Export JSON
          </Button>
          <Button leftSection={<IconPlayerPlay size={14} />} size="sm" onClick={runWorkflow} loading={loading}>
            Run Workflow
          </Button>
        </Group>
      </Group>

      <Card p={0} style={{ height: 620, overflow: 'hidden', borderColor: '#d8dee8' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          deleteKeyCode="Delete"
          defaultViewport={{ x: 0, y: 0, zoom: 0.9 }}
        >
          <Background color="#e6ebf2" gap={24} size={1} />
          <Controls showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            nodeColor={() => '#cbd5e1'}
            maskColor="rgba(248, 250, 252, 0.72)"
            style={{
              width: 180,
              height: 110,
              borderRadius: 6,
              border: '1px solid #e2e8f0',
              boxShadow: '0 6px 18px rgba(15, 23, 42, 0.08)',
            }}
          />
        </ReactFlow>
      </Card>
    </Stack>
  )
}
