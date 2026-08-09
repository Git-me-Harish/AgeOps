import { useEffect, useState } from 'react'
import {
  Stack, Group, Title, Text, Card, SimpleGrid,
  Badge, Tabs, Table, ScrollArea, Switch, Select,
  MultiSelect, NumberInput, Button, Skeleton, Code,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconSettings, IconShieldCheck, IconHistory, IconRobot } from '@tabler/icons-react'
import { useStore } from '../store'
import { AgentStatusGrid } from '../components/agents/AgentStatusGrid'
import type { AgentCard } from '../types'

const ALL_MCP_TOOLS = [
  'mlflow/get_model', 'mlflow/log_metrics', 'mlflow/search_runs',
  'mlflow/register_model', 'mlflow/transition_stage',
  'mlflow/log_agent_decision', 'mlflow/get_experiment_metrics',
  'kubernetes/create_job', 'kubernetes/delete_job',
  'prometheus/query', 'trivy/scan_image',
]

const MOCK_AUDIT_LOG = [
  { ts: '2025-07-15 10:42:01', agent: 'orchestrator',      action: 'created_execution_plan',         wf: 'wf-00001' },
  { ts: '2025-07-15 10:42:05', agent: 'security_agent',    action: 'pre_flight_check_passed',         wf: 'wf-00001' },
  { ts: '2025-07-15 10:42:12', agent: 'data_agent',        action: 'dataset_validated',               wf: 'wf-00001' },
  { ts: '2025-07-15 10:45:37', agent: 'training_agent',    action: 'model_registered',                wf: 'wf-00001' },
  { ts: '2025-07-15 10:46:10', agent: 'evaluation_agent',  action: 'threshold_gate_passed',           wf: 'wf-00001' },
  { ts: '2025-07-15 10:47:00', agent: 'deployment_agent',  action: 'canary_rollout_complete',         wf: 'wf-00001' },
  { ts: '2025-07-15 10:47:05', agent: 'governance_agent',  action: 'audit_trail_signed',              wf: 'wf-00001' },
]

function AgentConfigEditor({ agent }: { agent: AgentCard }) {
  const [tools, setTools] = useState<string[]>(agent.mcp_tools)
  const [timeout, setTimeout_] = useState(agent.timeout_seconds)
  const [maxTasks, setMaxTasks] = useState(agent.max_concurrent_tasks)
  const [hitlEnabled, setHitlEnabled] = useState(agent.requires_human_approval_for.length > 0)

  const save = () => {
    notifications.show({ title: 'Config saved', message: `${agent.name} configuration updated`, color: 'green' })
  }

  return (
    <Stack gap="md">
      <MultiSelect
        label="MCP Tools"
        description="Tools this agent can access at runtime"
        data={ALL_MCP_TOOLS}
        value={tools}
        onChange={setTools}
        searchable
      />
      <Group grow>
        <NumberInput
          label="Timeout (seconds)"
          value={timeout}
          onChange={v => setTimeout_(Number(v))}
          min={10} max={7200} step={10}
        />
        <NumberInput
          label="Max concurrent tasks"
          value={maxTasks}
          onChange={v => setMaxTasks(Number(v))}
          min={1} max={20}
        />
      </Group>
      <Switch
        label="Require human approval for high-risk actions"
        checked={hitlEnabled}
        onChange={e => setHitlEnabled(e.currentTarget.checked)}
        color="orange"
      />
      <Button size="sm" onClick={save}>Save Configuration</Button>
    </Stack>
  )
}

export default function AgentsPage() {
  const { agents, fetchAgents } = useStore()
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)

  useEffect(() => { fetchAgents() }, [fetchAgents])

  const agent = agents.find(a => a.agent_id === selectedAgent)

  return (
    <Stack gap="md">
      <div>
        <Title order={2} fw={700}>Agent Configuration</Title>
        <Text c="dimmed" size="sm">Configure individual agent capabilities, MCP tools, and guardrails</Text>
      </div>

      <Card p="md">
        <Text fw={600} size="sm" mb="md">All Agents</Text>
        {agents.length === 0
          ? <Skeleton height={160} />
          : <AgentStatusGrid agents={agents} />
        }
      </Card>

      <Card p="md">
        <Group justify="space-between" mb="md">
          <Text fw={600} size="sm">Agent Editor</Text>
          <Select
            placeholder="Select agent to configure…"
            data={agents.map(a => ({ value: a.agent_id, label: a.name }))}
            value={selectedAgent}
            onChange={setSelectedAgent}
            w={260} size="sm"
          />
        </Group>

        {agent ? (
          <Tabs defaultValue="config">
            <Tabs.List>
              <Tabs.Tab value="config"  leftSection={<IconSettings size={14} />}>Configuration</Tabs.Tab>
              <Tabs.Tab value="audit"   leftSection={<IconHistory size={14} />}>Audit Log</Tabs.Tab>
              <Tabs.Tab value="card"    leftSection={<IconRobot size={14} />}>Agent Card</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="config" pt="md">
              <AgentConfigEditor agent={agent} />
            </Tabs.Panel>

            <Tabs.Panel value="audit" pt="md">
              <ScrollArea h={280}>
                <Table striped highlightOnHover fz="xs">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Timestamp</Table.Th>
                      <Table.Th>Agent</Table.Th>
                      <Table.Th>Action</Table.Th>
                      <Table.Th>Workflow</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {MOCK_AUDIT_LOG
                      .filter(r => r.agent === agent.agent_id)
                      .map((row, i) => (
                        <Table.Tr key={i}>
                          <Table.Td c="dimmed">{row.ts}</Table.Td>
                          <Table.Td><Badge size="xs" variant="light">{row.agent}</Badge></Table.Td>
                          <Table.Td><Code fz="xs">{row.action}</Code></Table.Td>
                          <Table.Td><Code fz="xs">{row.wf}</Code></Table.Td>
                        </Table.Tr>
                      ))
                    }
                    {MOCK_AUDIT_LOG.filter(r => r.agent === agent.agent_id).length === 0 && (
                      <Table.Tr>
                        <Table.Td colSpan={4}>
                          <Text c="dimmed" ta="center" size="xs" py="md">No audit entries for this agent yet</Text>
                        </Table.Td>
                      </Table.Tr>
                    )}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Tabs.Panel>

            <Tabs.Panel value="card" pt="md">
              <Stack gap="xs">
                {[
                  ['ID', agent.agent_id],
                  ['Version', agent.version],
                  ['Description', agent.description],
                  ['Capabilities', agent.capabilities.join(', ')],
                  ['HITL actions', agent.requires_human_approval_for.join(', ') || 'None'],
                ].map(([k, v]) => (
                  <Group key={k} align="flex-start">
                    <Text size="sm" c="dimmed" w={130}>{k}</Text>
                    <Text size="sm" flex={1}>{v}</Text>
                  </Group>
                ))}
              </Stack>
            </Tabs.Panel>
          </Tabs>
        ) : (
          <Text c="dimmed" size="sm" ta="center" py="xl">Select an agent above to configure it</Text>
        )}
      </Card>

      {/* Full audit log */}
      <Card p="md">
        <Text fw={600} size="sm" mb="md">Recent Agent Audit Trail (all agents)</Text>
        <ScrollArea h={220}>
          <Table striped highlightOnHover fz="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Timestamp</Table.Th>
                <Table.Th>Agent</Table.Th>
                <Table.Th>Action</Table.Th>
                <Table.Th>Workflow</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {MOCK_AUDIT_LOG.map((row, i) => (
                <Table.Tr key={i}>
                  <Table.Td c="dimmed">{row.ts}</Table.Td>
                  <Table.Td><Badge size="xs" variant="light">{row.agent}</Badge></Table.Td>
                  <Table.Td><Code fz="xs">{row.action}</Code></Table.Td>
                  <Table.Td><Code fz="xs">{row.wf}</Code></Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Card>
    </Stack>
  )
}
