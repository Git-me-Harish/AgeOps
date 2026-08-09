import { useEffect, useState } from 'react'
import {
  Stack, Group, Title, Text, Card, Table, Badge,
  TextInput, Select, Tabs, ScrollArea, Skeleton,
  Button, ActionIcon, Collapse, Code,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  IconSearch, IconFilter, IconChevronDown, IconChevronUp,
  IconDownload, IconFlask,
} from '@tabler/icons-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTip, ResponsiveContainer, Legend,
} from 'recharts'

// Synthetic run data — in production fetched from /api/experiments
const MOCK_RUNS = Array.from({ length: 12 }, (_, i) => ({
  id: `run-${String(i + 1).padStart(3, '0')}`,
  name: `workflow-${(i + 1).toString(16).padStart(8, '0')}`,
  stage: i % 3 === 0 ? 'training' : i % 3 === 1 ? 'evaluation' : 'deployment',
  status: i === 2 ? 'failed' : 'finished',
  accuracy: +(0.72 + Math.random() * 0.18).toFixed(4),
  f1_score: +(0.70 + Math.random() * 0.18).toFixed(4),
  drift: +(Math.random() * 0.12).toFixed(4),
  duration: `${Math.round(30 + Math.random() * 120)}s`,
  started: `${i + 1}h ago`,
}))

const CHART_DATA = MOCK_RUNS.map((r, i) => ({
  run: `R${i + 1}`,
  accuracy: r.accuracy,
  f1_score: r.f1_score,
  drift: r.drift,
}))

function RunRow({ run }: { run: typeof MOCK_RUNS[0] }) {
  const [open, { toggle }] = useDisclosure(false)
  return (
    <>
      <Table.Tr style={{ cursor: 'pointer' }} onClick={toggle}>
        <Table.Td><Code fz="xs">{run.id}</Code></Table.Td>
        <Table.Td><Text fz="xs" truncate maw={160}>{run.name}</Text></Table.Td>
        <Table.Td><Badge size="xs" variant="light">{run.stage}</Badge></Table.Td>
        <Table.Td>
          <Badge size="xs" color={run.status === 'failed' ? 'red' : 'green'} variant="dot">
            {run.status}
          </Badge>
        </Table.Td>
        <Table.Td><Text fz="xs">{run.accuracy}</Text></Table.Td>
        <Table.Td><Text fz="xs">{run.f1_score}</Text></Table.Td>
        <Table.Td><Text fz="xs" c={+run.drift > 0.1 ? 'orange' : 'green'}>{run.drift}</Text></Table.Td>
        <Table.Td><Text fz="xs" c="dimmed">{run.duration}</Text></Table.Td>
        <Table.Td>
          <ActionIcon size="xs" variant="subtle">
            {open ? <IconChevronUp size={12} /> : <IconChevronDown size={12} />}
          </ActionIcon>
        </Table.Td>
      </Table.Tr>
      {open && (
        <Table.Tr>
          <Table.Td colSpan={9} p={0}>
            <Collapse in={open}>
              <Stack gap="xs" p="sm" bg="gray.0">
                <Text size="xs" fw={600}>Trace Spans</Text>
                {['data_agent.ingest', 'data_agent.validate', 'training_agent.run', 'evaluation_agent.run'].map(span => (
                  <Group key={span} gap="sm">
                    <Code fz="xs">{span}</Code>
                    <Text size="xs" c="dimmed">{Math.round(50 + Math.random() * 500)}ms</Text>
                    <Badge size="xs" color="green" variant="dot">success</Badge>
                  </Group>
                ))}
              </Stack>
            </Collapse>
          </Table.Td>
        </Table.Tr>
      )}
    </>
  )
}

export default function ExperimentsPage() {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [tab, setTab] = useState<string | null>('runs')

  const filtered = MOCK_RUNS.filter(r =>
    r.name.includes(search) &&
    (!statusFilter || r.status === statusFilter)
  )

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <div>
          <Title order={2} fw={700}>Experiments & Traces</Title>
          <Text c="dimmed" size="sm">MLflow experiment runs with full agent trace visibility</Text>
        </div>
        <Button leftSection={<IconDownload size={14} />} variant="light" size="sm">
          Export CSV
        </Button>
      </Group>

      <Tabs value={tab} onChange={setTab}>
        <Tabs.List>
          <Tabs.Tab value="runs"   leftSection={<IconFlask size={14} />}>Runs ({MOCK_RUNS.length})</Tabs.Tab>
          <Tabs.Tab value="charts" leftSection={<IconFilter size={14} />}>Metric Charts</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="runs" pt="md">
          <Group gap="sm" mb="sm">
            <TextInput
              placeholder="Search runs…"
              leftSection={<IconSearch size={14} />}
              value={search}
              onChange={e => setSearch(e.target.value)}
              size="sm" w={260}
            />
            <Select
              placeholder="Status"
              data={[{ value: 'finished', label: 'Finished' }, { value: 'failed', label: 'Failed' }]}
              value={statusFilter}
              onChange={setStatusFilter}
              clearable size="sm" w={140}
            />
          </Group>

          <Card p={0}>
            <ScrollArea>
              <Table striped highlightOnHover fz="xs">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Run ID</Table.Th>
                    <Table.Th>Workflow</Table.Th>
                    <Table.Th>Stage</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Accuracy</Table.Th>
                    <Table.Th>F1</Table.Th>
                    <Table.Th>Drift</Table.Th>
                    <Table.Th>Duration</Table.Th>
                    <Table.Th></Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {filtered.map(run => <RunRow key={run.id} run={run} />)}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="charts" pt="md">
          <Card p="md">
            <Text fw={600} size="sm" mb="md">Accuracy & F1 over runs</Text>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={CHART_DATA} margin={{ right: 8, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f3f5" />
                <XAxis dataKey="run" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.6, 1]} tick={{ fontSize: 11 }} />
                <RechartsTip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => v.toFixed(4)} />
                <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="accuracy" stroke="#40c057" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="f1_score" stroke="#228be6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="drift"    stroke="#fd7e14" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}
