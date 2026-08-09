import {
  Card, Group, Text, Badge, Stack, Button, Modal,
  Tabs, Table, ScrollArea, ThemeIcon, ActionIcon, Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  IconDatabase, IconHistory, IconShieldCheck,
  IconArrowUp, IconArrowDown, IconExternalLink,
} from '@tabler/icons-react'
import type { RegisteredModel, ModelStage } from '../../types'

const STAGE_COLOR: Record<ModelStage, string> = {
  Production: 'green',
  Staging:    'blue',
  Archived:   'gray',
  None:       'gray',
}

interface Props {
  model: RegisteredModel
}

export function ModelCard({ model }: Props) {
  const [opened, { open, close }] = useDisclosure(false)

  const handlePromote = () => {
    notifications.show({
      title: 'Stage transition queued',
      message: `${model.name} v${model.version} → Production`,
      color: 'green',
    })
    close()
  }

  const handleArchive = () => {
    notifications.show({
      title: 'Model archived',
      message: `${model.name} v${model.version} moved to Archived`,
      color: 'gray',
    })
    close()
  }

  return (
    <>
      <Card p="md" style={{ cursor: 'pointer' }} onClick={open}>
        <Group justify="space-between" mb="xs">
          <Group gap="xs">
            <ThemeIcon size="sm" color="blue" variant="light">
              <IconDatabase size={14} />
            </ThemeIcon>
            <Text fw={600} size="sm">{model.name}</Text>
          </Group>
          <Badge color={STAGE_COLOR[model.stage as ModelStage]} variant="light" size="sm">
            {model.stage}
          </Badge>
        </Group>
        <Text size="xs" c="dimmed" ff="monospace">v{model.version} · {model.run_id?.slice(0, 8)}</Text>
        {model.description && <Text size="xs" c="dimmed" mt={4} lineClamp={2}>{model.description}</Text>}
      </Card>

      <Modal opened={opened} onClose={close} title={`${model.name} v${model.version}`} size="lg" centered>
        <Tabs defaultValue="overview">
          <Tabs.List>
            <Tabs.Tab value="overview" leftSection={<IconDatabase size={14} />}>Overview</Tabs.Tab>
            <Tabs.Tab value="history"  leftSection={<IconHistory size={14} />}>Deploy History</Tabs.Tab>
            <Tabs.Tab value="security" leftSection={<IconShieldCheck size={14} />}>Security</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="overview" pt="md">
            <Stack gap="xs">
              <Group>
                <Text size="sm" w={120} c="dimmed">Run ID</Text>
                <Text size="sm" ff="monospace">{model.run_id}</Text>
              </Group>
              <Group>
                <Text size="sm" w={120} c="dimmed">Stage</Text>
                <Badge color={STAGE_COLOR[model.stage as ModelStage]} variant="light">{model.stage}</Badge>
              </Group>
              {Object.entries(model.tags).map(([k, v]) => (
                <Group key={k}>
                  <Text size="sm" w={120} c="dimmed">{k}</Text>
                  <Text size="sm">{v}</Text>
                </Group>
              ))}
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value="history" pt="md">
            <Table striped highlightOnHover fz="xs">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Date</Table.Th>
                  <Table.Th>Stage</Table.Th>
                  <Table.Th>Strategy</Table.Th>
                  <Table.Th>Result</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                <Table.Tr>
                  <Table.Td>Today</Table.Td>
                  <Table.Td><Badge size="xs" color="green">Production</Badge></Table.Td>
                  <Table.Td>Canary</Table.Td>
                  <Table.Td>✓ Success</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Yesterday</Table.Td>
                  <Table.Td><Badge size="xs" color="blue">Staging</Badge></Table.Td>
                  <Table.Td>Direct</Table.Td>
                  <Table.Td>✓ Success</Table.Td>
                </Table.Tr>
              </Table.Tbody>
            </Table>
          </Tabs.Panel>

          <Tabs.Panel value="security" pt="md">
            <Stack gap="sm">
              <Group>
                <ThemeIcon color="green" variant="light" size="md">
                  <IconShieldCheck size={16} />
                </ThemeIcon>
                <Stack gap={0}>
                  <Text size="sm" fw={500}>Security Agent Approved</Text>
                  <Text size="xs" c="dimmed">0 CRITICAL · 0 HIGH CVEs · No PII detected</Text>
                </Stack>
              </Group>
            </Stack>
          </Tabs.Panel>
        </Tabs>

        <Group justify="flex-end" mt="lg" gap="sm">
          <Button size="xs" variant="light" color="gray" onClick={handleArchive}
            leftSection={<IconArrowDown size={12} />}>
            Archive
          </Button>
          <Button size="xs" color="green" onClick={handlePromote}
            leftSection={<IconArrowUp size={12} />}
            disabled={model.stage === 'Production'}>
            Promote to Production
          </Button>
        </Group>
      </Modal>
    </>
  )
}
