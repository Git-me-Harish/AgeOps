import { Stack, Group, Text, Badge, ActionIcon, Box, ThemeIcon } from '@mantine/core'
import { IconAlertTriangle, IconShieldCheck, IconActivity, IconX } from '@tabler/icons-react'
import { useState } from 'react'

export type Alert = {
  id: string
  severity: 'critical' | 'warning' | 'info'
  source: string
  message: string
  time: string
}

// Severity config
const SEV = {
  critical: { color: 'red',    icon: IconAlertTriangle },
  warning:  { color: 'orange', icon: IconAlertTriangle },
  info:     { color: 'blue',   icon: IconActivity },
}

// Static seed alerts — in production these come from a WebSocket / SSE feed
const SEED_ALERTS: Alert[] = [
  { id: '1', severity: 'warning',  source: 'Monitoring Agent', message: 'Drift score 0.13 approaching threshold 0.15', time: '2 min ago' },
  { id: '2', severity: 'info',     source: 'Security Agent',   message: 'Pre-flight check passed for workflow a1b2c3d4', time: '5 min ago' },
  { id: '3', severity: 'critical', source: 'Deployment Agent', message: 'Canary rollback triggered: error rate 7.2% > 5%', time: '12 min ago' },
  { id: '4', severity: 'info',     source: 'Data Agent',       message: 'Dataset validated: 9,847 rows, score 0.96', time: '18 min ago' },
]

interface Props {
  alerts?: Alert[]
}

export function AlertPanel({ alerts = SEED_ALERTS }: Props) {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())
  const visible = alerts.filter(a => !dismissed.has(a.id))

  return (
    <Stack gap="xs">
      {visible.length === 0 && (
        <Box p="md" ta="center">
          <ThemeIcon size="xl" color="green" variant="light" mx="auto" mb="xs">
            <IconShieldCheck size={22} />
          </ThemeIcon>
          <Text size="sm" c="dimmed">All clear — no active alerts</Text>
        </Box>
      )}
      {visible.map(alert => {
        const cfg = SEV[alert.severity]
        const Icon = cfg.icon
        return (
          <Group
            key={alert.id}
            p="sm"
            gap="sm"
            align="flex-start"
            style={{
              borderRadius: 8,
              border: `1px solid var(--mantine-color-${cfg.color}-2)`,
              background: `var(--mantine-color-${cfg.color}-0)`,
            }}
          >
            <ThemeIcon size="sm" color={cfg.color} variant="light" mt={2} style={{ flexShrink: 0 }}>
              <Icon size={12} />
            </ThemeIcon>
            <Box flex={1} style={{ minWidth: 0 }}>
              <Group gap="xs" mb={2}>
                <Badge size="xs" color={cfg.color} variant="filled">{alert.severity}</Badge>
                <Text size="xs" c="dimmed">{alert.source}</Text>
                <Text size="xs" c="dimmed" ml="auto">{alert.time}</Text>
              </Group>
              <Text size="xs">{alert.message}</Text>
            </Box>
            <ActionIcon
              size="xs" variant="subtle" color="gray"
              onClick={() => setDismissed(s => new Set([...s, alert.id]))}
            >
              <IconX size={10} />
            </ActionIcon>
          </Group>
        )
      })}
    </Stack>
  )
}
