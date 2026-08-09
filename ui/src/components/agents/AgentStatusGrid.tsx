import { SimpleGrid, Card, Group, Badge, Text, ThemeIcon, Stack } from '@mantine/core'
import {
  IconRobot, IconDatabase, IconBrain, IconChartBar,
  IconRocket, IconActivity, IconShieldCheck, IconLock, IconTrendingUp,
} from '@tabler/icons-react'
import type { AgentCard } from '../../types'

const AGENT_ICONS: Record<string, typeof IconRobot> = {
  orchestrator:         IconRobot,
  planner:              IconBrain,
  data_agent:           IconDatabase,
  training_agent:       IconBrain,
  evaluation_agent:     IconChartBar,
  deployment_agent:     IconRocket,
  monitoring_agent:     IconActivity,
  governance_agent:     IconShieldCheck,
  security_agent:       IconLock,
  rl_optimization_agent: IconTrendingUp,
}

const AGENT_COLORS: Record<string, string> = {
  orchestrator:         'blue',
  planner:              'indigo',
  data_agent:           'cyan',
  training_agent:       'green',
  evaluation_agent:     'teal',
  deployment_agent:     'orange',
  monitoring_agent:     'violet',
  governance_agent:     'grape',
  security_agent:       'red',
  rl_optimization_agent: 'yellow',
}

interface Props {
  agents: AgentCard[]
  statuses?: Record<string, 'online' | 'offline' | 'busy' | 'error'>
}

export function AgentStatusGrid({ agents, statuses = {} }: Props) {
  return (
    <SimpleGrid cols={{ base: 2, sm: 3, md: 5 }} spacing="sm">
      {agents.map(agent => {
        const Icon = AGENT_ICONS[agent.agent_id] ?? IconRobot
        const color = AGENT_COLORS[agent.agent_id] ?? 'blue'
        const status = statuses[agent.agent_id] ?? 'online'
        const statusColor = { online: 'green', offline: 'gray', busy: 'yellow', error: 'red' }[status]

        return (
          <Card key={agent.agent_id} p="sm">
            <Stack gap={6} align="center">
              <ThemeIcon size="xl" radius="xl" color={color} variant="light">
                <Icon size={22} stroke={1.5} />
              </ThemeIcon>
              <Text size="xs" fw={600} ta="center" lineClamp={2}>
                {agent.name.replace(' Agent', '')}
              </Text>
              <Badge size="xs" color={statusColor} variant="dot">
                {status}
              </Badge>
            </Stack>
          </Card>
        )
      })}
    </SimpleGrid>
  )
}
