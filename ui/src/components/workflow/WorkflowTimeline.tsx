import { Box, Group, Text, Badge, Stack, Progress, ThemeIcon } from '@mantine/core'
import {
  IconDatabase, IconBrain, IconChartBar,
  IconRocket, IconActivity, IconCheck, IconX, IconClock,
} from '@tabler/icons-react'
import type { WorkflowStage, Workflow } from '../../types'

const STAGES: { id: WorkflowStage; label: string; icon: typeof IconDatabase }[] = [
  { id: 'data',       label: 'Data',       icon: IconDatabase },
  { id: 'training',   label: 'Training',   icon: IconBrain },
  { id: 'evaluation', label: 'Evaluation', icon: IconChartBar },
  { id: 'deployment', label: 'Deployment', icon: IconRocket },
  { id: 'monitoring', label: 'Monitoring', icon: IconActivity },
]

const STAGE_ORDER: WorkflowStage[] = ['data', 'training', 'evaluation', 'deployment', 'monitoring', 'done']

function getStageStatus(stage: WorkflowStage, current: WorkflowStage, hasError: boolean) {
  const stageIdx = STAGE_ORDER.indexOf(stage)
  const currentIdx = STAGE_ORDER.indexOf(current)
  if (current === 'error' && stageIdx === currentIdx - 1) return 'error'
  if (stageIdx < currentIdx) return 'done'
  if (stageIdx === currentIdx) return 'active'
  return 'pending'
}

interface Props {
  workflow: Workflow | null
}

export function WorkflowTimeline({ workflow }: Props) {
  if (!workflow) {
    return (
      <Box p="xl" ta="center">
        <Text c="dimmed" size="sm">No active workflow. Start one from the Dashboard.</Text>
      </Box>
    )
  }

  const { current_stage, status, errors } = workflow
  const progress = Math.round(
    (STAGE_ORDER.indexOf(current_stage === 'done' ? 'done' : current_stage) / (STAGE_ORDER.length - 1)) * 100
  )

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text size="sm" fw={500}>
          Workflow <Text span c="dimmed" ff="monospace">{workflow.workflow_id.slice(0, 8)}</Text>
        </Text>
        <Badge
          color={status === 'completed' ? 'green' : status === 'failed' ? 'red' : status === 'running' ? 'blue' : 'gray'}
          variant="light"
        >
          {status}
        </Badge>
      </Group>

      <Progress value={progress} animated={status === 'running'} color={status === 'failed' ? 'red' : 'blue'} />

      {/* Stage tiles */}
      <Group gap="xs" grow>
        {STAGES.map(({ id, label, icon: Icon }) => {
          const stageStatus = getStageStatus(id, current_stage, status === 'failed')
          const color = stageStatus === 'done' ? 'green' : stageStatus === 'active' ? 'blue' : stageStatus === 'error' ? 'red' : 'gray'
          const StatusIcon = stageStatus === 'done' ? IconCheck : stageStatus === 'error' ? IconX : stageStatus === 'active' ? Icon : IconClock

          return (
            <Box
              key={id}
              p="xs"
              ta="center"
              style={{
                borderRadius: 8,
                border: `1.5px solid var(--mantine-color-${color}-${stageStatus === 'active' ? '4' : '2'})`,
                background: `var(--mantine-color-${color}-0)`,
                opacity: stageStatus === 'pending' ? 0.5 : 1,
              }}
            >
              <ThemeIcon size="md" color={color} variant="light" mx="auto" mb={4}>
                <StatusIcon size={14} stroke={1.5} />
              </ThemeIcon>
              <Text size="xs" fw={500} c={`${color}.7`}>{label}</Text>
            </Box>
          )
        })}
      </Group>

      {errors.length > 0 && (
        <Box p="xs" bg="red.0" style={{ borderRadius: 8, border: '1px solid var(--mantine-color-red-2)' }}>
          {errors.map((e, i) => (
            <Text key={i} size="xs" c="red.7">• {e}</Text>
          ))}
        </Box>
      )}
    </Stack>
  )
}
