import { useEffect, useCallback } from 'react'
import { Grid, Card, Title, Text, Stack, Group, Skeleton } from '@mantine/core'
import { useStore } from '../store'
import { AgentStatusGrid } from '../components/agents/AgentStatusGrid'
import { WorkflowTimeline } from '../components/workflow/WorkflowTimeline'
import { MetricsPanel } from '../components/dashboard/MetricsPanel'
import { AlertPanel } from '../components/dashboard/AlertPanel'
import { QuickActions } from '../components/dashboard/QuickActions'

export default function DashboardPage() {
  const { agents, fetchAgents, metricsSummary, fetchMetrics, workflows, activeWorkflowId, pollWorkflow } =
    useStore()

  const activeWorkflow = activeWorkflowId ? workflows[activeWorkflowId] : null

  // Initial data fetch
  useEffect(() => {
    fetchAgents()
    fetchMetrics()
  }, [fetchAgents, fetchMetrics])

  // Poll active workflow every 3 s while running
  useEffect(() => {
    if (!activeWorkflowId) return
    if (activeWorkflow?.status === 'completed' || activeWorkflow?.status === 'failed') return
    const id = setInterval(() => pollWorkflow(activeWorkflowId), 3000)
    return () => clearInterval(id)
  }, [activeWorkflowId, activeWorkflow?.status, pollWorkflow])

  return (
    <Stack gap="lg">
      {/* Page header */}
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2} fw={700}>Dashboard</Title>
          <Text c="dimmed" size="sm">Real-time overview of the Multi-Agent MLOps platform</Text>
        </div>
        <QuickActions />
      </Group>

      {/* Workflow timeline */}
      <Card p="md">
        <Text fw={600} size="sm" mb="md">Active Workflow</Text>
        <WorkflowTimeline workflow={activeWorkflow ?? null} />
      </Card>

      <Grid gutter="md">
        {/* Agent status grid */}
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <Card p="md" h="100%">
            <Text fw={600} size="sm" mb="md">Agent Health</Text>
            {agents.length === 0
              ? <Skeleton height={160} />
              : <AgentStatusGrid agents={agents} />
            }
          </Card>
        </Grid.Col>

        {/* Alert panel */}
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <Card p="md" h="100%" style={{ maxHeight: 320, overflowY: 'auto' }}>
            <Text fw={600} size="sm" mb="md">Alerts</Text>
            <AlertPanel />
          </Card>
        </Grid.Col>
      </Grid>

      {/* Metrics charts */}
      <Card p="md">
        <Text fw={600} size="sm" mb="md">Experiment Metrics</Text>
        <MetricsPanel summary={metricsSummary} loading={!metricsSummary} />
      </Card>
    </Stack>
  )
}
