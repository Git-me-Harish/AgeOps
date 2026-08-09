import { Button, Group, Modal, TextInput, Stack, Text } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { useState } from 'react'
import { IconRefresh, IconPlayerPlay, IconShieldSearch, IconRobot } from '@tabler/icons-react'
import { useStore } from '../../store'

export function QuickActions() {
  const [opened, { open, close }] = useDisclosure(false)
  const [datasetUri, setDatasetUri] = useState('s3://mlflow-artifacts/datasets/latest.csv')
  const startWorkflow = useStore(s => s.startWorkflow)
  const loading = useStore(s => s.loadingWorkflow)

  const handleStart = async () => {
    if (!datasetUri.trim()) return
    close()
    try {
      const id = await startWorkflow(datasetUri)
      notifications.show({
        title: 'Workflow started',
        message: `ID: ${id.slice(0, 8)}… running in background`,
        color: 'green',
      })
    } catch (err) {
      notifications.show({ title: 'Error', message: String(err), color: 'red' })
    }
  }

  return (
    <>
      <Group gap="sm" wrap="wrap">
        <Button
          leftSection={<IconPlayerPlay size={16} />}
          onClick={open}
          loading={loading}
        >
          Start Pipeline
        </Button>
        <Button
          leftSection={<IconRefresh size={16} />}
          variant="light" color="green"
          onClick={() =>
            notifications.show({ title: 'Retraining queued', message: 'All production models queued for retraining', color: 'green' })
          }
        >
          Retrain All Models
        </Button>
        <Button
          leftSection={<IconShieldSearch size={16} />}
          variant="light" color="orange"
          onClick={() =>
            notifications.show({ title: 'Security scan triggered', message: 'Trivy + SAST scan running in CI/CD', color: 'orange' })
          }
        >
          Run Security Scan
        </Button>
        <Button
          leftSection={<IconRobot size={16} />}
          variant="light" color="violet"
          onClick={() =>
            notifications.show({ title: 'RL agent triggered', message: 'Daily RL optimization job started', color: 'violet' })
          }
        >
          Run RL Optimizer
        </Button>
      </Group>

      <Modal opened={opened} onClose={close} title="Start New Pipeline" centered>
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            Provide the Cloudflare R2 URI for the dataset to kick off a full end-to-end workflow.
          </Text>
          <TextInput
            label="Dataset URI"
            placeholder="s3://mlflow-artifacts/datasets/mydata.csv"
            value={datasetUri}
            onChange={e => setDatasetUri(e.target.value)}
            required
          />
          <Group justify="flex-end">
            <Button variant="subtle" onClick={close}>Cancel</Button>
            <Button onClick={handleStart} loading={loading}>Start</Button>
          </Group>
        </Stack>
      </Modal>
    </>
  )
}
