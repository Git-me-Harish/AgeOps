import { useEffect, useState } from 'react'
import {
  Stack, Group, Title, Text, TextInput, SegmentedControl,
  SimpleGrid, Skeleton, Badge,
} from '@mantine/core'
import { IconSearch } from '@tabler/icons-react'
import { useStore } from '../store'
import { ModelCard } from '../components/models/ModelCard'
import type { ModelStage, RegisteredModel } from '../types'

const MOCK_MODELS: RegisteredModel[] = [
  { name: 'mlops-model', version: '3', stage: 'Production' as ModelStage, run_id: 'abc123def456', description: 'GBM classifier — production-grade', tags: { framework: 'sklearn', team: 'mlops' } },
  { name: 'mlops-model', version: '2', stage: 'Staging'    as ModelStage, run_id: 'bcd234ef5678', description: 'Candidate with higher F1 score',     tags: { framework: 'sklearn' } },
  { name: 'mlops-model', version: '1', stage: 'Archived'   as ModelStage, run_id: 'cde345fg6789', description: 'Initial baseline model',             tags: {} },
  { name: 'fraud-detector', version: '1', stage: 'Staging' as ModelStage, run_id: 'def456gh7890', description: 'XGBoost fraud detection v1',          tags: { domain: 'fraud' } },
]

export default function ModelRegistryPage() {
  const { models, fetchModels } = useStore()
  const [search, setSearch] = useState('')
  const [stage, setStage] = useState('all')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetchModels(stage === 'all' ? undefined : stage).finally(() => setLoading(false))
  }, [stage, fetchModels])

  // Fall back to mock data while API isn't connected
  const displayModels = (models.length > 0 ? models : MOCK_MODELS)
    .filter(m => m.name.toLowerCase().includes(search.toLowerCase()))
    .filter(m => stage === 'all' || m.stage === stage)

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2} fw={700}>Model Registry</Title>
          <Text c="dimmed" size="sm">Browse, search, and manage all model versions</Text>
        </div>
        <Group gap="xs">
          <Badge color="green" variant="light" size="lg">
            {displayModels.filter(m => m.stage === 'Production').length} in Production
          </Badge>
          <Badge color="blue" variant="light" size="lg">
            {displayModels.filter(m => m.stage === 'Staging').length} in Staging
          </Badge>
        </Group>
      </Group>

      <Group gap="sm">
        <TextInput
          placeholder="Search models…"
          leftSection={<IconSearch size={14} />}
          value={search}
          onChange={e => setSearch(e.target.value)}
          w={260} size="sm"
        />
        <SegmentedControl
          size="sm"
          value={stage}
          onChange={setStage}
          data={[
            { label: 'All',        value: 'all' },
            { label: 'Production', value: 'Production' },
            { label: 'Staging',    value: 'Staging' },
            { label: 'Archived',   value: 'Archived' },
          ]}
        />
      </Group>

      {loading
        ? <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
            {[1,2,3,4].map(i => <Skeleton key={i} height={100} radius="md" />)}
          </SimpleGrid>
        : <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
            {displayModels.map((m, i) => (
              <ModelCard key={`${m.name}-${m.version}-${i}`} model={m} />
            ))}
          </SimpleGrid>
      }
    </Stack>
  )
}
