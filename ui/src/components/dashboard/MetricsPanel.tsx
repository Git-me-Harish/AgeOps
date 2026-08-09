import { Card, Group, Text, Stack, SimpleGrid, ThemeIcon, Progress, Skeleton } from '@mantine/core'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTip,
  ResponsiveContainer, BarChart, Bar, Legend,
} from 'recharts'
import { IconTrendingUp, IconActivity, IconShieldCheck, IconAlertTriangle } from '@tabler/icons-react'
import type { MetricsSummary } from '../../types'

interface MetricCardProps {
  label: string
  value: number | undefined
  min?: number
  max?: number
  threshold?: number
  color: string
  icon: typeof IconTrendingUp
  unit?: string
  lowerIsBetter?: boolean
}

function MetricCard({ label, value, min, max, threshold, color, icon: Icon, unit = '', lowerIsBetter }: MetricCardProps) {
  const pct = value !== undefined && max ? Math.round((value / max) * 100) : 0
  const good = value !== undefined && threshold !== undefined
    ? (lowerIsBetter ? value < threshold : value >= threshold)
    : true

  return (
    <Card p="md">
      <Group justify="space-between" mb="xs">
        <Group gap="xs">
          <ThemeIcon size="sm" color={color} variant="light">
            <Icon size={14} stroke={1.5} />
          </ThemeIcon>
          <Text size="sm" fw={500}>{label}</Text>
        </Group>
        {value !== undefined && (
          <Text size="lg" fw={700} c={good ? `${color}.7` : 'red.6'}>
            {(value * (unit === '%' ? 100 : 1)).toFixed(unit === '%' ? 1 : 3)}{unit}
          </Text>
        )}
      </Group>
      {value !== undefined ? (
        <Progress value={pct} color={good ? color : 'red'} size="sm" />
      ) : (
        <Skeleton height={8} radius="xl" />
      )}
      {min !== undefined && max !== undefined && (
        <Group justify="space-between" mt={4}>
          <Text size="xs" c="dimmed">min {min.toFixed(3)}</Text>
          <Text size="xs" c="dimmed">max {max.toFixed(3)}</Text>
        </Group>
      )}
    </Card>
  )
}

// Synthetic sparkline data — replace with real MLflow time-series query
const SPARKLINE_DATA = Array.from({ length: 12 }, (_, i) => ({
  run: `R${i + 1}`,
  accuracy: 0.72 + Math.random() * 0.18,
  drift: Math.random() * 0.12,
  validation: 0.85 + Math.random() * 0.1,
}))

interface Props {
  summary: MetricsSummary | null
  loading?: boolean
}

export function MetricsPanel({ summary, loading }: Props) {
  return (
    <Stack gap="md">
      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="sm">
        <MetricCard
          label="Accuracy" unit="%"
          value={summary?.accuracy?.mean}
          min={summary?.accuracy?.min}
          max={summary?.accuracy?.max}
          threshold={0.80} color="green" icon={IconTrendingUp}
        />
        <MetricCard
          label="F1 Score" unit="%"
          value={summary?.f1_score?.mean}
          min={summary?.f1_score?.min}
          max={summary?.f1_score?.max}
          threshold={0.75} color="blue" icon={IconActivity}
        />
        <MetricCard
          label="Drift Score"
          value={summary?.drift_score?.mean}
          min={summary?.drift_score?.min}
          max={summary?.drift_score?.max ?? 0.3}
          threshold={0.15} color="orange" icon={IconAlertTriangle}
          lowerIsBetter
        />
        <MetricCard
          label="Validation" unit="%"
          value={summary?.validation_score?.mean}
          min={summary?.validation_score?.min}
          max={summary?.validation_score?.max}
          threshold={0.85} color="violet" icon={IconShieldCheck}
        />
      </SimpleGrid>

      {/* Area chart — accuracy & drift over last N runs */}
      <Card p="md">
        <Text size="sm" fw={600} mb="sm">Accuracy vs Drift (last 12 runs)</Text>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={SPARKLINE_DATA} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <defs>
              <linearGradient id="gradAcc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#40c057" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#40c057" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradDrift" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fd7e14" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#fd7e14" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f3f5" />
            <XAxis dataKey="run" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
            <RechartsTip
              contentStyle={{ fontSize: 12, borderRadius: 8 }}
              formatter={(v: number) => v.toFixed(3)}
            />
            <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
            <Area type="monotone" dataKey="accuracy" stroke="#40c057" fill="url(#gradAcc)" strokeWidth={2} dot={false} />
            <Area type="monotone" dataKey="drift"    stroke="#fd7e14" fill="url(#gradDrift)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
    </Stack>
  )
}
