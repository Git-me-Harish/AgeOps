import {
  Stack, Group, Title, Text, Card, SimpleGrid, ThemeIcon,
  Badge, Progress, Table, ScrollArea, Button, Tabs,
  RingProgress, ActionIcon, Tooltip, Code,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  IconShieldCheck, IconAlertTriangle, IconBug, IconLock,
  IconFileReport, IconRefresh, IconX,
} from '@tabler/icons-react'

// ── Mock data (replace with API calls in production) ──────────────────────────
const SECURITY_SCORE = 87

const VULN_SUMMARY = [
  { severity: 'Critical', count: 0, color: 'red' },
  { severity: 'High',     count: 2, color: 'orange' },
  { severity: 'Medium',   count: 7, color: 'yellow' },
  { severity: 'Low',      count: 14, color: 'blue' },
]

const POLICY_VIOLATIONS = [
  { ts: '2025-07-15 09:12', policy: 'block-unsafe-deployments', detail: "Deployment used 'latest' tag — blocked", remediation: 'Use semantic versioning (e.g. v1.2.3)' },
  { ts: '2025-07-15 10:05', policy: 'require-nonroot',          detail: 'Container ran as root — blocked',         remediation: 'Add runAsNonRoot: true to securityContext' },
]

const VULNS = [
  { pkg: 'requests', version: '2.28.0', cve: 'CVE-2023-32681', severity: 'High',   fix: '2.31.0' },
  { pkg: 'cryptography', version: '41.0.0', cve: 'CVE-2023-49083', severity: 'High', fix: '41.0.6' },
  { pkg: 'paramiko', version: '2.12.0', cve: 'CVE-2023-48795', severity: 'Medium', fix: '3.4.0' },
]

const COMPLIANCE_CHECKS = [
  { control: 'GDPR – Data lineage tracked',       status: 'pass' },
  { control: 'GDPR – Right to erasure supported', status: 'pass' },
  { control: 'SOC 2 – Audit logging enabled',      status: 'pass' },
  { control: 'SOC 2 – TLS in transit',             status: 'pass' },
  { control: 'HIPAA – PHI access logged',          status: 'warn' },
  { control: 'HIPAA – PHI encryption at rest',     status: 'pass' },
]

function StatCard({ label, value, color, icon: Icon }: { label: string; value: number; color: string; icon: typeof IconShieldCheck }) {
  return (
    <Card p="md">
      <Group gap="sm">
        <ThemeIcon size="xl" color={color} variant="light" radius="xl">
          <Icon size={20} stroke={1.5} />
        </ThemeIcon>
        <div>
          <Text size="xl" fw={700} c={color + '.7'}>{value}</Text>
          <Text size="xs" c="dimmed">{label}</Text>
        </div>
      </Group>
    </Card>
  )
}

export default function SecurityPage() {
  const triggerScan = () =>
    notifications.show({ title: 'Security scan triggered', message: 'Trivy + SAST running in CI/CD pipeline', color: 'orange' })

  const downloadReport = () =>
    notifications.show({ title: 'Report generated', message: 'SOC 2 compliance report downloaded', color: 'blue' })

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <div>
          <Title order={2} fw={700}>Security & Governance</Title>
          <Text c="dimmed" size="sm">Zero-trust posture, OPA policy enforcement, compliance automation</Text>
        </div>
        <Group gap="xs">
          <Button leftSection={<IconRefresh size={14} />} variant="light" color="orange" size="sm" onClick={triggerScan}>
            Trigger Scan
          </Button>
          <Button leftSection={<IconFileReport size={14} />} variant="light" size="sm" onClick={downloadReport}>
            Export Report
          </Button>
        </Group>
      </Group>

      {/* Security score + stat cards */}
      <SimpleGrid cols={{ base: 2, sm: 5 }} spacing="sm">
        <Card p="md" style={{ gridColumn: 'span 1' }}>
          <Group gap="sm" justify="center">
            <RingProgress
              size={90} thickness={8}
              sections={[{ value: SECURITY_SCORE, color: SECURITY_SCORE >= 80 ? 'green' : 'orange' }]}
              label={<Text ta="center" fw={700} size="lg">{SECURITY_SCORE}</Text>}
            />
            <div>
              <Text fw={600} size="sm">Security Score</Text>
              <Badge color="green" variant="light" size="sm">Good</Badge>
            </div>
          </Group>
        </Card>
        <StatCard label="Critical CVEs"      value={0}  color="red"    icon={IconBug} />
        <StatCard label="High CVEs"          value={2}  color="orange" icon={IconAlertTriangle} />
        <StatCard label="Policy Violations"  value={2}  color="yellow" icon={IconLock} />
        <StatCard label="Compliance Checks"  value={6}  color="green"  icon={IconShieldCheck} />
      </SimpleGrid>

      <Tabs defaultValue="vulns">
        <Tabs.List>
          <Tabs.Tab value="vulns"      leftSection={<IconBug size={14} />}>Vulnerabilities</Tabs.Tab>
          <Tabs.Tab value="policies"   leftSection={<IconLock size={14} />}>Policy Violations</Tabs.Tab>
          <Tabs.Tab value="compliance" leftSection={<IconShieldCheck size={14} />}>Compliance</Tabs.Tab>
        </Tabs.List>

        {/* ── Vulnerabilities ── */}
        <Tabs.Panel value="vulns" pt="md">
          <Card p="md" mb="sm">
            <Text fw={600} size="sm" mb="sm">Severity Breakdown</Text>
            <Stack gap="xs">
              {VULN_SUMMARY.map(v => (
                <Group key={v.severity} gap="sm">
                  <Text size="xs" w={60}>{v.severity}</Text>
                  <Progress value={v.count === 0 ? 0 : (v.count / 14) * 100} color={v.color} size="sm" flex={1} />
                  <Badge size="xs" color={v.color} variant="light">{v.count}</Badge>
                </Group>
              ))}
            </Stack>
          </Card>

          <Card p={0}>
            <ScrollArea>
              <Table striped highlightOnHover fz="xs">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Package</Table.Th>
                    <Table.Th>Version</Table.Th>
                    <Table.Th>CVE</Table.Th>
                    <Table.Th>Severity</Table.Th>
                    <Table.Th>Fix Version</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {VULNS.map(v => (
                    <Table.Tr key={v.cve}>
                      <Table.Td><Code fz="xs">{v.pkg}</Code></Table.Td>
                      <Table.Td>{v.version}</Table.Td>
                      <Table.Td><Code fz="xs">{v.cve}</Code></Table.Td>
                      <Table.Td>
                        <Badge size="xs" color={v.severity === 'High' ? 'orange' : 'yellow'} variant="light">
                          {v.severity}
                        </Badge>
                      </Table.Td>
                      <Table.Td c="green.7" fw={600}>{v.fix}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Card>
        </Tabs.Panel>

        {/* ── Policy Violations ── */}
        <Tabs.Panel value="policies" pt="md">
          <Stack gap="sm">
            {POLICY_VIOLATIONS.map((v, i) => (
              <Card key={i} p="md" withBorder style={{ borderColor: 'var(--mantine-color-orange-3)' }}>
                <Group justify="space-between" mb="xs">
                  <Group gap="xs">
                    <ThemeIcon size="sm" color="orange" variant="light">
                      <IconAlertTriangle size={12} />
                    </ThemeIcon>
                    <Code fz="xs">{v.policy}</Code>
                  </Group>
                  <Text size="xs" c="dimmed">{v.ts}</Text>
                </Group>
                <Text size="xs" mb={4}>{v.detail}</Text>
                <Text size="xs" c="green.7">→ Remediation: {v.remediation}</Text>
              </Card>
            ))}
          </Stack>
        </Tabs.Panel>

        {/* ── Compliance ── */}
        <Tabs.Panel value="compliance" pt="md">
          <Card p={0}>
            <Table striped fz="xs">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Control</Table.Th>
                  <Table.Th>Status</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {COMPLIANCE_CHECKS.map((c, i) => (
                  <Table.Tr key={i}>
                    <Table.Td>{c.control}</Table.Td>
                    <Table.Td>
                      <Badge
                        size="xs"
                        color={c.status === 'pass' ? 'green' : c.status === 'warn' ? 'yellow' : 'red'}
                        variant="light"
                      >
                        {c.status === 'pass' ? '✓ Pass' : '⚠ Warn'}
                      </Badge>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}
