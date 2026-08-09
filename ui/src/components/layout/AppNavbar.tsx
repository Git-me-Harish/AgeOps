import { Stack, NavLink, Text, Divider, Box } from '@mantine/core'
import {
  IconLayoutDashboard, IconGitBranch, IconFlask,
  IconDatabase, IconRobot, IconShieldCheck,
} from '@tabler/icons-react'
import { useNavigate, useLocation } from 'react-router-dom'

const NAV_ITEMS = [
  { label: 'Dashboard',         icon: IconLayoutDashboard, path: '/dashboard' },
  { label: 'Workflow Designer', icon: IconGitBranch,       path: '/workflows' },
  { label: 'Experiments',       icon: IconFlask,           path: '/experiments' },
  { label: 'Model Registry',    icon: IconDatabase,        path: '/models' },
  { label: 'Agents',            icon: IconRobot,           path: '/agents' },
  { label: 'Security',          icon: IconShieldCheck,     path: '/security' },
]

export function AppNavbar() {
  const navigate = useNavigate()
  const { pathname } = useLocation()

  return (
    <Box p="sm" h="100%">
      <Stack gap={4}>
        <Text size="xs" fw={600} c="dimmed" tt="uppercase" px="xs" mb={4}>
          Navigation
        </Text>
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.path}
            label={item.label}
            leftSection={<item.icon size={18} stroke={1.5} />}
            active={pathname === item.path}
            onClick={() => navigate(item.path)}
            styles={{ root: { borderRadius: 8 } }}
          />
        ))}
        <Divider my="sm" />
        <Text size="xs" c="dimmed" px="xs">
          Budget: <Text span fw={600} c="green.7">₹0 / ₹300</Text>
        </Text>
        <Text size="xs" c="dimmed" px="xs">
          Infra: Oracle Cloud + R2 + Neon
        </Text>
      </Stack>
    </Box>
  )
}
