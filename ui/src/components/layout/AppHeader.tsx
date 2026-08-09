import {
  Group, Text, ActionIcon, Tooltip, Avatar, Badge,
  Box, Burger, useMantineTheme,
} from '@mantine/core'
import {
  IconBell, IconSearch, IconBrandGithub, IconRobot,
} from '@tabler/icons-react'
import { useStore } from '../../store'

export function AppHeader() {
  const theme = useMantineTheme()
  const toggle = useStore(s => s.toggleSidebar)

  return (
    <Group h="100%" px="md" justify="space-between">
      {/* Left: burger + brand */}
      <Group gap="sm">
        <Burger onClick={toggle} size="sm" hiddenFrom="sm" />
        <Group gap="xs">
          <IconRobot size={28} color={theme.colors.blue[6]} stroke={1.5} />
          <Text fw={700} size="lg" c="blue.7">
            MLOps
            <Text span c="gray.6" fw={400} size="sm"> Multi-Agent</Text>
          </Text>
        </Group>
      </Group>

      {/* Right: actions */}
      <Group gap="xs">
        <Tooltip label="Global search (Ctrl+K)">
          <ActionIcon variant="subtle" color="gray" size="lg">
            <IconSearch size={18} />
          </ActionIcon>
        </Tooltip>

        <Tooltip label="Notifications">
          <ActionIcon variant="subtle" color="gray" size="lg" pos="relative">
            <IconBell size={18} />
            <Badge
              size="xs" color="red" variant="filled"
              pos="absolute" top={4} right={4}
              style={{ pointerEvents: 'none' }}
            >
              2
            </Badge>
          </ActionIcon>
        </Tooltip>

        <Tooltip label="GitHub">
          <ActionIcon
            variant="subtle" color="gray" size="lg"
            component="a"
            href="https://github.com/Git-me-Harish/multi-agent-mlops"
            target="_blank"
          >
            <IconBrandGithub size={18} />
          </ActionIcon>
        </Tooltip>

        <Avatar size="sm" color="blue" radius="xl">H</Avatar>
      </Group>
    </Group>
  )
}
