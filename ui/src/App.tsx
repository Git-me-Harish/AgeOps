import { AppShell } from '@mantine/core'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useStore } from './store'
import { AppNavbar } from './components/layout/AppNavbar'
import { AppHeader } from './components/layout/AppHeader'
import DashboardPage from './pages/DashboardPage'
import WorkflowDesignerPage from './pages/WorkflowDesignerPage'
import ExperimentsPage from './pages/ExperimentsPage'
import ModelRegistryPage from './pages/ModelRegistryPage'
import AgentsPage from './pages/AgentsPage'
import SecurityPage from './pages/SecurityPage'

export default function App() {
  const sidebarOpen = useStore(s => s.sidebarOpen)

  return (
    <AppShell
      header={{ height: 60 }}
      navbar={{ width: 240, breakpoint: 'sm', collapsed: { mobile: !sidebarOpen } }}
      padding="md"
    >
      <AppShell.Header>
        <AppHeader />
      </AppShell.Header>

      <AppShell.Navbar>
        <AppNavbar />
      </AppShell.Navbar>

      <AppShell.Main bg="gray.0">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/workflows" element={<WorkflowDesignerPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/models" element={<ModelRegistryPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/security" element={<SecurityPage />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  )
}
