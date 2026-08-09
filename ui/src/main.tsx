import '@mantine/core/styles.css'
import '@mantine/notifications/styles.css'
import '@mantine/charts/styles.css'

import React from 'react'
import ReactDOM from 'react-dom/client'
import { MantineProvider, createTheme } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { BrowserRouter } from 'react-router-dom'
import App from './App'

const theme = createTheme({
  primaryColor: 'blue',
  fontFamily: 'Inter, -apple-system, sans-serif',
  headings: { fontFamily: 'Inter, -apple-system, sans-serif' },
  colors: {
    // Agent-specific colours matching the design system
    agentData:       ['#e7f5ff','#d0ebff','#a5d8ff','#74c0fc','#4dabf7','#339af0','#228be6','#1c7ed6','#1971c2','#1864ab'],
    agentTraining:   ['#ebfbee','#d3f9d8','#b2f2bb','#8ce99a','#69db7c','#51cf66','#40c057','#37b24d','#2f9e44','#2b8a3e'],
    agentDeployment: ['#fff4e6','#ffe8cc','#ffd8a8','#ffc078','#ffa94d','#ff922b','#fd7e14','#f76707','#e8590c','#d9480f'],
    agentMonitor:    ['#f8f0fc','#f3d9fa','#eebefa','#e599f7','#da77f2','#cc5de8','#be4bdb','#ae3ec9','#9c36b5','#862e9c'],
    agentSecurity:   ['#fff5f5','#ffe3e3','#ffc9c9','#ffa8a8','#ff8787','#ff6b6b','#fa5252','#f03e3e','#e03131','#c92a2a'],
  },
  defaultRadius: 'md',
  components: {
    Card: { defaultProps: { shadow: 'xs', withBorder: true, radius: 'md' } },
    Button: { defaultProps: { radius: 'md' } },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="light">
      <Notifications position="top-right" />
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </MantineProvider>
  </React.StrictMode>
)
