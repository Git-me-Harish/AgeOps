import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { Providers } from './providers';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans-loaded', display: 'swap' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono-loaded', display: 'swap' });

export const metadata: Metadata = {
  title: 'Multi-Agent MLOps',
  description: 'Command center for the multi-agent MLOps platform',
};

// Runs synchronously before first paint so the page never flashes the
// wrong theme while React hydrates — reads the same 'mlops-theme'
// localStorage key components/ui/theme-toggle.tsx writes to.
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem('mlops-theme');
    if (stored === 'light' || stored === 'dark') {
      document.documentElement.setAttribute('data-theme', stored);
    }
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
