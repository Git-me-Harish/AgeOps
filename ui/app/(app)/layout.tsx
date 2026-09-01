import { auth, authIsConfigured } from '@/auth';
import { roleAtLeast, type Role } from '@/lib/server/rbac';
import { Sidebar } from '@/components/layout/sidebar';
import { DevModeBanner } from '@/components/layout/dev-mode-banner';

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = authIsConfigured ? await auth() : null;
  const role = (session?.user as { role?: Role } | undefined)?.role;

  return (
    <div className="stack" style={{ minHeight: '100vh' }}>
      <DevModeBanner authConfigured={authIsConfigured} />
      <div className="row" style={{ flex: 1, alignItems: 'stretch' }}>
        <Sidebar isAdmin={roleAtLeast(role, 'admin')} />
        <div className="stack" style={{ flex: 1, minWidth: 0 }}>
          {children}
        </div>
      </div>
    </div>
  );
}
