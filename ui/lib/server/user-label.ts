import { auth, authIsConfigured } from '@/auth';

export async function getUserLabel(): Promise<string> {
  if (!authIsConfigured) return 'dev-mode';
  const session = await auth();
  return session?.user?.name ?? 'Signed out';
}
