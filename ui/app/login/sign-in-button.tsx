'use client';

import { signIn } from 'next-auth/react';
import { Github } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function SignInButton({ callbackUrl }: { callbackUrl: string }) {
  return (
    <Button style={{ width: '100%', justifyContent: 'center' }} onClick={() => signIn('github', { callbackUrl })}>
      <Github size={16} /> Sign in with GitHub
    </Button>
  );
}
