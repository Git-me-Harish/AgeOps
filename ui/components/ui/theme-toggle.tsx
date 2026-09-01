'use client';

import { useEffect, useState } from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';
import clsx from 'clsx';

type ThemeChoice = 'light' | 'dark' | 'system';
const STORAGE_KEY = 'mlops-theme';

function applyTheme(choice: ThemeChoice) {
  const root = document.documentElement;
  if (choice === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.setAttribute('data-theme', choice);
  }
}

const OPTIONS: { value: ThemeChoice; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'Match system', Icon: Monitor },
];

/**
 * Three-way theme control (light / dark / system) — see the blocking
 * inline script in app/layout.tsx for the pre-paint application that
 * avoids a flash of the wrong theme on load; this component only handles
 * the interactive switch and keeps localStorage in sync afterwards.
 */
export function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>('system');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const stored = (localStorage.getItem(STORAGE_KEY) as ThemeChoice | null) ?? 'system';
    setChoice(stored);
    setMounted(true);
    // The very first toggle interaction of a session shouldn't inherit
    // the page-load transition suppression below — but subsequent ones
    // should animate, so this flag flips off after the first paint.
    document.documentElement.classList.add('theme-ready');
  }, []);

  function select(next: ThemeChoice) {
    setChoice(next);
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  }

  // Avoid rendering a choice that might not match the pre-paint script's
  // decision until after hydration reads the real stored value.
  if (!mounted) return <div style={{ width: 96, height: 28 }} />;

  return (
    <div role="radiogroup" aria-label="Theme" className="theme-toggle">
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          role="radio"
          aria-checked={choice === value}
          title={label}
          onClick={() => select(value)}
          className={clsx('theme-toggle-btn', choice === value && 'theme-toggle-btn-active')}
        >
          <Icon size={14} strokeWidth={2.25} />
        </button>
      ))}
    </div>
  );
}
