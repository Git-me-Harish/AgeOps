'use client';

/**
 * Real-time event stream client (Phase 6 §6.3). Connects directly to the
 * orchestrator's /ws/events (see mcp_servers/api_server.py) — in
 * production this is reachable at the same ingress host under /ws
 * (configs/kubernetes/ingress.yaml), so no BFF proxy hop is needed for a
 * WebSocket the way REST calls go through /api/gateway. For local dev,
 * point NEXT_PUBLIC_WS_URL at the orchestrator running on localhost:8000.
 *
 * Auto-reconnects with backoff; a dropped connection surfaces as
 * `connected: false` in the returned state rather than silently going
 * quiet — the Command Center page uses that to show a real "live" vs
 * "reconnecting" indicator instead of implying freshness it doesn't have.
 */
import { useEffect, useRef, useState } from 'react';
import { WsEvent, WsEventSchema } from './schemas';

function wsUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window === 'undefined') return '';
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/events`;
}

export function useEventStream(onEvent: (event: WsEvent) => void) {
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryDelay = 1000;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let closedByCleanup = false;

    function connect() {
      const url = wsUrl();
      if (!url) return;
      socket = new WebSocket(url);

      socket.onopen = () => {
        setConnected(true);
        retryDelay = 1000;
      };
      socket.onmessage = (evt) => {
        try {
          const parsed = JSON.parse(evt.data);
          if (parsed.channel === '_heartbeat') return;
          const result = WsEventSchema.safeParse(parsed);
          if (result.success) onEventRef.current(result.data);
        } catch {
          // Non-JSON frame — ignore rather than crash the stream.
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (closedByCleanup) return;
        retryTimer = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 15000);
      };
      socket.onerror = () => {
        socket?.close();
      };
    }

    connect();
    return () => {
      closedByCleanup = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  return { connected };
}
