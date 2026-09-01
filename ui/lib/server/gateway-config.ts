/**
 * Internal backend service map for the Next.js BFF (app/api/gateway/...).
 * These env vars are set directly in configs/kubernetes/ingress.yaml's
 * mlops-ui Deployment (in-cluster DNS) and via .env.local for local dev
 * (each server run standalone with `uvicorn ...`).
 */
export const GATEWAY_SERVICES = {
  orchestrator: process.env.API_INTERNAL_URL ?? 'http://localhost:8000',
  registry: process.env.REGISTRY_SERVER_INTERNAL_URL ?? 'http://localhost:8002',
  packaging: process.env.PACKAGING_SERVER_INTERNAL_URL ?? 'http://localhost:8003',
  monitoring: process.env.MONITORING_SERVER_INTERNAL_URL ?? 'http://localhost:8004',
  // Prometheus itself, not an mlops-mcp server — same settings.prometheus_url
  // the Python agents already query (configs/settings.py), exposed here so
  // real chart data (Command Center, Live Serving) never has to be
  // hardcoded. Read-only: only GET /api/v1/query[_range] paths are ever hit.
  prometheus: process.env.PROMETHEUS_INTERNAL_URL ?? 'http://localhost:9090',
} as const;

export type GatewayService = keyof typeof GATEWAY_SERVICES;

export function isGatewayService(value: string): value is GatewayService {
  return value in GATEWAY_SERVICES;
}
