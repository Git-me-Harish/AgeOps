import Redis from 'ioredis';

/**
 * Redis-backed fixed-window rate limiter — deliberately not an in-memory
 * counter, because an in-memory counter is silently wrong the moment
 * there's more than one Next.js replica (mlops-ui runs 2 replicas per
 * configs/kubernetes/ingress.yaml): each replica would enforce its own
 * separate limit, letting a client get N-times the intended budget by
 * hitting different pods. Redis is already real infrastructure here
 * (agents/events.py, the LLM semantic cache) — same instance, new keyspace.
 *
 * Degrades open (never blocks a request) if Redis is unreachable — a rate
 * limiter that can itself cause an outage is worse than no rate limiter.
 */
let client: Redis | null | undefined;

function getClient(): Redis | null {
  if (client !== undefined) return client;
  if (!process.env.REDIS_URL) {
    client = null;
    return client;
  }
  client = new Redis(process.env.REDIS_URL, {
    lazyConnect: true,
    maxRetriesPerRequest: 1,
    retryStrategy: () => null, // don't keep retrying — fail fast and degrade open
  });
  client.on('error', () => {}); // swallow — every call site already treats failure as "allow"
  return client;
}

export interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  resetSeconds: number;
}

/**
 * @param key identity to limit on — session user id, or `ip:<addr>` for
 *   unauthenticated dev-mode requests
 * @param bucket a short label so different endpoints get independent
 *   budgets (e.g. "read", "mutate")
 */
export async function checkRateLimit(
  key: string, bucket: string, limit: number, windowSeconds: number,
): Promise<RateLimitResult> {
  const redis = getClient();
  if (!redis) return { allowed: true, limit, remaining: limit, resetSeconds: windowSeconds };

  const redisKey = `ratelimit:${bucket}:${key}:${Math.floor(Date.now() / 1000 / windowSeconds)}`;
  try {
    if (redis.status === 'wait') await redis.connect().catch(() => {});
    const count = await redis.incr(redisKey);
    if (count === 1) await redis.expire(redisKey, windowSeconds);
    const remaining = Math.max(0, limit - count);
    return { allowed: count <= limit, limit, remaining, resetSeconds: windowSeconds };
  } catch {
    return { allowed: true, limit, remaining: limit, resetSeconds: windowSeconds };
  }
}

// Defaults: reads are cheap and frequent (dashboards poll), mutations are
// deliberately tighter since they trigger real backend side effects
// (K8s patches, retraining workflows, model promotions).
export const READ_LIMIT = { count: 120, windowSeconds: 60 };
export const MUTATE_LIMIT = { count: 20, windowSeconds: 60 };
