"""
agents/llm_gateway.py

LLM Gateway — single point of LLM access for every agent in the system.

No agent calls OpenAI / Anthropic / Ollama directly. All LLM calls flow
through this gateway. This gives us:

  1. Model routing      — primary (gpt-4o-mini / claude-haiku) + fallback (Ollama)
  2. Semantic cache     — Redis hash + pgvector cosine similarity (avoid duplicate API calls)
  3. Token budget       — per-agent, per-workflow hard limits (settings.llm_token_budget_per_agent)
  4. Retry + backoff    — tenacity with jitter on rate limit / transient errors
  5. Structured output  — instructor + Pydantic schema enforcement
  6. Cost tracking      — tokens + cost_usd → MLflow metric + Neon token_usage table
  7. Prompt versioning  — templates stored in Neon, retrieved by (role, version)
  8. Ollama fallback    — local llama3.1:8b at $0 when API quota is exhausted

Usage (in any agent):
    from agents.llm_gateway import LLMGateway, GatewayResponse

    gw = LLMGateway(pool, workflow_id="abc-123")
    response = await gw.complete(
        prompt="Analyse this dataset and recommend a training framework.",
        agent_role="planner",
        response_model=ExecutionPlan,   # Pydantic model → structured output
    )
    plan: ExecutionPlan = response.parsed
    print(f"tokens used: {response.total_tokens}  cost: ${response.cost_usd:.4f}")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Any, Optional, Type, TypeVar

import mlflow
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from configs.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Per-model pricing (USD per 1K tokens) — update when providers change pricing 
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":          {"input": 0.000150, "output": 0.000600},
    "gpt-4o":               {"input": 0.005000, "output": 0.015000},
    "claude-haiku-4-5":    {"input": 0.000800, "output": 0.004000},
    "claude-sonnet-4-6":  {"input": 0.003000, "output": 0.015000},
    "ollama/llama3.1:8b":   {"input": 0.0,      "output": 0.0},       # self-hosted = $0
}

# Roles that MUST use the cloud model — never fall back to Ollama for these
_CLOUD_ONLY_ROLES: frozenset[str] = frozenset({
    "security",
    "deployment",
})


# Response type
class GatewayResponse(BaseModel):
    """Typed result returned by LLMGateway.complete()."""
    text: str                              # raw response text
    parsed: Optional[Any] = None           # Pydantic model instance if response_model given
    model: str = ""                        # actual model used
    was_fallback: bool = False             # True if Ollama served this
    was_cache_hit: bool = False            # True if Redis/pgvector cache served
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    model_config = {"arbitrary_types_allowed": True}


# Budget tracker (in-memory per workflow — reset on new workflow)
class TokenBudget:
    """
    Per-agent, per-workflow token budget enforcer.
    Raises BudgetExceededError before making an API call that would breach the limit.
    """

    def __init__(self, limit_per_agent: int = 50_000) -> None:
        self._limit = limit_per_agent
        self._used: dict[str, int] = {}   # agent_role → tokens_consumed

    def check(self, agent_role: str, estimated_tokens: int = 1000) -> None:
        used = self._used.get(agent_role, 0)
        if used + estimated_tokens > self._limit:
            raise BudgetExceededError(
                f"Agent '{agent_role}' budget exhausted: "
                f"used={used} limit={self._limit} requested={estimated_tokens}"
            )

    def record(self, agent_role: str, tokens: int) -> None:
        self._used[agent_role] = self._used.get(agent_role, 0) + tokens

    def remaining(self, agent_role: str) -> int:
        return max(0, self._limit - self._used.get(agent_role, 0))

    def summary(self) -> dict[str, int]:
        return dict(self._used)


class BudgetExceededError(RuntimeError):
    """Raised when an agent would exceed its token budget."""


# Semantic cache
class SemanticCache:
    """
    Two-level semantic cache:
      Level 1 — Redis exact hash (SHA-256 of canonical prompt) → O(1) lookup
      Level 2 — pgvector cosine similarity on embedding → fuzzy match

    Cache hits return the stored response without an API call.
    Similarity threshold: settings.semantic_cache_similarity_threshold (default 0.92)
    TTL: settings.semantic_cache_ttl_seconds (default 3600)
    """

    def __init__(self, redis_client: Any, pool: Optional[Any]) -> None:
        self._redis = redis_client
        self._pool = pool   # asyncpg pool for pgvector level-2 lookup

    async def get(self, prompt: str, agent_role: str) -> Optional[str]:
        """Return cached response string or None on miss."""
        # Level 1: exact hash
        key = self._cache_key(prompt, agent_role)
        try:
            cached = await self._redis.get(key)
            if cached:
                logger.debug("SemanticCache L1 hit: role=%s key=%s", agent_role, key[:16])
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception as exc:
            logger.warning("Redis cache GET failed (non-fatal): %s", exc)

        # Level 2: pgvector similarity (only if DB pool available)
        if self._pool is not None:
            try:
                embedding = await self._embed(prompt)
                if embedding:
                    result = await self._pgvector_lookup(embedding, agent_role)
                    if result:
                        logger.debug("SemanticCache L2 hit: role=%s", agent_role)
                        # Backfill L1 for next time
                        await self._set_redis(key, result)
                        return result
            except Exception as exc:
                logger.warning("SemanticCache L2 lookup failed (non-fatal): %s", exc)

        return None

    async def set(
        self,
        prompt: str,
        agent_role: str,
        response: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Store a response in both Redis and pgvector."""
        key = self._cache_key(prompt, agent_role)
        try:
            await self._set_redis(key, response)
        except Exception as exc:
            logger.warning("Redis cache SET failed (non-fatal): %s", exc)

        if self._pool is not None:
            try:
                embedding = await self._embed(prompt)
                if embedding:
                    await self._pgvector_store(
                        query_hash=key,
                        agent_role=agent_role,
                        model=model,
                        response=response,
                        embedding=embedding,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            except Exception as exc:
                logger.warning("SemanticCache pgvector SET failed (non-fatal): %s", exc)

    async def _set_redis(self, key: str, value: str) -> None:
        await self._redis.setex(key, settings.semantic_cache_ttl_seconds, value)

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding using OpenAI text-embedding-3-small (1536 dims)."""
        try:
            import litellm
            response = await asyncio.to_thread(
                litellm.embedding,
                model="text-embedding-3-small",
                input=[text[:2048]],   # truncate to safe length
            )
            return response.data[0]["embedding"]
        except Exception as exc:
            logger.warning(
                "Embedding generation failed; L2 semantic cache disabled for this request: %s",
                exc,
            )
            return None

    async def _pgvector_lookup(
        self,
        embedding: list[float],
        agent_role: str,
    ) -> Optional[str]:
        """
        Find a similar cached response using cosine similarity.

        The SELECT and UPDATE run inside the same acquired connection so we
        use the matched row's id for the UPDATE — PostgreSQL does not support
        LIMIT on UPDATE statements, so the id-based approach is the only
        correct pattern here.
        """
        threshold = settings.semantic_cache_similarity_threshold
        embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, response_text,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM llm_cache
                WHERE agent_role = $2
                  AND expires_at > NOW()
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY similarity DESC
                LIMIT 1
                """,
                embedding_str, agent_role, threshold,
            )

            if row:
                # id-based UPDATE — deterministic, no LIMIT required
                await conn.execute(
                    """
                    UPDATE llm_cache
                    SET hit_count  = hit_count + 1,
                        last_hit_at = NOW()
                    WHERE id = $1
                    """,
                    row["id"],
                )
                return row["response_text"]

        return None

    async def _pgvector_store(
        self,
        query_hash: str,
        agent_role: str,
        model: str,
        response: str,
        embedding: list[float],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        from datetime import datetime, timedelta, timezone
        expires_at = datetime.now(tz=timezone.utc) + timedelta(
            seconds=settings.semantic_cache_ttl_seconds
        )
        embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_cache
                    (query_hash, agent_role, model, response_text, embedding,
                     prompt_tokens, completion_tokens, expires_at)
                VALUES ($1,$2,$3,$4,$5::vector,$6,$7,$8)
                ON CONFLICT (query_hash) DO UPDATE
                SET model             = excluded.model,
                    response_text     = excluded.response_text,
                    prompt_tokens     = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    expires_at        = excluded.expires_at,
                    last_hit_at       = NOW()
                """,
                query_hash, agent_role, model, response, embedding_str,
                prompt_tokens, completion_tokens, expires_at,
            )

    @staticmethod
    def _cache_key(prompt: str, agent_role: str) -> str:
        canonical = f"{agent_role}::{prompt.strip()}"
        return "llmcache:" + hashlib.sha256(canonical.encode()).hexdigest()

# LLM Gateway
class LLMGateway:
    """
    Single point of LLM access for all agents.

    Constructor injection — callers provide the DB pool and workflow context.
    All methods are async — call with await.

    Example:
        gw = LLMGateway(pool=pool, workflow_id="wf-123")
        resp = await gw.complete(
            prompt="...",
            agent_role="planner",
            response_model=ExecutionPlan,
        )
    """

    def __init__(
        self,
        pool: Optional[Any],
        workflow_id: str,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        self._pool = pool
        self._workflow_id = workflow_id
        self._budget = budget or TokenBudget(
            limit_per_agent=settings.llm_token_budget_per_agent
        )
        self._cache: Optional[SemanticCache] = None
        self._redis: Optional[Any] = None

    async def _ensure_cache(self) -> None:
        if self._cache is not None:
            return

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=False,
            )
            await self._redis.ping()

            self._cache = SemanticCache(
                redis_client=self._redis,
                pool=self._pool,
            )

            logger.debug("SemanticCache initialised (Redis + pgvector)")

        except Exception as exc:
            logger.warning(
                "Redis unavailable — semantic cache disabled (non-fatal): %s",
                exc,
            )
            self._cache = None


    async def aclose(self) -> None:
        """Close the Redis client owned by this gateway."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as exc:
                logger.debug("Redis client cleanup failed: %s", exc)
            finally:
                self._redis = None
                self._cache = None

    # Main entry point 
    async def complete(
        self,
        prompt: str,
        agent_role: str,
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        prompt_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_cache: bool = False,
    ) -> GatewayResponse:
        start_ms = int(time.monotonic() * 1000)

        # Budget pre-check (fast — no I/O)
        self._budget.check(agent_role, estimated_tokens=max_tokens or 1000)

        await self._ensure_cache()

        resolved_system_prompt = system_prompt
        if resolved_system_prompt is None:
            resolved_system_prompt = (
                await self.get_prompt_template(agent_role, version=prompt_version)
                or self._default_system(agent_role)
            )

        cache_prompt = self._cache_prompt(prompt, resolved_system_prompt)

        # Cache lookup
        if not skip_cache and self._cache is not None:
            cached = await self._cache.get(cache_prompt, agent_role)
            if cached is not None:
                parsed = self._parse_structured(cached, response_model)
                resp = GatewayResponse(
                    text=cached,
                    parsed=parsed,
                    was_cache_hit=True,
                    latency_ms=int(time.monotonic() * 1000) - start_ms,
                )
                await self._record_usage(agent_role, resp)
                return resp

        # Determine which model to use
        model, is_fallback = self._select_model(agent_role)

        # Execute with retry
        try:
            raw_text, usage = await self._call_llm(
                prompt=prompt,
                model=model,
                system_prompt=resolved_system_prompt,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                response_model=response_model,
            )
        except Exception as exc:
            # Primary model failed — attempt Ollama fallback if allowed
            if not is_fallback and agent_role not in _CLOUD_ONLY_ROLES:
                logger.warning(
                    "Primary model %s failed for role %s — trying Ollama fallback: %s",
                    model, agent_role, exc,
                )
                model = f"ollama/{settings.ollama_model}"
                is_fallback = True
                raw_text, usage = await self._call_llm(
                    prompt=prompt,
                    model=model,
                    system_prompt=resolved_system_prompt,
                    temperature=temperature if temperature is not None else settings.llm_temperature,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    response_model=None,   # Ollama doesn't support instructor structured output
                )
            else:
                raise

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self._compute_cost(model, prompt_tokens, completion_tokens)
        latency_ms = int(time.monotonic() * 1000) - start_ms

        # Budget post-record
        self._budget.record(agent_role, total_tokens)

        # Parse structured output
        parsed = self._parse_structured(raw_text, response_model)

        resp = GatewayResponse(
            text=raw_text,
            parsed=parsed,
            model=model,
            was_fallback=is_fallback,
            was_cache_hit=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        # Populate cache
        if not skip_cache and self._cache is not None and not is_fallback:
            await self._cache.set(
                prompt=cache_prompt,
                agent_role=agent_role,
                response=raw_text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # Persist usage record + MLflow metrics
        await self._record_usage(agent_role, resp)

        logger.info(
            "LLMGateway: role=%s model=%s tokens=%d cost=$%.4f latency_ms=%d cache=%s fallback=%s",
            agent_role, model, total_tokens, cost_usd, latency_ms,
            resp.was_cache_hit, is_fallback,
        )
        return resp

    # Template retrieval 

    async def get_prompt_template(
        self,
        role: str,
        version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve the latest (or specific) prompt template for an agent role.
        Returns None if no template is registered — caller uses inline prompt.
        """
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                if version:
                    row = await conn.fetchrow(
                        "SELECT content FROM prompt_templates WHERE role=$1 AND version=$2 LIMIT 1",
                        role, version,
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT content FROM prompt_templates "
                        "WHERE role=$1 AND deprecated_at IS NULL "
                        "ORDER BY created_at DESC LIMIT 1",
                        role,
                    )
            return row["content"] if row else None
        except Exception as exc:
            logger.warning("Prompt template retrieval failed: %s", exc)
            return None

    def budget_summary(self) -> dict[str, int]:
        """Return token usage summary across all agents in this workflow."""
        return self._budget.summary()

    # Internal 

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        reraise=True,
    )
    async def _call_llm(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        response_model: Optional[Type[T]],
    ) -> tuple[str, dict]:
        """
        Execute the LLM call via litellm.
        Uses instructor for structured output when response_model is given.
        Retry decorator handles transient failures (rate limits, timeouts).
        """
        import litellm

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        if response_model is not None:
            # Structured output via instructor
            try:
                import instructor
                client = instructor.from_litellm(litellm.completion)
                create = client.chat.completions.create
                if hasattr(client.chat.completions, "create_with_completion"):
                    create = client.chat.completions.create_with_completion

                result = await asyncio.to_thread(
                    create,
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_base=settings.ollama_base_url if model.startswith("ollama/") else None,
                )
                completion = None
                if isinstance(result, tuple) and len(result) == 2:
                    result, completion = result
                else:
                    completion = getattr(result, "_raw_response", None)

                raw_text = result.model_dump_json()
                usage = self._extract_usage(completion)
                if usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0:
                    usage = self._estimate_usage(messages, raw_text)
                return raw_text, usage
            except ImportError:
                logger.warning("instructor not installed — falling back to raw JSON parsing")

        # Raw completion
        response = await asyncio.to_thread(
            litellm.completion,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=settings.ollama_base_url if model.startswith("ollama/") else None,
            timeout=settings.ollama_timeout_seconds if model.startswith("ollama/") else 60,
        )
        raw_text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return raw_text, usage

    @staticmethod
    def _cache_prompt(prompt: str, system_prompt: str) -> str:
        """Include the resolved system prompt so versioned prompts cannot share stale cache entries."""
        return f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{prompt.strip()}"

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract token usage from LiteLLM/OpenAI-style objects or dictionaries."""
        if response is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

        return {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
        }

    @staticmethod
    def _estimate_usage(messages: list[dict[str, str]], raw_text: str) -> dict[str, int]:
        """
        Last-resort token estimate for clients that do not expose usage.
        Prefer provider usage whenever it is available; this prevents structured
        calls from being recorded as zero-cost/zero-token events.
        """
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        completion_chars = len(raw_text)
        return {
            "prompt_tokens": max(1, math.ceil(prompt_chars / 4)),
            "completion_tokens": max(1, math.ceil(completion_chars / 4)),
        }

    def _select_model(self, agent_role: str) -> tuple[str, bool]:
        """
        Decide which model to use for this agent role.
        Returns (model_string, is_fallback).

        Cloud-only roles: always use the primary model.
        All others: use primary by default; fallback happens in complete() on failure.
        """
        primary = settings.llm_model
        return primary, False

    @staticmethod
    def _default_system(agent_role: str) -> str:
        """Default system prompt per agent role."""
        role_prompts = {
            "planner": (
                "You are an expert MLOps architect. Your role is to analyse datasets, "
                "consult historical experiment data, and produce structured execution plans "
                "for machine learning workflows. Always reason step-by-step before concluding. "
                "Respond only with valid JSON matching the schema provided."
            ),
            "training": (
                "You are an expert ML engineer. Select the optimal training framework "
                "and hyperparameters based on dataset characteristics and prior experiment results. "
                "Reason about compute constraints and expected training time."
            ),
            "evaluation": (
                "You are a rigorous ML evaluator. Your task is to critically assess "
                "model predictions, identify edge cases, check for bias, and provide "
                "a structured verdict on model quality. Be sceptical — err on the side "
                "of blocking promotion if uncertain."
            ),
            "governance": (
                "You are an MLOps governance officer. Enforce policies strictly. "
                "Document every decision with clear rationale. Fail-closed when uncertain."
            ),
            "security": (
                "You are an ML supply chain security analyst. Identify vulnerabilities, "
                "credential leaks, and policy violations. Never approve a deployment "
                "with unmitigated critical CVEs."
            ),
            "monitoring": (
                "You are an ML monitoring engineer. Interpret drift metrics, latency trends, "
                "and accuracy changes. Recommend retraining when thresholds are breached."
            ),
        }
        return role_prompts.get(
            agent_role,
            "You are a helpful MLOps assistant. Respond concisely and accurately.",
        )

    @staticmethod
    def _parse_structured(text: str, response_model: Optional[Type[T]]) -> Optional[T]:
        """Attempt to parse raw text into a Pydantic model. Returns None on failure."""
        if response_model is None or not text:
            return None
        try:
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return response_model.model_validate_json(clean)
        except Exception as exc:
            logger.warning("Structured output parsing failed: %s — raw: %s...", exc, text[:100])
            return None

    @staticmethod
    def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute cost in USD from token counts and model pricing."""
        pricing = _MODEL_PRICING.get(model, {"input": 0.001, "output": 0.001})
        return (
            (prompt_tokens / 1000) * pricing["input"] +
            (completion_tokens / 1000) * pricing["output"]
        )

    async def _record_usage(self, agent_role: str, resp: GatewayResponse) -> None:
        """Persist token usage to Neon and log to MLflow."""
        # MLflow metrics
        try:
            mlflow.log_metrics({
                f"{agent_role}.prompt_tokens":     resp.prompt_tokens,
                f"{agent_role}.completion_tokens": resp.completion_tokens,
                f"{agent_role}.total_tokens":      resp.total_tokens,
                f"{agent_role}.cost_usd":          resp.cost_usd,
                f"{agent_role}.latency_ms":        resp.latency_ms,
            })
        except Exception:
            pass

        # Neon persistence
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO token_usage (
                        workflow_id, agent_role, model, was_fallback, was_cache_hit,
                        prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    self._workflow_id,
                    agent_role,
                    resp.model or "cache",
                    resp.was_fallback,
                    resp.was_cache_hit,
                    resp.prompt_tokens,
                    resp.completion_tokens,
                    resp.total_tokens,
                    resp.cost_usd,
                    resp.latency_ms,
                )
        except Exception as exc:
            logger.warning("token_usage persistence failed (non-fatal): %s", exc)
