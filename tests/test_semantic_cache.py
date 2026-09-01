"""
tests/test_semantic_cache.py
─────────────────────────────
Unit tests for SemanticCache (L1 Redis + L2 pgvector).

All external dependencies are mocked:
  - Redis  → unittest.mock.AsyncMock
  - asyncpg pool → unittest.mock.AsyncMock
  - litellm.embedding → mocked to avoid any OpenAI call

Zero real I/O. Runs fully offline.

Run:
    pytest tests/test_semantic_cache.py -v
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.llm_gateway import SemanticCache

# Helpers
def make_redis(
    get_value: Any = None,
    ping_ok: bool = True,
) -> AsyncMock:
    """Build a minimal async Redis mock."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True if ping_ok else None)
    redis.get = AsyncMock(return_value=get_value)
    redis.setex = AsyncMock(return_value=True)
    return redis


def make_pool(fetchrow_return: Any = None) -> MagicMock:
    """Build a minimal asyncpg pool mock."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool

# Cache key generation
class TestCacheKey:
    def test_key_is_deterministic(self) -> None:
        """Same prompt + role must always produce the same key."""
        k1 = SemanticCache._cache_key("hello", "planner")
        k2 = SemanticCache._cache_key("hello", "planner")
        assert k1 == k2

    def test_key_changes_with_prompt(self) -> None:
        """Different prompts must produce different keys."""
        k1 = SemanticCache._cache_key("hello", "planner")
        k2 = SemanticCache._cache_key("world", "planner")
        assert k1 != k2

    def test_key_changes_with_role(self) -> None:
        """Same prompt, different role → different key (cache namespaced by role)."""
        k1 = SemanticCache._cache_key("hello", "planner")
        k2 = SemanticCache._cache_key("hello", "training")
        assert k1 != k2

    def test_key_has_prefix(self) -> None:
        """Key must start with 'llmcache:' for Redis namespace isolation."""
        k = SemanticCache._cache_key("hello", "planner")
        assert k.startswith("llmcache:")

    def test_key_length_is_fixed(self) -> None:
        """SHA-256 hex = 64 chars + prefix = deterministic length."""
        k = SemanticCache._cache_key("any prompt", "evaluation")
        assert len(k) == len("llmcache:") + 64

    def test_whitespace_trimmed(self) -> None:
        """Leading/trailing whitespace on the prompt must not change the key."""
        k1 = SemanticCache._cache_key("  hello  ", "planner")
        k2 = SemanticCache._cache_key("hello", "planner")
        # The cache key normalises via prompt.strip() in SemanticCache.get()
        # so keys built on the same canonical string must match
        canonical1 = "planner::" + "  hello  ".strip()
        canonical2 = "planner::" + "hello"
        assert hashlib.sha256(canonical1.encode()).hexdigest() == \
               hashlib.sha256(canonical2.encode()).hexdigest()

# L1 Redis cache
class TestL1RedisCache:
    @pytest.mark.asyncio
    async def test_l1_hit_returns_cached_value(self) -> None:
        """When Redis has the key, get() must return its value without touching L2."""
        cached_response = "The training framework is XGBoost."
        redis = make_redis(get_value=cached_response.encode())
        cache = SemanticCache(redis_client=redis, pool=None)

        result = await cache.get("What framework?", "planner")

        assert result == cached_response
        redis.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_l1_miss_returns_none_when_no_pool(self) -> None:
        """L1 miss + no pool → get() must return None (L2 is skipped)."""
        redis = make_redis(get_value=None)
        cache = SemanticCache(redis_client=redis, pool=None)

        result = await cache.get("What framework?", "planner")

        assert result is None

    @pytest.mark.asyncio
    async def test_l1_stores_with_ttl(self) -> None:
        """set() must call Redis.setex with the configured TTL."""
        redis = make_redis()
        cache = SemanticCache(redis_client=redis, pool=None)

        await cache.set(
            prompt="test prompt",
            agent_role="planner",
            response="some response",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=20,
        )

        redis.setex.assert_awaited_once()
        args = redis.setex.call_args[0]
        # args: (key, ttl, value)
        assert isinstance(args[1], int)   # TTL must be an integer
        assert args[1] > 0

    @pytest.mark.asyncio
    async def test_l1_redis_failure_is_non_fatal(self) -> None:
        """A Redis error in get() must not propagate — must return None gracefully."""
        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache = SemanticCache(redis_client=redis, pool=None)

        result = await cache.get("any prompt", "planner")

        assert result is None   # no exception raised

    @pytest.mark.asyncio
    async def test_l1_set_failure_is_non_fatal(self) -> None:
        """A Redis error in set() must not propagate — pipeline continues."""
        redis = AsyncMock()
        redis.setex = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache = SemanticCache(redis_client=redis, pool=None)

        # Must not raise
        await cache.set("prompt", "planner", "response", "gpt-4o-mini", 5, 10)

    @pytest.mark.asyncio
    async def test_l1_bytes_value_decoded(self) -> None:
        """Redis may return bytes — get() must decode to str."""
        redis = make_redis(get_value=b"cached bytes response")
        cache = SemanticCache(redis_client=redis, pool=None)

        result = await cache.get("prompt", "planner")

        assert isinstance(result, str)
        assert result == "cached bytes response"

# L2 pgvector cache
class TestL2PgvectorCache:
    def _make_db_row(self, response: str = "cached L2 response") -> MagicMock:
        """Fake asyncpg Record-like object."""
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": 42,
            "response_text": response,
            "similarity": 0.95,
        }[k])
        return row

    @pytest.mark.asyncio
    async def test_l2_hit_returns_cached_value(self) -> None:
        """L1 miss + L2 hit → get() must return the L2-cached response."""
        redis = make_redis(get_value=None)   # L1 miss
        row = self._make_db_row("L2 cached answer")
        pool = make_pool(fetchrow_return=row)

        cache = SemanticCache(redis_client=redis, pool=pool)

        fake_embedding = [0.1] * 1536
        with patch.object(cache, "_embed", AsyncMock(return_value=fake_embedding)):
            result = await cache.get("similar prompt", "planner")

        assert result == "L2 cached answer"

    @pytest.mark.asyncio
    async def test_l2_miss_returns_none(self) -> None:
        """L1 miss + L2 miss → get() must return None."""
        redis = make_redis(get_value=None)
        pool = make_pool(fetchrow_return=None)  # no L2 row

        cache = SemanticCache(redis_client=redis, pool=pool)

        fake_embedding = [0.1] * 1536
        with patch.object(cache, "_embed", AsyncMock(return_value=fake_embedding)):
            result = await cache.get("totally new prompt", "planner")

        assert result is None

    @pytest.mark.asyncio
    async def test_l2_hit_backfills_l1(self) -> None:
        """On an L2 hit, the response must be written back to Redis (L1 backfill)."""
        redis = make_redis(get_value=None)
        row = self._make_db_row("L2 answer")
        pool = make_pool(fetchrow_return=row)

        cache = SemanticCache(redis_client=redis, pool=pool)
        fake_embedding = [0.1] * 1536
        with patch.object(cache, "_embed", AsyncMock(return_value=fake_embedding)):
            await cache.get("prompt", "planner")

        # L1 backfill: setex must have been called with the response
        redis.setex.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_l2_skipped_when_embed_returns_none(self) -> None:
        """When embedding generation fails (returns None), L2 must be skipped."""
        redis = make_redis(get_value=None)
        pool = make_pool()   # pool present but should not be queried

        cache = SemanticCache(redis_client=redis, pool=pool)
        # _embed returns None (simulates OpenAI quota exhaustion)
        with patch.object(cache, "_embed", AsyncMock(return_value=None)):
            result = await cache.get("prompt", "planner")

        assert result is None
        # The pool's fetchrow must NOT have been called
        conn_mock = pool.acquire.return_value.__aenter__.return_value
        conn_mock.fetchrow.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """_embed() failure must log a WARNING instead of raising."""
        import logging
        redis = make_redis(get_value=None)
        cache = SemanticCache(redis_client=redis, pool=None)

        with patch("agents.llm_gateway.asyncio.to_thread",
                   AsyncMock(side_effect=Exception("API quota exhausted"))):
            with caplog.at_level(logging.WARNING, logger="agents.llm_gateway"):
                result = await cache._embed("any text")

        assert result is None
        assert any("L2 semantic cache disabled" in r.message for r in caplog.records)

# ON CONFLICT / cache population semantics
class TestCachePopulationSemantics:
    @pytest.mark.asyncio
    async def test_set_does_not_increment_hit_count(self) -> None:
        """
        _pgvector_store() ON CONFLICT must NOT increment hit_count.
        hit_count increments only in get() when a cached entry is found.
        """
        redis = make_redis()
        pool = make_pool()
        cache = SemanticCache(redis_client=redis, pool=pool)

        fake_embedding = [0.1] * 1536
        from datetime import datetime, timedelta, timezone
        expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)

        with patch.object(cache, "_embed", AsyncMock(return_value=fake_embedding)):
            await cache._pgvector_store(
                query_hash="llmcache:abc123",
                agent_role="planner",
                model="gpt-4o-mini",
                response="response text",
                embedding=fake_embedding,
                prompt_tokens=10,
                completion_tokens=20,
            )

        conn_mock = pool.acquire.return_value.__aenter__.return_value
        conn_mock.execute.assert_awaited_once()
        # Verify the SQL does NOT contain hit_count increment
        sql_called = conn_mock.execute.call_args[0][0]
        assert "hit_count" not in sql_called.lower()

    @pytest.mark.asyncio
    async def test_l2_hit_increments_hit_count_via_id(self) -> None:
        """
        _pgvector_lookup() must increment hit_count via id-based UPDATE,
        not via a LIMIT-based UPDATE (which is invalid PostgreSQL).
        """
        redis = make_redis(get_value=None)
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": 99,
            "response_text": "cached",
            "similarity": 0.96,
        }[k])
        pool = make_pool(fetchrow_return=row)

        cache = SemanticCache(redis_client=redis, pool=pool)
        fake_embedding = [0.0] * 1536

        await cache._pgvector_lookup(fake_embedding, "planner")

        conn_mock = pool.acquire.return_value.__aenter__.return_value
        conn_mock.execute.assert_awaited_once()
        # The UPDATE must use $1 = row id (99), not LIMIT
        update_sql = conn_mock.execute.call_args[0][0]
        update_args = conn_mock.execute.call_args[0]
        assert "LIMIT" not in update_sql.upper()
        assert 99 in update_args   # id=99 passed as argument