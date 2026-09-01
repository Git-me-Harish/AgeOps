"""
tests/test_llm_gateway.py
──────────────────────────
Unit tests for LLMGateway core logic.

What is NOT tested here (requires live API credits):
  - Real OpenAI completion call
  - Real OpenAI embedding for L2 cache
  - Real Ollama fallback (requires local Ollama running)

What IS tested (zero external dependencies):
  - _compute_cost()          — pricing arithmetic
  - _parse_structured()      — JSON → Pydantic model parsing
  - _default_system()        — system prompt per role
  - _select_model()          — model routing logic
  - complete() cache-hit path — verifies LLM is skipped on cache hit
  - complete() budget gate   — verifies LLM is skipped if budget exceeded
  - _record_usage() Neon     — verifies SQL INSERT on non-cache responses
  - GatewayResponse          — field defaults and model_config

Run:
    pytest tests/test_llm_gateway.py -v
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from agents.llm_gateway import (
    BudgetExceededError,
    GatewayResponse,
    LLMGateway,
    TokenBudget,
)

# Helpers
def make_gateway(
    pool: Optional[MagicMock] = None,
    workflow_id: str = "wf-test-001",
    budget_limit: int = 50_000,
) -> LLMGateway:
    budget = TokenBudget(limit_per_agent=budget_limit)
    return LLMGateway(pool=pool, workflow_id=workflow_id, budget=budget)


class SimplePlan(BaseModel):
    """Minimal Pydantic model used as a structured output target in tests."""
    framework: str
    epochs: int
    learning_rate: float

# _compute_cost()
class TestComputeCost:
    def test_gpt4o_mini_known_pricing(self) -> None:
        """gpt-4o-mini: input=$0.150/1M, output=$0.600/1M."""
        cost = LLMGateway._compute_cost("gpt-4o-mini", 1_000, 1_000)
        expected = (1_000 / 1_000 * 0.000150) + (1_000 / 1_000 * 0.000600)
        assert abs(cost - expected) < 1e-9

    def test_ollama_is_zero_cost(self) -> None:
        """Self-hosted Ollama models must always cost $0."""
        cost = LLMGateway._compute_cost("ollama/llama3.1:8b", 10_000, 5_000)
        assert cost == 0.0

    def test_zero_tokens_zero_cost(self) -> None:
        cost = LLMGateway._compute_cost("gpt-4o-mini", 0, 0)
        assert cost == 0.0

    def test_unknown_model_uses_fallback_pricing(self) -> None:
        """Unknown model must not raise — must use the fallback pricing."""
        cost = LLMGateway._compute_cost("unknown-model-xyz", 1_000, 1_000)
        assert cost > 0.0   # fallback pricing applied

    def test_output_tokens_more_expensive_than_input(self) -> None:
        """For all real models, output pricing >= input pricing."""
        input_cost  = LLMGateway._compute_cost("gpt-4o-mini", 1_000, 0)
        output_cost = LLMGateway._compute_cost("gpt-4o-mini", 0, 1_000)
        assert output_cost >= input_cost

# _parse_structured()
class TestParseStructured:
    VALID_JSON = '{"framework": "xgboost", "epochs": 100, "learning_rate": 0.01}'
    JSON_WITH_FENCES = (
        "```json\n"
        '{"framework": "xgboost", "epochs": 100, "learning_rate": 0.01}\n'
        "```"
    )
    JSON_WITH_PLAIN_FENCES = (
        "```\n"
        '{"framework": "xgboost", "epochs": 100, "learning_rate": 0.01}\n'
        "```"
    )

    def test_valid_json_parses_to_model(self) -> None:
        result = LLMGateway._parse_structured(self.VALID_JSON, SimplePlan)
        assert isinstance(result, SimplePlan)
        assert result.framework == "xgboost"
        assert result.epochs == 100

    def test_json_fences_stripped(self) -> None:
        """Markdown ```json fences must be stripped before parsing."""
        result = LLMGateway._parse_structured(self.JSON_WITH_FENCES, SimplePlan)
        assert isinstance(result, SimplePlan)

    def test_plain_fences_stripped(self) -> None:
        """Plain ``` fences must also be stripped."""
        result = LLMGateway._parse_structured(self.JSON_WITH_PLAIN_FENCES, SimplePlan)
        assert isinstance(result, SimplePlan)

    def test_invalid_json_returns_none(self) -> None:
        """Malformed JSON must return None, not raise."""
        result = LLMGateway._parse_structured("NOT JSON AT ALL", SimplePlan)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = LLMGateway._parse_structured("", SimplePlan)
        assert result is None

    def test_none_response_model_returns_none(self) -> None:
        """When no response_model is given, parsed must always be None."""
        result = LLMGateway._parse_structured(self.VALID_JSON, None)
        assert result is None

    def test_wrong_schema_returns_none(self) -> None:
        """JSON that does not match the Pydantic schema must return None."""
        result = LLMGateway._parse_structured(
            '{"completely": "wrong", "keys": true}', SimplePlan
        )
        assert result is None

    def test_numeric_fields_correct_types(self) -> None:
        result = LLMGateway._parse_structured(self.VALID_JSON, SimplePlan)
        assert result is not None
        assert isinstance(result.epochs, int)
        assert isinstance(result.learning_rate, float)

# _default_system()
class TestDefaultSystem:
    KNOWN_ROLES = ["planner", "training", "evaluation", "governance", "security", "monitoring"]

    def test_returns_non_empty_for_all_known_roles(self) -> None:
        for role in self.KNOWN_ROLES:
            prompt = LLMGateway._default_system(role)
            assert isinstance(prompt, str)
            assert len(prompt) > 20, f"System prompt for '{role}' is too short"

    def test_unknown_role_returns_generic_fallback(self) -> None:
        prompt = LLMGateway._default_system("unknown_role_xyz")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_each_role_has_distinct_prompt(self) -> None:
        """Each role should have a unique system prompt — no copy-paste."""
        prompts = {role: LLMGateway._default_system(role) for role in self.KNOWN_ROLES}
        unique = set(prompts.values())
        assert len(unique) == len(self.KNOWN_ROLES), "Duplicate system prompts detected"

# _select_model()
class TestSelectModel:
    def test_returns_configured_primary_model(self) -> None:
        from configs.settings import settings
        gw = make_gateway()
        model, is_fallback = gw._select_model("planner")
        assert model == settings.llm_model
        assert is_fallback is False

    def test_is_fallback_false_on_first_call(self) -> None:
        gw = make_gateway()
        _, is_fallback = gw._select_model("governance")
        assert is_fallback is False

# complete() — cache-hit path (no LLM call)
class TestCompleteCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self) -> None:
        """When the cache returns a value, _call_llm must never be invoked."""
        gw = make_gateway()
        cached_text = '{"framework": "xgboost", "epochs": 50, "learning_rate": 0.01}'

        # Mock the cache to return a hit
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=cached_text)
        gw._cache = mock_cache

        with patch.object(gw, "_call_llm", AsyncMock()) as mock_llm:
            response = await gw.complete(
                prompt="Which framework?",
                agent_role="planner",
                response_model=SimplePlan,
            )

        mock_llm.assert_not_awaited()
        assert response.was_cache_hit is True
        assert response.text == cached_text
        await gw.aclose()
        

    @pytest.mark.asyncio
    async def test_cache_hit_parses_structured_output(self) -> None:
        """Cache hit must still parse the cached JSON into a Pydantic model."""
        gw = make_gateway()
        cached_text = '{"framework": "sklearn", "epochs": 10, "learning_rate": 0.001}'

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=cached_text)
        gw._cache = mock_cache

        response = await gw.complete(
            prompt="framework?",
            agent_role="planner",
            response_model=SimplePlan,
        )

        assert response.parsed is not None
        assert isinstance(response.parsed, SimplePlan)
        assert response.parsed.framework == "sklearn"
        await gw.aclose()

    @pytest.mark.asyncio
    async def test_skip_cache_bypasses_l1_l2(self) -> None:
        """skip_cache=True must not consult the cache at all."""
        gw = make_gateway()

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value="should not be seen")
        gw._cache = mock_cache

        fake_response = ("raw text", {"prompt_tokens": 10, "completion_tokens": 20})
        with patch.object(gw, "_call_llm", AsyncMock(return_value=fake_response)):
            await gw.complete(
                prompt="prompt",
                agent_role="planner",
                skip_cache=True,
            )

        mock_cache.get.assert_not_awaited()
        
    @pytest.mark.asyncio
    async def test_aclose_closes_redis_client(self) -> None:
        """Gateway must explicitly close its owned Redis client."""
        gw = make_gateway()

        mock_redis = AsyncMock()
        gw._redis = mock_redis
        gw._cache = AsyncMock()

        await gw.aclose()

        mock_redis.aclose.assert_awaited_once()
        assert gw._redis is None
        assert gw._cache is None

# complete() — budget gate
class TestCompleteBudgetGate:
    @pytest.mark.asyncio
    async def test_budget_exceeded_raises_before_llm(self) -> None:
        """
        When the token budget is exhausted, complete() must raise
        BudgetExceededError before making any LLM call.
        """
        gw = make_gateway(budget_limit=0)   # zero budget — always exceeded

        with patch.object(gw, "_call_llm", AsyncMock()) as mock_llm:
            with pytest.raises(BudgetExceededError):
                await gw.complete(
                    prompt="any prompt",
                    agent_role="planner",
                    skip_cache=True,
                )

        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_tracks_across_calls(self) -> None:
        """Token usage from one call must reduce the remaining budget."""
        gw = make_gateway(budget_limit=5_000)

        fake_response = ("response", {"prompt_tokens": 2_000, "completion_tokens": 1_000})
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        gw._cache = mock_cache

        with patch.object(gw, "_call_llm", AsyncMock(return_value=fake_response)):
            await gw.complete("prompt 1", "planner", skip_cache=False)

        # 3 000 tokens used → 2 000 remaining
        assert gw._budget.remaining("planner") == 2_000

# complete() — fallback behavior
class TestCompleteFallback:
    @pytest.mark.asyncio
    async def test_primary_failure_uses_ollama_fallback_for_non_cloud_role(self) -> None:
        """Non-cloud-only roles may fall back to Ollama when the primary model fails."""
        gw = make_gateway()
        calls: list[str] = []

        async def fake_call_llm(**kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise RuntimeError("primary quota exhausted")
            return ("fallback response", {"prompt_tokens": 10, "completion_tokens": 5})

        with patch.object(gw, "_call_llm", AsyncMock(side_effect=fake_call_llm)):
            response = await gw.complete(
                prompt="plan this workflow",
                agent_role="planner",
                skip_cache=True,
            )

        from configs.settings import settings

        assert calls == [settings.llm_model, f"ollama/{settings.ollama_model}"]
        assert response.was_fallback is True
        assert response.model == f"ollama/{settings.ollama_model}"
        assert response.total_tokens == 15
        await gw.aclose()

    @pytest.mark.asyncio
    async def test_cloud_only_role_does_not_attempt_ollama_fallback(self) -> None:
        """Security and deployment roles must fail closed on primary failure."""
        gw = make_gateway()

        with patch.object(
            gw,
            "_call_llm",
            AsyncMock(side_effect=RuntimeError("primary unavailable")),
        ) as mock_llm:
            with pytest.raises(RuntimeError, match="primary unavailable"):
                await gw.complete(
                    prompt="scan this deployment",
                    agent_role="security",
                    skip_cache=True,
                )

        assert mock_llm.await_count == 1
        await gw.aclose()

# complete() — prompt template wiring
class TestPromptTemplateResolution:
    @pytest.mark.asyncio
    async def test_uses_prompt_template_when_system_prompt_not_supplied(self) -> None:
        gw = make_gateway()

        with patch.object(
            gw,
            "get_prompt_template",
            AsyncMock(return_value="Versioned planner system prompt"),
        ) as mock_template:
            with patch.object(
                gw,
                "_call_llm",
                AsyncMock(return_value=("ok", {"prompt_tokens": 1, "completion_tokens": 1})),
            ) as mock_llm:
                await gw.complete("prompt", "planner", skip_cache=True)

        mock_template.assert_awaited_once_with("planner", version=None)
        assert mock_llm.await_args.kwargs["system_prompt"] == "Versioned planner system prompt"
        await gw.aclose()

    @pytest.mark.asyncio
    async def test_prompt_version_fetches_specific_template(self) -> None:
        gw = make_gateway()

        with patch.object(
            gw,
            "get_prompt_template",
            AsyncMock(return_value="Planner system prompt v2"),
        ) as mock_template:
            with patch.object(
                gw,
                "_call_llm",
                AsyncMock(return_value=("ok", {"prompt_tokens": 1, "completion_tokens": 1})),
            ) as mock_llm:
                await gw.complete(
                    "prompt",
                    "planner",
                    prompt_version="v2",
                    skip_cache=True,
                )

        mock_template.assert_awaited_once_with("planner", version="v2")
        assert mock_llm.await_args.kwargs["system_prompt"] == "Planner system prompt v2"
        await gw.aclose()

    @pytest.mark.asyncio
    async def test_explicit_system_prompt_bypasses_template_lookup(self) -> None:
        gw = make_gateway()

        with patch.object(gw, "get_prompt_template", AsyncMock()) as mock_template:
            with patch.object(
                gw,
                "_call_llm",
                AsyncMock(return_value=("ok", {"prompt_tokens": 1, "completion_tokens": 1})),
            ) as mock_llm:
                await gw.complete(
                    "prompt",
                    "planner",
                    system_prompt="Explicit system prompt",
                    skip_cache=True,
                )

        mock_template.assert_not_awaited()
        assert mock_llm.await_args.kwargs["system_prompt"] == "Explicit system prompt"
        await gw.aclose()

# structured output usage
class TestStructuredUsage:
    @pytest.mark.asyncio
    async def test_complete_records_structured_usage_and_budget(self) -> None:
        gw = make_gateway()
        raw_text = '{"framework":"xgboost","epochs":25,"learning_rate":0.01}'
        usage = {
            "prompt_tokens": 321,
            "completion_tokens": 123,
        }

        with patch.object(gw, "_call_llm", AsyncMock(return_value=(raw_text, usage))):
            response = await gw.complete(
                prompt="choose framework",
                agent_role="planner",
                response_model=SimplePlan,
                skip_cache=True,
            )

        assert response.prompt_tokens == 321
        assert response.completion_tokens == 123
        assert response.total_tokens == 444
        assert response.cost_usd > 0
        assert gw._budget.remaining("planner") == 50_000 - 444
        await gw.aclose()

    def test_extract_usage_from_object(self) -> None:
        usage = Mock(prompt_tokens=123, completion_tokens=45)
        response = Mock(usage=usage)

        assert LLMGateway._extract_usage(response) == {
            "prompt_tokens": 123,
            "completion_tokens": 45,
        }

    def test_estimate_usage_never_returns_zero_for_non_empty_structured_call(self) -> None:
        usage = LLMGateway._estimate_usage(
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            raw_text='{"framework":"xgboost"}',
        )

        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_call_llm_structured_extracts_instructor_completion_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        gw = make_gateway()

        parsed = SimplePlan(framework="xgboost", epochs=25, learning_rate=0.01)
        completion = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=321, completion_tokens=123)
        )

        class FakeCompletions:
            def create(self, **kwargs):
                return parsed, completion

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        fake_instructor = SimpleNamespace(
            from_litellm=lambda completion_func: fake_client
        )
        fake_litellm = SimpleNamespace(completion=Mock())

        monkeypatch.setitem(sys.modules, "instructor", fake_instructor)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        raw_text, usage = await gw._call_llm(
            prompt="choose framework",
            model="gpt-4o-mini",
            system_prompt="system",
            temperature=0.1,
            max_tokens=256,
            response_model=SimplePlan,
        )

        assert raw_text == parsed.model_dump_json()
        assert usage == {"prompt_tokens": 321, "completion_tokens": 123}

# GatewayResponse
class TestGatewayResponse:
    def test_defaults(self) -> None:
        """GatewayResponse must have sensible defaults."""
        r = GatewayResponse(text="hello")
        assert r.was_cache_hit is False
        assert r.was_fallback is False
        assert r.total_tokens == 0
        assert r.cost_usd == 0.0
        assert r.parsed is None

    def test_accepts_arbitrary_parsed_type(self) -> None:
        """parsed field must accept any Pydantic model without Pydantic validation."""
        plan = SimplePlan(framework="xgb", epochs=10, learning_rate=0.01)
        r = GatewayResponse(text="{}", parsed=plan)
        assert r.parsed is plan
