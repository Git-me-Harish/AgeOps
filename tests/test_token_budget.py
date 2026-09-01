"""
tests/test_token_budget.py

Unit tests for TokenBudget.

Zero external dependencies — no OpenAI, no Redis, no Postgres.
All tests run entirely in-process.

Run:
    pytest tests/test_token_budget.py -v
"""
import pytest

from agents.llm_gateway import BudgetExceededError, TokenBudget


# Fixtures 
@pytest.fixture
def budget() -> TokenBudget:
    """Fresh budget with a 10 000 token limit per agent."""
    return TokenBudget(limit_per_agent=10_000)


# check() 
class TestBudgetCheck:
    def test_passes_when_under_limit(self, budget: TokenBudget) -> None:
        """First call well within the limit must not raise."""
        budget.check("planner", estimated_tokens=1_000)   # should not raise

    def test_passes_at_exact_limit(self, budget: TokenBudget) -> None:
        """Checking exactly at the limit must not raise."""
        budget.check("planner", estimated_tokens=10_000)

    def test_raises_when_over_limit(self, budget: TokenBudget) -> None:
        """Requesting more than the limit on an empty budget must raise."""
        with pytest.raises(BudgetExceededError):
            budget.check("planner", estimated_tokens=10_001)

    def test_raises_after_partial_usage(self, budget: TokenBudget) -> None:
        """After recording 9 000 tokens, a 2 000-token request must raise."""
        budget.record("planner", 9_000)
        with pytest.raises(BudgetExceededError):
            budget.check("planner", estimated_tokens=2_000)

    def test_passes_just_within_remaining(self, budget: TokenBudget) -> None:
        """After recording 9 000 tokens, a 1 000-token request must pass."""
        budget.record("planner", 9_000)
        budget.check("planner", estimated_tokens=1_000)   # should not raise

    def test_different_roles_are_independent(self, budget: TokenBudget) -> None:
        """Exhausting the planner budget must not affect the training budget."""
        budget.record("planner", 10_000)
        with pytest.raises(BudgetExceededError):
            budget.check("planner", estimated_tokens=1)
        # training is untouched — must not raise
        budget.check("training", estimated_tokens=9_000)

    def test_error_message_contains_role_and_numbers(self, budget: TokenBudget) -> None:
        """BudgetExceededError message must be actionable — include role, used, limit."""
        budget.record("evaluation", 9_500)
        with pytest.raises(BudgetExceededError, match="evaluation"):
            budget.check("evaluation", estimated_tokens=1_000)


# record() 
class TestBudgetRecord:
    def test_record_accumulates(self, budget: TokenBudget) -> None:
        """Multiple record() calls on the same role must sum correctly."""
        budget.record("planner", 1_000)
        budget.record("planner", 2_000)
        budget.record("planner", 3_000)
        assert budget.remaining("planner") == 10_000 - 6_000

    def test_record_zero_tokens(self, budget: TokenBudget) -> None:
        """Recording 0 tokens is a no-op — remaining must not change."""
        before = budget.remaining("planner")
        budget.record("planner", 0)
        assert budget.remaining("planner") == before

    def test_record_does_not_affect_other_roles(self, budget: TokenBudget) -> None:
        """Recording tokens for one role must not change another role's budget."""
        budget.record("planner", 5_000)
        assert budget.remaining("governance") == 10_000


# remaining() 
class TestBudgetRemaining:
    def test_full_budget_on_new_role(self, budget: TokenBudget) -> None:
        """A role with no prior usage must return the full limit."""
        assert budget.remaining("security") == 10_000

    def test_remaining_never_goes_negative(self, budget: TokenBudget) -> None:
        """
        Even if record() is called with more tokens than the limit,
        remaining() must return 0 (not a negative number).
        """
        budget.record("planner", 99_999)
        assert budget.remaining("planner") == 0

    def test_remaining_after_exact_exhaustion(self, budget: TokenBudget) -> None:
        budget.record("training", 10_000)
        assert budget.remaining("training") == 0


# summary() 
class TestBudgetSummary:
    def test_summary_empty_on_new_budget(self, budget: TokenBudget) -> None:
        """No usage recorded → summary must be an empty dict."""
        assert budget.summary() == {}

    def test_summary_contains_all_active_roles(self, budget: TokenBudget) -> None:
        """summary() must return one entry per role that has been recorded."""
        budget.record("planner",    1_000)
        budget.record("training",   2_000)
        budget.record("evaluation", 3_000)
        summary = budget.summary()
        assert summary == {"planner": 1_000, "training": 2_000, "evaluation": 3_000}

    def test_summary_is_copy_not_reference(self, budget: TokenBudget) -> None:
        """Mutating the returned summary dict must not affect the budget state."""
        budget.record("planner", 500)
        summary = budget.summary()
        summary["planner"] = 99_999
        # Internal state must be unchanged
        assert budget.remaining("planner") == 10_000 - 500


# Custom limit 
class TestCustomLimit:
    def test_small_limit(self) -> None:
        """Budget with limit=100 must enforce correctly."""
        b = TokenBudget(limit_per_agent=100)
        b.check("planner", estimated_tokens=100)
        with pytest.raises(BudgetExceededError):
            b.check("planner", estimated_tokens=101)

    def test_zero_limit_always_raises(self) -> None:
        """Budget with limit=0 must raise on any request > 0."""
        b = TokenBudget(limit_per_agent=0)
        with pytest.raises(BudgetExceededError):
            b.check("planner", estimated_tokens=1)