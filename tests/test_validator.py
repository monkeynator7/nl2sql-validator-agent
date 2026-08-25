import pytest
import pandas as pd

from app.agents.validator import validate_business_rules


@pytest.mark.asyncio
async def test_validator_returns_expected_shape():
    """Smoke test — validator returns a dict with valid/errors keys."""
    df = pd.DataFrame()
    result = await validate_business_rules(df)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "errors" in result


@pytest.mark.asyncio
async def test_validator_flags_negative_sales_net():
    df = pd.DataFrame({"sales_net": [10.0, -1.0], "discount": [0, 0]})
    result = await validate_business_rules(df)
    assert result["valid"] is False
    assert any("sales_net_never_negative" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_validator_llm_timeout_does_not_raise(monkeypatch):
    """LLM timeout is swallowed; declarative rules still decide validity."""

    async def slow_llm(*_a, **_kw):
        import asyncio

        await asyncio.sleep(5.0)
        return '{"violations": []}'

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-timeout")
    monkeypatch.setenv("VALIDATOR_LLM_TIMEOUT_SEC", "0.05")

    import app.agents.validator as v

    monkeypatch.setattr(v.anthropic_client, "chat_completion", slow_llm)

    df = pd.DataFrame({"sales_net": [1.0], "discount": [0]})
    result = await validate_business_rules(df)
    assert "valid" in result
    assert "errors" in result
