"""Moonshot/Kimi model catalog and request-parameter behavior."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.openai_client import MoonshotChatOpenAI
from tradingagents.llm_clients.validators import validate_model


class _StructuredResult(BaseModel):
    decision: str


def _bound_kwargs(runnable):
    first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
    return getattr(first, "kwargs", {})


def _client(
    model: str,
    *,
    thinking_enabled: bool = True,
    k3_reasoning_effort: str = "max",
    temperature: float | None = None,
) -> MoonshotChatOpenAI:
    return MoonshotChatOpenAI(
        model=model,
        api_key="placeholder",
        base_url="https://api.moonshot.cn/v1",
        thinking_enabled=thinking_enabled,
        k3_reasoning_effort=k3_reasoning_effort,
        temperature=temperature,
    )


@pytest.mark.unit
def test_kimi_k3_is_offered_and_validated():
    for mode in ("quick", "deep"):
        model_ids = [value for _, value in get_model_options("moonshot", mode)]
        assert model_ids[0] == "kimi-k3"
    assert validate_model("moonshot", "kimi-k3") is True


@pytest.mark.unit
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_kimi_k3_uses_only_its_supported_reasoning_and_sampling_parameters(effort):
    payload = _client(
        "kimi-k3",
        thinking_enabled=False,
        k3_reasoning_effort=effort,
        temperature=0.2,
    )._get_request_payload(
        [HumanMessage(content="hi")]
    )

    assert payload["reasoning_effort"] == effort
    assert "temperature" not in payload
    assert "thinking" not in payload.get("extra_body", {})


def test_kimi_k3_rejects_invalid_reasoning_effort():
    with pytest.raises(ValueError, match="moonshot_k3_reasoning_effort"):
        _client("kimi-k3", k3_reasoning_effort="medium")._get_request_payload(
            [HumanMessage(content="hi")]
        )


@pytest.mark.unit
def test_kimi_k3_structured_output_uses_json_schema_without_tool_choice():
    kwargs = _bound_kwargs(_client("kimi-k3").with_structured_output(_StructuredResult))

    assert kwargs["response_format"] is _StructuredResult
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize("model", ["kimi-k2.6", "kimi-k2.5"])
def test_thinking_kimi_k2_structured_output_uses_json_schema(model):
    kwargs = _bound_kwargs(
        _client(model, thinking_enabled=True).with_structured_output(_StructuredResult)
    )

    assert kwargs["response_format"] is _StructuredResult
    assert "tool_choice" not in kwargs


@pytest.mark.unit
def test_non_thinking_kimi_k2_keeps_function_calling_structured_output():
    kwargs = _bound_kwargs(
        _client("kimi-k2.6", thinking_enabled=False).with_structured_output(
            _StructuredResult
        )
    )

    assert kwargs.get("tools")
    assert kwargs.get("tool_choice") is not None


@pytest.mark.unit
@pytest.mark.parametrize("model", ["kimi-k2.6", "kimi-k2.5"])
def test_kimi_k2_models_keep_configurable_thinking(model):
    payload = _client(model, thinking_enabled=False)._get_request_payload(
        [HumanMessage(content="hi")]
    )

    assert "reasoning_effort" not in payload
    assert payload["extra_body"]["thinking"] == {"type": "disabled"}
