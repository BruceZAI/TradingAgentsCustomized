"""DeepSeek thinking configuration propagation through the graph."""

import pytest

from tradingagents.llm_clients.factory import build_llm_kwargs


@pytest.mark.unit
def test_graph_forwards_deepseek_thinking_mode():
    kwargs = build_llm_kwargs({"llm_provider": "deepseek", "deepseek_thinking_mode": "max"})

    assert kwargs["deepseek_thinking_mode"] == "max"


@pytest.mark.unit
def test_graph_uses_official_deepseek_default():
    kwargs = build_llm_kwargs({"llm_provider": "deepseek"})

    assert kwargs["deepseek_thinking_mode"] == "high"
