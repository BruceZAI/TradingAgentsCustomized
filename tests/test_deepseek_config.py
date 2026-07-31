"""DeepSeek thinking configuration propagation through the graph."""

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_graph_forwards_deepseek_thinking_mode():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"deepseek_thinking_mode": "max"}

    kwargs = graph._get_provider_kwargs("deepseek")

    assert kwargs["deepseek_thinking_mode"] == "max"


@pytest.mark.unit
def test_graph_uses_official_deepseek_default():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {}

    kwargs = graph._get_provider_kwargs("deepseek")

    assert kwargs["deepseek_thinking_mode"] == "high"
