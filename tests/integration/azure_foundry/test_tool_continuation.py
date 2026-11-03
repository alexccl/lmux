"""Azure Foundry chat over the Responses API carries encrypted reasoning through a tool-use loop."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, ChatResponse, FunctionDefinition, Tool, ToolMessage, UserMessage
from lmux_azure_foundry.params import AzureFoundryParams

_DEPLOYMENT = "gpt-6-luna"
_MAX_TOKENS = 2048
_PARAMS = AzureFoundryParams(api="responses", reasoning_effort="low")
_REQUIRES = ("AZURE_FOUNDRY_KEY", "AZURE_FOUNDRY_ENDPOINT")
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "azure_foundry"
_CALL_CASSETTE = _CASSETTES / "tool_continuation_call.json"
_RESULT_CASSETTE = _CASSETTES / "tool_continuation_result.json"
_STREAM_CASSETTE = _CASSETTES / "tool_continuation_stream.json"

_WEATHER_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)
_PROMPT = UserMessage(content="What is the weather in Denver? Use the get_weather tool.")
_WEATHER = json.dumps({"tempF": 68, "conditions": "clear"})


@pytest.mark.live
@pytest.mark.record
def test_tool_continuation(scenario: Callable[..., Any], foundry_provider: Callable[..., Any]) -> None:
    def _first(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
        return foundry_provider(auth, transport).chat(
            _DEPLOYMENT, [_PROMPT], tools=[_WEATHER_TOOL], max_tokens=_MAX_TOKENS, provider_params=_PARAMS
        )

    first = scenario(_CALL_CASSETTE, _first, requires=_REQUIRES)
    assert first.tool_calls is not None
    assert first.continuation is not None

    history = [
        _PROMPT,
        first.to_assistant_message(),
        ToolMessage(content=_WEATHER, tool_call_id=first.tool_calls[0].id),
    ]

    def _second(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
        return foundry_provider(auth, transport).chat(
            _DEPLOYMENT, history, tools=[_WEATHER_TOOL], max_tokens=_MAX_TOKENS, provider_params=_PARAMS
        )

    second = scenario(_RESULT_CASSETTE, _second, requires=_REQUIRES)
    assert "68" in (second.content or "") or "clear" in (second.content or "").lower()


@pytest.mark.live
@pytest.mark.record
def test_streamed_tool_call_continuation(scenario: Callable[..., Any], foundry_provider: Callable[..., Any]) -> None:
    def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401
        return list(
            foundry_provider(auth, transport).chat_stream(
                _DEPLOYMENT, [_PROMPT], tools=[_WEATHER_TOOL], max_tokens=_MAX_TOKENS, provider_params=_PARAMS
            )
        )

    chunks = scenario(_STREAM_CASSETTE, _stream, requires=_REQUIRES)
    assert any(c.tool_call_deltas for c in chunks)
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].continuation is not None
