"""Offline tests: no OpenAI key and no internet needed.

Run with:  pytest -q
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage

from weather_agent import build_graph
from weather_agent.codes import describe_weather_code
from weather_agent.formatting import format_weather
from weather_agent.http import get_json_with_retry
from weather_agent.tools import current_weather, geocode_city


# ---------- weather codes ----------

def test_known_code():
    assert describe_weather_code(63) == "Moderate rain"


def test_unknown_code_falls_back():
    assert describe_weather_code(42) == "Unknown conditions (code 42)"


def test_missing_code_falls_back():
    assert describe_weather_code(None) == "Unknown conditions"


# ---------- formatting ----------

def test_format_weather():
    line = format_weather("Paris", {"temperature_2m": 16.4, "weather_code": 63, "wind_speed_10m": 12.1})
    assert line == "Paris: Moderate rain, 16.4°C, wind 12.1 km/h"


# ---------- retry with backoff ----------

@patch("weather_agent.http.time.sleep")
@patch("weather_agent.http.requests.get")
def test_retry_succeeds_after_failures(mock_get, mock_sleep):
    good = MagicMock()
    good.json.return_value = {"ok": True}
    mock_get.side_effect = [requests.ConnectionError(), requests.Timeout(), good]

    assert get_json_with_retry("https://x", {}) == {"ok": True}
    assert mock_get.call_count == 3
    assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]  # backoff grows


@patch("weather_agent.http.time.sleep")
@patch("weather_agent.http.requests.get", side_effect=requests.ConnectionError())
def test_retry_gives_up(mock_get, mock_sleep):
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        get_json_with_retry("https://x", {})


# ---------- tools ----------

@patch("weather_agent.tools.get_json_with_retry", return_value={"results": []})
def test_geocode_unknown_city_raises(_):
    with pytest.raises(ValueError, match="couldn't find"):
        geocode_city.invoke({"city": "Atlantisville"})


@patch("weather_agent.tools.get_json_with_retry",
       return_value={"current": {"temperature_2m": 30, "weather_code": 2, "wind_speed_10m": 9}})
def test_current_weather_formats(_):
    out = current_weather.invoke({"latitude": 6.45, "longitude": 3.39, "city": "Lagos"})
    assert out == "Lagos: Partly cloudy, 30°C, wind 9 km/h"


# ---------- full graph with a scripted fake model ----------

class FakeLLM:
    """Pretends to be a chat model: returns scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def _api(url, params):
    if "geocoding" in url:
        return {"results": [{"name": "Paris", "country": "France", "latitude": 48.85, "longitude": 2.35}]}
    return {"current": {"temperature_2m": 16.4, "weather_code": 63, "wind_speed_10m": 12.1}}


@patch("weather_agent.tools.get_json_with_retry", side_effect=_api)
def test_graph_loops_geocode_then_weather(_):
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[{"name": "geocode_city", "args": {"city": "Paris"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "current_weather",
                                           "args": {"latitude": 48.85, "longitude": 2.35, "city": "Paris"},
                                           "id": "2"}]),
        AIMessage(content="done with tools"),
        AIMessage(content="Paris: Moderate rain, 16.4°C."),
    ])
    result = build_graph(llm).invoke({"messages": [HumanMessage("Weather in Paris?")]})

    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert "Paris: Moderate rain, 16.4°C, wind 12.1 km/h" in tool_outputs
    assert result["messages"][-1].content == "Paris: Moderate rain, 16.4°C."


@patch("weather_agent.tools.get_json_with_retry", return_value={"results": []})
def test_graph_survives_tool_error(_):
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[{"name": "geocode_city", "args": {"city": "Atlantisville"}, "id": "1"}]),
        AIMessage(content="couldn't find it"),
        AIMessage(content="I couldn't find Atlantisville."),
    ])
    result = build_graph(llm).invoke({"messages": [HumanMessage("Weather in Atlantisville?")]})

    tool_msg = next(m for m in result["messages"] if m.type == "tool")
    assert "couldn't find" in tool_msg.content  # error became a message, not a crash
    assert result["messages"][-1].content == "I couldn't find Atlantisville."


def test_graph_skips_tools_when_not_needed():
    llm = FakeLLM([AIMessage(content="no tools"), AIMessage(content="You're welcome!")])
    result = build_graph(llm).invoke({"messages": [HumanMessage("Thanks!")]})
    assert not any(m.type == "tool" for m in result["messages"])
    assert result["messages"][-1].content == "You're welcome!"
