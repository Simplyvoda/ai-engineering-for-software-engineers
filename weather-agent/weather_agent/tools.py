"""Tools the model can call: geocoding and current weather (Open-Meteo, no API key)."""

from langchain_core.tools import tool

from .formatting import format_weather
from .http import get_json_with_retry

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@tool
def geocode_city(city: str) -> dict:
    """Look up a city's latitude and longitude. Call this before current_weather."""
    data = get_json_with_retry(GEOCODE_URL, {"name": city, "count": 1, "format": "json"})
    results = data.get("results") or []
    if not results:
        # Deliberate, honest failure: no pretending we found it.
        raise ValueError(f"I couldn't find a city called '{city}'.")
    place = results[0]
    return {
        "name": place.get("name", city),
        "country": place.get("country", ""),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
    }


@tool
def current_weather(latitude: float, longitude: float, city: str) -> str:
    """Get the current weather at a latitude/longitude. `city` is used for the label."""
    data = get_json_with_retry(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
        },
    )
    current = data.get("current")
    if not current:
        raise ValueError("The forecast service didn't return data.")
    return format_weather(city, current)


TOOLS = [geocode_city, current_weather]
