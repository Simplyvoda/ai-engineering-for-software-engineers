"""Turn raw API data into one-line human summaries."""

from .codes import describe_weather_code


def format_weather(city: str, current: dict) -> str:
    """Example output: 'Paris: Moderate rain, 16.4°C, wind 12.1 km/h'."""
    description = describe_weather_code(current.get("weather_code"))
    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")

    parts = [description]
    parts.append(f"{temp}°C" if temp is not None else "temperature unavailable")
    parts.append(f"wind {wind} km/h" if wind is not None else "wind unavailable")
    return f"{city}: " + ", ".join(parts)
