"""WMO weather codes -> human-readable descriptions.

Open-Meteo returns conditions as numbers (e.g. 63). This module translates
those numbers back into words a person understands.
"""

WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def describe_weather_code(code) -> str:
    """Return a description for a WMO code.

    Fallback: if the code is unknown (or missing), say so honestly instead of
    crashing or inventing a description.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown conditions"
    return WEATHER_CODES.get(code, f"Unknown conditions (code {code})")
