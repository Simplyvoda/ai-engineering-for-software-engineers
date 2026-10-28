# Weather Agent (LangGraph + Open-Meteo)

A LangGraph agent that answers weather questions using real data from
[Open-Meteo](https://open-meteo.com) (free, no API key needed for weather).

## Folder structure

```
weather-agent/
├── main.py                  # entry point: runs example questions
├── requirements.txt
├── .env.example             # copy to .env and add your OpenAI key
├── .gitignore
├── weather_agent/
│   ├── __init__.py          # exposes build_graph
│   ├── codes.py             # WMO weather codes -> words (+ fallback)
│   ├── http.py              # GET with retries + exponential backoff
│   ├── formatting.py        # raw JSON -> "Paris: Moderate rain, 16.4°C, wind 12.1 km/h"
│   ├── tools.py             # geocode_city, current_weather (the model's tools)
│   └── graph.py             # LangGraph wiring: llm <-> tools loop, then compose_final
└── tests/
    └── test_weather_agent.py  # offline tests (no key, no internet)
```

## How it flows

```
START -> llm --(wants tools?)--> tools --+
          ^                              |
          +------------------------------+
          |
          +--(no more tools)--> compose_final -> END
```

1. `llm` reads the question and asks for `geocode_city`.
2. `tools` runs it and loops back to `llm`.
3. `llm` asks for `current_weather` with the coordinates.
4. `tools` runs it and loops back again.
5. When no more tools are needed, `compose_final` writes the answer.

Tool errors (unknown city, API down after retries) become messages the model
reads, so the graph never crashes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your OpenAI key into .env
```

## Run

```bash
python main.py                                  # built-in examples
python main.py "What's the weather in Abuja?"   # your own question
pytest -q                                       # offline tests
```
