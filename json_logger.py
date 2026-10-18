import logging
import time
import json
from datetime import datetime, timezone

from openai import OpenAI

logger = logging.getLogger("llm")

client = OpenAI()

def call_with_logging(messages: list[dict], model: str = "gpt-4o") -> str:
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency_ms": round(latency_ms, 1),
            "success": True,
        }))
        return response.choices[0].message.content
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "success": False,
            "error": str(exc),
        }))
        raise