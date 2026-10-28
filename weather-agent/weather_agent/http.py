"""HTTP helper with retries and exponential backoff.

The internet is flaky. Instead of failing on the first hiccup, we try a few
times, waiting a little longer between each attempt (1s, 2s, 4s, ...).
"""

import time

import requests

DEFAULT_TIMEOUT = 10  # seconds


def get_json_with_retry(url: str, params: dict, retries: int = 3, backoff: float = 1.0) -> dict:
    """GET a URL and return parsed JSON, retrying on network/server errors.

    - Attempt the request.
    - On failure, wait `backoff * 2**attempt` seconds and try again.
    - After `retries` failed attempts, raise the last error.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()  # 4xx/5xx count as failures
            return response.json()
        except (requests.RequestException, ValueError) as err:
            last_error = err
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Request to {url} failed after {retries} attempts: {last_error}")
