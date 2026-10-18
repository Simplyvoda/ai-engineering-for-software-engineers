import time
import random
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()

def call_with_retries(messages: list[dict], max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            return response.choices[0].message.content

        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            base_wait = 2 ** attempt
            jitter = random.uniform(0, base_wait * 0.3)
            wait_time = base_wait + jitter
            print(f"Rate limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        except APIError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Max retries exceeded")
