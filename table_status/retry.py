import random
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai.errors import APIError

T = TypeVar("T")

# Rate limited, server overloaded or temporarily unavailable: worth waiting and trying again.
RETRYABLE_CODES = {429, 500, 503}


def is_retryable(error: Exception) -> bool:
    return isinstance(error, APIError) and error.code in RETRYABLE_CODES


def call_with_retry(
    func: Callable[[], T],
    max_attempts: int = 5,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call func, waiting 1s, 2s, 4s... plus jitter between attempts when Gemini is rate limited."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as error:
            if attempt == max_attempts or not is_retryable(error):
                raise
            sleep(base_delay_s * 2 ** (attempt - 1) + random.uniform(0, base_delay_s))
    raise AssertionError("unreachable")
