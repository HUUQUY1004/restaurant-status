from types import SimpleNamespace

import pytest
from google.genai.errors import APIError

from table_status.gemini import GeminiTableClassifier
from table_status.models import FrameReading, TableReading, TableStatus
from table_status.retry import call_with_retry


def api_error(code: int) -> APIError:
    return APIError(code, {"error": {"code": code, "message": "test", "status": "TEST"}})


class Flaky:
    """Fails with the given errors first, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_retries_rate_limit_then_succeeds_with_growing_waits():
    waits = []
    func = Flaky(api_error(429), api_error(503))
    assert call_with_retry(func, base_delay_s=1.0, sleep=waits.append) == "ok"
    assert func.calls == 3
    assert 1.0 <= waits[0] < 2.0
    assert 2.0 <= waits[1] < 3.0


def test_does_not_retry_client_errors():
    func = Flaky(api_error(400))
    with pytest.raises(APIError):
        call_with_retry(func, sleep=lambda _: None)
    assert func.calls == 1


def test_gives_up_after_max_attempts():
    func = Flaky(*[api_error(429)] * 5)
    with pytest.raises(APIError):
        call_with_retry(func, max_attempts=3, sleep=lambda _: None)
    assert func.calls == 3


def test_classifier_survives_one_rate_limit():
    parsed = FrameReading(tables=[TableReading(table_id="T1", status=TableStatus.EMPTY_CLEAN, confidence=1, note="")])
    responses = Flaky(api_error(429))

    def generate_content(**_):
        responses()
        return SimpleNamespace(parsed=parsed)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    classifier = GeminiTableClassifier(client, "test-model", sleep=lambda _: None)

    assert classifier.classify(b"jpeg", ["T1"])[0].status == TableStatus.EMPTY_CLEAN
    assert responses.calls == 2
