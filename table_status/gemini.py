import os
import time
from collections.abc import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from table_status.models import FrameReading, TableReading, TableStatus
from table_status.retry import call_with_retry

DEFAULT_MODEL = "gemini-3-flash-preview"

CLASSIFY_PROMPT = """You are watching a restaurant security camera frame.
Each dining table is outlined with a box labelled {table_ids}.
For every labelled table, pick exactly one status:
- empty_clean: nobody seated, table clear and ready for new guests
- guests_waiting: guests seated, no food on the table yet
- guests_eating: guests seated with food or drinks in front of them
- needs_attention: a seated guest is looking around, waving or clearly trying to get staff attention
- needs_cleaning: nobody seated but dirty dishes, cups or leftovers remain
- unknown: the table is hidden or too blurry to judge
Return one entry per labelled table with a confidence from 0 to 1 and a short note on what you see."""

DETECT_PROMPT = """Detect every dining table in this restaurant image, including the chairs around it.
Return box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000."""


class DetectedTable(BaseModel):
    box_2d: list[int]


class DetectedTables(BaseModel):
    tables: list[DetectedTable]


def make_client() -> genai.Client:
    load_dotenv()
    return genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )


def model_name() -> str:
    load_dotenv()
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


class GeminiTableClassifier:
    """Asks Gemini for the status of each outlined table in one frame."""

    def __init__(self, client: genai.Client, model: str, sleep: Callable[[float], None] = time.sleep):
        self._client = client
        self._model = model
        self._sleep = sleep

    def classify(self, image_jpeg: bytes, table_ids: list[str]) -> list[TableReading]:
        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"),
                    CLASSIFY_PROMPT.format(table_ids=", ".join(table_ids)),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FrameReading,
                    temperature=0,
                ),
            ),
            sleep=self._sleep,
        )
        return align_readings(response.parsed.tables, table_ids)


class GeminiTableDetector:
    """Finds table boxes once so nobody has to draw zones by hand."""

    def __init__(self, client: genai.Client, model: str, sleep: Callable[[float], None] = time.sleep):
        self._client = client
        self._model = model
        self._sleep = sleep

    def detect_boxes(self, image_jpeg: bytes) -> list[list[int]]:
        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=[types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"), DETECT_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DetectedTables,
                    temperature=0,
                ),
            ),
            sleep=self._sleep,
        )
        return [table.box_2d for table in response.parsed.tables if len(table.box_2d) == 4]


def align_readings(readings: list[TableReading], table_ids: list[str]) -> list[TableReading]:
    """Keep one reading per known table, in zone order, marking skipped tables as unknown."""
    by_id = {reading.table_id: reading for reading in readings}
    return [
        by_id.get(table_id)
        or TableReading(table_id=table_id, status=TableStatus.UNKNOWN, confidence=0.0, note="not returned by model")
        for table_id in table_ids
    ]
