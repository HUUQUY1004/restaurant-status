import json
from pathlib import Path

import cv2
import numpy as np

from table_status.models import Zone

ZONE_COLOR_BGR = (255, 200, 0)


def load_zones(path: Path) -> list[Zone]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Zone(**item) for item in data]


def save_zones(path: Path, zones: list[Zone]) -> None:
    payload = [zone.model_dump() for zone in zones]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def zones_from_boxes(boxes: list[list[int]], width: int, height: int) -> list[Zone]:
    """Turn Gemini boxes [ymin, xmin, ymax, xmax] (0-1000 scale) into pixel zones T1..Tn."""
    pixel_boxes = [
        (
            round(xmin * width / 1000),
            round(ymin * height / 1000),
            round(xmax * width / 1000),
            round(ymax * height / 1000),
        )
        for ymin, xmin, ymax, xmax in boxes
    ]
    # Number tables left to right, then top to bottom, so ids stay stable between runs.
    pixel_boxes.sort(key=lambda box: (box[0], box[1]))
    return [
        Zone(table_id=f"T{index}", x1=x1, y1=y1, x2=x2, y2=y2)
        for index, (x1, y1, x2, y2) in enumerate(pixel_boxes, start=1)
    ]


def draw_zones(
    frame: np.ndarray,
    zones: list[Zone],
    colors: dict[str, tuple[int, int, int]] | None = None,
    captions: dict[str, str] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    for zone in zones:
        color = (colors or {}).get(zone.table_id, ZONE_COLOR_BGR)
        caption = (captions or {}).get(zone.table_id, zone.table_id)
        cv2.rectangle(annotated, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 3)
        _draw_caption(annotated, caption, (zone.x1, zone.y1), color)
    return annotated


def _draw_caption(image: np.ndarray, text: str, origin: tuple[int, int], color) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin[0], max(origin[1], text_h + 8)
    cv2.rectangle(image, (x, y - text_h - 8), (x + text_w + 8, y), color, -1)
    cv2.putText(image, text, (x + 4, y - 5), font, scale, (0, 0, 0), thickness)
