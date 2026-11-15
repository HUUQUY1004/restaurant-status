from dataclasses import dataclass

import cv2
import numpy as np

from table_status.models import Zone

CROP_SIZE = (64, 64)


@dataclass
class _Reference:
    crop: np.ndarray
    timestamp_s: float


def zone_crop(frame: np.ndarray, zone: Zone) -> np.ndarray:
    """Small blurred grayscale crop, so camera noise does not count as a change."""
    crop = frame[zone.y1 : zone.y2, zone.x1 : zone.x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(gray, CROP_SIZE), (5, 5), 0)


def zone_difference(before: np.ndarray, after: np.ndarray) -> float:
    """Mean absolute pixel change on a 0-255 scale."""
    return float(cv2.absdiff(before, after).mean())


class ChangeDetector:
    """Decides which tables look different since the model last checked them."""

    def __init__(self, zones: list[Zone], threshold: float, max_age_s: float):
        self._zones = zones
        self._threshold = threshold
        self._max_age_s = max_age_s
        self._references: dict[str, _Reference] = {}

    def tables_to_check(self, frame: np.ndarray, timestamp_s: float) -> list[str]:
        return [zone.table_id for zone in self._zones if self._needs_check(frame, timestamp_s, zone)]

    def remember(self, frame: np.ndarray, timestamp_s: float, table_ids: list[str]) -> None:
        for zone in self._zones:
            if zone.table_id in table_ids:
                self._references[zone.table_id] = _Reference(zone_crop(frame, zone), timestamp_s)

    def _needs_check(self, frame: np.ndarray, timestamp_s: float, zone: Zone) -> bool:
        reference = self._references.get(zone.table_id)
        if reference is None:
            return True
        # Re-check now and then even without movement, so slow changes are not missed forever.
        if timestamp_s - reference.timestamp_s >= self._max_age_s:
            return True
        return zone_difference(reference.crop, zone_crop(frame, zone)) > self._threshold
