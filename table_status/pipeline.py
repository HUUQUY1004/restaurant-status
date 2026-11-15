from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from table_status.change import ChangeDetector
from table_status.frames import encode_jpeg, is_blank_frame, iter_frames
from table_status.models import FrameResult, RunResult, TableReading, TableStatus, Zone
from table_status.zones import draw_zones

RESULTS_FILE = "results.json"


class TableClassifier(Protocol):
    def classify(self, image_jpeg: bytes, table_ids: list[str]) -> list[TableReading]: ...


def load_run(run_dir: Path) -> RunResult | None:
    results_path = Path(run_dir) / RESULTS_FILE
    if not results_path.exists():
        return None
    return RunResult.model_validate_json(results_path.read_text(encoding="utf-8"))


def save_run(run_dir: Path, run: RunResult) -> None:
    (Path(run_dir) / RESULTS_FILE).write_text(run.model_dump_json(indent=2), encoding="utf-8")


def unknown_readings(table_ids: list[str], note: str) -> list[TableReading]:
    return [
        TableReading(table_id=table_id, status=TableStatus.UNKNOWN, confidence=1.0, note=note)
        for table_id in table_ids
    ]


def merge_readings(
    previous: list[TableReading] | None, fresh: list[TableReading], table_ids: list[str]
) -> list[TableReading]:
    """Fresh model answers win; tables the model was not asked about keep their last reading."""
    by_id = {reading.table_id: reading for reading in previous or []}
    by_id.update({reading.table_id: reading for reading in fresh})
    fallback = unknown_readings(table_ids, note="no reading yet")
    return [by_id.get(table_id, fallback[index]) for index, table_id in enumerate(table_ids)]


class FrameReader:
    """Turns one video frame into table readings, calling the model only when it is worth it."""

    def __init__(self, zones: list[Zone], classifier: TableClassifier, change_detector: ChangeDetector | None):
        self._zones = zones
        self._table_ids = [zone.table_id for zone in zones]
        self._classifier = classifier
        self._change_detector = change_detector
        self._last_readings: list[TableReading] | None = None

    def read(self, frame: np.ndarray, timestamp_s: float) -> tuple[list[TableReading], list[str]]:
        if is_blank_frame(frame):
            return unknown_readings(self._table_ids, note="frame is blank, model not called"), []

        to_check = self._tables_to_check(frame, timestamp_s)
        fresh = self._classify(frame, to_check) if to_check else []
        readings = merge_readings(self._last_readings, fresh, self._table_ids)
        if self._change_detector:
            self._change_detector.remember(frame, timestamp_s, to_check)
        self._last_readings = readings
        return readings, to_check

    def _tables_to_check(self, frame: np.ndarray, timestamp_s: float) -> list[str]:
        if self._change_detector is None or self._last_readings is None:
            return list(self._table_ids)
        return self._change_detector.tables_to_check(frame, timestamp_s)

    def _classify(self, frame: np.ndarray, table_ids: list[str]) -> list[TableReading]:
        # Only outline the tables we ask about, so the model does not answer for the others.
        zones = [zone for zone in self._zones if zone.table_id in table_ids]
        return self._classifier.classify(encode_jpeg(draw_zones(frame, zones)), table_ids)


def analyze_video(
    video_path: Path,
    run_dir: Path,
    zones: list[Zone],
    classifier: TableClassifier,
    interval_s: float,
    change_detector: ChangeDetector | None = None,
    on_progress: Callable[[FrameResult], None] | None = None,
) -> RunResult:
    run_dir = Path(run_dir)
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir) or RunResult(video=str(video_path), interval_s=interval_s, zones=zones)
    # Resume support: frames already classified were paid for, so skip them.
    done = {frame.timestamp_s for frame in run.frames}
    reader = FrameReader(zones, classifier, change_detector)

    for timestamp_s, frame in iter_frames(video_path, interval_s):
        if timestamp_s in done:
            continue
        image_path = run_dir / "frames" / f"{round(timestamp_s * 1000):08d}ms.jpg"
        cv2.imwrite(str(image_path), frame)
        readings, checked = reader.read(frame, timestamp_s)
        result = FrameResult(
            timestamp_s=timestamp_s, image_path=image_path.name, readings=readings, checked_tables=checked
        )
        run.frames.append(result)
        save_run(run_dir, run)
        if on_progress:
            on_progress(result)

    run.frames.sort(key=lambda frame: frame.timestamp_s)
    save_run(run_dir, run)
    return run
