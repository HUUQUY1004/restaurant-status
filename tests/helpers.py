from pathlib import Path

import cv2
import numpy as np

from table_status.models import FrameResult, RunResult, TableReading, TableStatus, Zone

S = TableStatus


def make_run(statuses_by_table: dict[str, list[TableStatus]], interval_s: float = 30.0) -> RunResult:
    table_ids = list(statuses_by_table)
    frame_count = len(next(iter(statuses_by_table.values())))
    frames = [
        FrameResult(
            timestamp_s=index * interval_s,
            image_path=f"{index:06d}.jpg",
            readings=[
                TableReading(table_id=t, status=statuses_by_table[t][index], confidence=0.9, note="")
                for t in table_ids
            ],
        )
        for index in range(frame_count)
    ]
    zones = [Zone(table_id=t, x1=0, y1=0, x2=10, y2=10) for t in table_ids]
    return RunResult(video="test.mp4", interval_s=interval_s, zones=zones, frames=frames)


def textured_frame(size: tuple[int, int] = (64, 48)) -> np.ndarray:
    width, height = size
    gradient = np.tile(np.linspace(40, 220, width, dtype=np.uint8), (height, 1))
    return cv2.merge([gradient, gradient, gradient])


def write_test_video(
    path: Path, seconds: int, fps: int = 5, size: tuple[int, int] = (64, 48), blank: bool = False
) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8) if blank else textured_frame(size)
    for _ in range(seconds * fps):
        writer.write(frame)
    writer.release()
    return path
