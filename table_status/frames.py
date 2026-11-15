from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_frames(video_path: Path, interval_s: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp in seconds, BGR frame) every interval_s seconds."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        duration_s = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        timestamp_s = 0.0
        while timestamp_s < duration_s:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp_s * fps))
            ok, frame = capture.read()
            if not ok:
                break
            yield timestamp_s, frame
            timestamp_s += interval_s
    finally:
        capture.release()


def read_frame_at(video_path: Path, timestamp_s: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp_s * fps))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"No frame at {timestamp_s}s in {video_path}")
    return frame


def is_blank_frame(frame: np.ndarray, min_brightness: float = 15.0, min_contrast: float = 5.0) -> bool:
    """True for black or flat frames (fades, lost signal) that the model would only guess about."""
    return float(frame.mean()) < min_brightness or float(frame.std()) < min_contrast


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return buffer.tobytes()
