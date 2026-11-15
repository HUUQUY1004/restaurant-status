from pathlib import Path

import cv2
import numpy as np

from table_status.models import FrameResult, TableStatus, Zone
from table_status.zones import draw_zones

STATUS_HEX = {
    TableStatus.EMPTY_CLEAN: "#2e9e5b",
    TableStatus.GUESTS_WAITING: "#e0a400",
    TableStatus.GUESTS_EATING: "#2f6fdf",
    TableStatus.NEEDS_ATTENTION: "#e5484d",
    TableStatus.NEEDS_CLEANING: "#8e4ec6",
    TableStatus.UNKNOWN: "#8b8d98",
}

STATUS_LABELS = {
    TableStatus.EMPTY_CLEAN: "Ready",
    TableStatus.GUESTS_WAITING: "Waiting for food",
    TableStatus.GUESTS_EATING: "Eating",
    TableStatus.NEEDS_ATTENTION: "Needs attention",
    TableStatus.NEEDS_CLEANING: "Needs cleaning",
    TableStatus.UNKNOWN: "Unknown",
}


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return blue, green, red


def status_overlay(run_dir: Path, frame: FrameResult, zones: list[Zone]) -> np.ndarray:
    """Return an RGB image with each table box coloured by its status."""
    image = cv2.imread(str(Path(run_dir) / "frames" / frame.image_path))
    colors = {r.table_id: hex_to_bgr(STATUS_HEX[r.status]) for r in frame.readings}
    captions = {r.table_id: f"{r.table_id} {STATUS_LABELS[r.status]}" for r in frame.readings}
    return cv2.cvtColor(draw_zones(image, zones, colors, captions), cv2.COLOR_BGR2RGB)
