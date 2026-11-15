from dataclasses import dataclass
from itertools import groupby

from table_status.models import RunResult, TableStatus


@dataclass(frozen=True)
class Segment:
    table_id: str
    status: TableStatus
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def smooth_statuses(statuses: list[TableStatus], min_run: int) -> list[TableStatus]:
    """Ignore status flickers shorter than min_run frames by keeping the previous stable status."""
    smoothed: list[TableStatus] = []
    for status, group in groupby(statuses):
        run_length = len(list(group))
        is_stable = run_length >= min_run and status != TableStatus.UNKNOWN
        keep_status = status if is_stable or not smoothed else smoothed[-1]
        smoothed.extend([keep_status] * run_length)
    return smoothed


def build_segments(run: RunResult, min_run: int = 2) -> list[Segment]:
    frames = sorted(run.frames, key=lambda frame: frame.timestamp_s)
    if not frames:
        return []
    timestamps = [frame.timestamp_s for frame in frames]
    video_end_s = timestamps[-1] + run.interval_s

    segments: list[Segment] = []
    for table_id in run.table_ids:
        raw = [frame.status_of(table_id) for frame in frames]
        segments.extend(_to_segments(table_id, smooth_statuses(raw, min_run), timestamps, video_end_s))
    return segments


def _to_segments(table_id, statuses, timestamps, video_end_s) -> list[Segment]:
    segments: list[Segment] = []
    index = 0
    for status, group in groupby(statuses):
        run_length = len(list(group))
        start_s = timestamps[index]
        index += run_length
        end_s = timestamps[index] if index < len(timestamps) else video_end_s
        segments.append(Segment(table_id, status, start_s, end_s))
    return segments
