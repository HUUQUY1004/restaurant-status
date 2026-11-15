from dataclasses import dataclass
from statistics import mean

from table_status.models import OCCUPIED_STATUSES, TableStatus
from table_status.timeline import Segment


@dataclass(frozen=True)
class TableMetrics:
    table_id: str
    visits: int
    avg_visit_s: float | None
    avg_cleaning_wait_s: float | None
    attention_events: int
    longest_attention_s: float | None


def find_visits(segments: list[Segment]) -> list[tuple[float, float]]:
    """A visit is one unbroken stretch of guests sitting at the table."""
    visits: list[tuple[float, float]] = []
    for segment in sorted(segments, key=lambda s: s.start_s):
        if segment.status not in OCCUPIED_STATUSES:
            continue
        if visits and visits[-1][1] == segment.start_s:
            visits[-1] = (visits[-1][0], segment.end_s)
        else:
            visits.append((segment.start_s, segment.end_s))
    return visits


def table_metrics(table_id: str, segments: list[Segment]) -> TableMetrics:
    own = [segment for segment in segments if segment.table_id == table_id]
    visit_durations = [end - start for start, end in find_visits(own)]
    cleaning_durations = _durations_in(own, TableStatus.NEEDS_CLEANING)
    attention_durations = _durations_in(own, TableStatus.NEEDS_ATTENTION)
    return TableMetrics(
        table_id=table_id,
        visits=len(visit_durations),
        avg_visit_s=_average(visit_durations),
        avg_cleaning_wait_s=_average(cleaning_durations),
        attention_events=len(attention_durations),
        longest_attention_s=max(attention_durations) if attention_durations else None,
    )


def all_table_metrics(table_ids: list[str], segments: list[Segment]) -> list[TableMetrics]:
    return [table_metrics(table_id, segments) for table_id in table_ids]


def _durations_in(segments: list[Segment], status: TableStatus) -> list[float]:
    return [segment.duration_s for segment in segments if segment.status == status]


def _average(values: list[float]) -> float | None:
    return mean(values) if values else None
