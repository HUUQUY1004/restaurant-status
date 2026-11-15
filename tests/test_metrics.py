from table_status.metrics import find_visits, table_metrics
from table_status.timeline import Segment
from tests.helpers import S


def seg(status, start_min, end_min, table_id="T1"):
    return Segment(table_id, status, start_min * 60.0, end_min * 60.0)


def test_visit_spans_waiting_eating_and_attention():
    segments = [
        seg(S.EMPTY_CLEAN, 0, 5),
        seg(S.GUESTS_WAITING, 5, 10),
        seg(S.GUESTS_EATING, 10, 40),
        seg(S.NEEDS_ATTENTION, 40, 45),
        seg(S.NEEDS_CLEANING, 45, 52),
        seg(S.EMPTY_CLEAN, 52, 60),
    ]
    assert find_visits(segments) == [(300.0, 2700.0)]


def test_metrics_for_two_visits():
    segments = [
        seg(S.GUESTS_EATING, 0, 30),
        seg(S.NEEDS_CLEANING, 30, 36),
        seg(S.GUESTS_WAITING, 36, 40),
        seg(S.NEEDS_ATTENTION, 40, 43),
        seg(S.GUESTS_EATING, 43, 66),
        seg(S.NEEDS_CLEANING, 66, 70),
    ]
    metrics = table_metrics("T1", segments)
    assert metrics.visits == 2
    assert metrics.avg_visit_s == 30 * 60
    assert metrics.avg_cleaning_wait_s == 5 * 60
    assert metrics.attention_events == 1
    assert metrics.longest_attention_s == 3 * 60


def test_metrics_ignore_other_tables_and_handle_no_data():
    segments = [seg(S.GUESTS_EATING, 0, 30, table_id="T2"), seg(S.EMPTY_CLEAN, 0, 30)]
    metrics = table_metrics("T1", segments)
    assert metrics.visits == 0
    assert metrics.avg_visit_s is None
    assert metrics.avg_cleaning_wait_s is None
    assert metrics.longest_attention_s is None
