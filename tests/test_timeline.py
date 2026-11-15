from table_status.timeline import Segment, build_segments, smooth_statuses
from tests.helpers import S, make_run


def test_smooth_ignores_single_frame_flicker():
    raw = [S.GUESTS_EATING, S.GUESTS_EATING, S.NEEDS_ATTENTION, S.GUESTS_EATING]
    assert smooth_statuses(raw, min_run=2) == [S.GUESTS_EATING] * 4


def test_smooth_keeps_change_that_holds_long_enough():
    raw = [S.GUESTS_EATING, S.GUESTS_EATING, S.NEEDS_CLEANING, S.NEEDS_CLEANING]
    assert smooth_statuses(raw, min_run=2) == raw


def test_smooth_replaces_unknown_with_previous_status():
    raw = [S.GUESTS_EATING, S.GUESTS_EATING, S.UNKNOWN, S.UNKNOWN, S.GUESTS_EATING]
    assert smooth_statuses(raw, min_run=2) == [S.GUESTS_EATING] * 5


def test_smooth_keeps_first_run_even_if_short():
    raw = [S.EMPTY_CLEAN, S.GUESTS_WAITING, S.GUESTS_WAITING]
    assert smooth_statuses(raw, min_run=2) == raw


def test_smooth_with_min_run_one_changes_nothing():
    raw = [S.EMPTY_CLEAN, S.GUESTS_WAITING, S.EMPTY_CLEAN]
    assert smooth_statuses(raw, min_run=1) == raw


def test_build_segments_merges_frames_and_closes_last_segment():
    run = make_run({"T1": [S.EMPTY_CLEAN, S.EMPTY_CLEAN, S.GUESTS_EATING, S.GUESTS_EATING]})
    assert build_segments(run, min_run=2) == [
        Segment("T1", S.EMPTY_CLEAN, 0.0, 60.0),
        Segment("T1", S.GUESTS_EATING, 60.0, 120.0),
    ]


def test_build_segments_handles_each_table_separately():
    run = make_run({"T1": [S.EMPTY_CLEAN] * 2, "T2": [S.NEEDS_CLEANING] * 2})
    segments = build_segments(run, min_run=1)
    assert {(s.table_id, s.status) for s in segments} == {("T1", S.EMPTY_CLEAN), ("T2", S.NEEDS_CLEANING)}


def test_build_segments_empty_run():
    run = make_run({"T1": []})
    assert build_segments(run) == []
