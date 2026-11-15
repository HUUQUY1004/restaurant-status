import cv2

from table_status.change import ChangeDetector
from table_status.models import TableReading, TableStatus, Zone
from table_status.pipeline import analyze_video, merge_readings
from table_status.usage import model_usage
from tests.helpers import textured_frame, write_test_video

LEFT = Zone(table_id="T1", x1=0, y1=0, x2=30, y2=48)
RIGHT = Zone(table_id="T2", x1=34, y1=0, x2=64, y2=48)


def with_object_on_left(frame):
    changed = frame.copy()
    cv2.rectangle(changed, (5, 10), (25, 40), (0, 0, 255), -1)
    return changed


def detector(max_age_s=100.0):
    return ChangeDetector([LEFT, RIGHT], threshold=6.0, max_age_s=max_age_s)


def test_checks_every_table_the_first_time():
    assert detector().tables_to_check(textured_frame(), 0) == ["T1", "T2"]


def test_skips_tables_that_did_not_change():
    d = detector()
    frame = textured_frame()
    d.remember(frame, 0, ["T1", "T2"])
    assert d.tables_to_check(frame, 1) == []


def test_only_rechecks_the_table_that_changed():
    d = detector()
    frame = textured_frame()
    d.remember(frame, 0, ["T1", "T2"])
    assert d.tables_to_check(with_object_on_left(frame), 1) == ["T1"]


def test_rechecks_after_max_age_even_without_change():
    d = detector(max_age_s=60)
    frame = textured_frame()
    d.remember(frame, 0, ["T1", "T2"])
    assert d.tables_to_check(frame, 59) == []
    assert d.tables_to_check(frame, 60) == ["T1", "T2"]


def test_merge_keeps_old_reading_for_tables_not_asked():
    old = [
        TableReading(table_id="T1", status=TableStatus.GUESTS_EATING, confidence=1, note="old"),
        TableReading(table_id="T2", status=TableStatus.GUESTS_EATING, confidence=1, note="old"),
    ]
    fresh = [TableReading(table_id="T2", status=TableStatus.NEEDS_CLEANING, confidence=1, note="new")]
    merged = merge_readings(old, fresh, ["T1", "T2"])
    assert [(r.table_id, r.note) for r in merged] == [("T1", "old"), ("T2", "new")]


class RecordingClassifier:
    def __init__(self):
        self.asked = []

    def classify(self, image_jpeg, table_ids):
        self.asked.append(list(table_ids))
        return [TableReading(table_id=t, status=TableStatus.GUESTS_EATING, confidence=1, note="") for t in table_ids]


def test_static_video_only_calls_model_once(tmp_path):
    video = write_test_video(tmp_path / "static.mp4", seconds=6)
    classifier = RecordingClassifier()

    run = analyze_video(
        video, tmp_path / "run", [LEFT, RIGHT], classifier, interval_s=1, change_detector=detector()
    )

    assert classifier.asked == [["T1", "T2"]]
    assert all(frame.status_of("T2") == TableStatus.GUESTS_EATING for frame in run.frames)
    usage = model_usage(run)
    assert (usage.model_calls, usage.table_checks, usage.naive_table_checks) == (1, 2, 12)
    assert round(usage.checks_saved_ratio, 2) == 0.83
    assert round(usage.calls_saved_ratio, 2) == 0.83


def test_without_detector_every_frame_is_sent(tmp_path):
    video = write_test_video(tmp_path / "static.mp4", seconds=3)
    classifier = RecordingClassifier()
    analyze_video(video, tmp_path / "run", [LEFT, RIGHT], classifier, interval_s=1)
    assert len(classifier.asked) == 3
