import numpy as np

from table_status.frames import is_blank_frame, iter_frames, read_frame_at
from table_status.models import TableReading, TableStatus, Zone
from table_status.pipeline import analyze_video, load_run
from table_status.zones import draw_zones, load_zones, save_zones, zones_from_boxes
from tests.helpers import textured_frame, write_test_video


def test_zones_from_boxes_scales_and_numbers_left_to_right():
    boxes = [[100, 500, 300, 700], [0, 0, 500, 250]]
    zones = zones_from_boxes(boxes, width=1000, height=2000)
    assert zones == [
        Zone(table_id="T1", x1=0, y1=0, x2=250, y2=1000),
        Zone(table_id="T2", x1=500, y1=200, x2=700, y2=600),
    ]


def test_zones_round_trip(tmp_path):
    zones = [Zone(table_id="T1", x1=1, y1=2, x2=3, y2=4)]
    save_zones(tmp_path / "zones.json", zones)
    assert load_zones(tmp_path / "zones.json") == zones


def test_draw_zones_does_not_modify_original():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    annotated = draw_zones(frame, [Zone(table_id="T1", x1=5, y1=20, x2=40, y2=45)])
    assert frame.sum() == 0
    assert annotated.sum() > 0


def test_iter_frames_samples_every_interval(tmp_path):
    video = write_test_video(tmp_path / "clip.mp4", seconds=10)
    timestamps = [t for t, _ in iter_frames(video, interval_s=3)]
    assert timestamps == [0, 3, 6, 9]


def test_read_frame_at_returns_image(tmp_path):
    video = write_test_video(tmp_path / "clip.mp4", seconds=2)
    assert read_frame_at(video, 1.0).shape == (48, 64, 3)


class CountingClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, image_jpeg, table_ids):
        self.calls += 1
        return [TableReading(table_id=t, status=TableStatus.EMPTY_CLEAN, confidence=1, note="") for t in table_ids]


def test_analyze_video_saves_results_and_resumes_without_new_calls(tmp_path):
    video = write_test_video(tmp_path / "clip.mp4", seconds=6)
    zones = [Zone(table_id="T1", x1=0, y1=0, x2=20, y2=20)]
    run_dir = tmp_path / "run"

    first = CountingClassifier()
    run = analyze_video(video, run_dir, zones, first, interval_s=2)
    assert first.calls == 3
    assert [f.timestamp_s for f in run.frames] == [0, 2, 4]
    assert (run_dir / "frames" / "00002000ms.jpg").exists()

    second = CountingClassifier()
    analyze_video(video, run_dir, zones, second, interval_s=2)
    assert second.calls == 0
    assert len(load_run(run_dir).frames) == 3


def test_analyze_video_keeps_sub_second_frames_apart(tmp_path):
    video = write_test_video(tmp_path / "clip.mp4", seconds=2)
    zones = [Zone(table_id="T1", x1=0, y1=0, x2=20, y2=20)]
    run = analyze_video(video, tmp_path / "run", zones, CountingClassifier(), interval_s=0.5)
    image_names = [frame.image_path for frame in run.frames]
    assert len(image_names) == 4
    assert len(set(image_names)) == 4


def test_is_blank_frame_flags_black_and_flat_frames_only():
    assert is_blank_frame(np.zeros((10, 10, 3), dtype=np.uint8))
    assert is_blank_frame(np.full((10, 10, 3), 128, dtype=np.uint8))
    assert not is_blank_frame(textured_frame())


def test_analyze_video_skips_model_on_blank_frames(tmp_path):
    video = write_test_video(tmp_path / "black.mp4", seconds=4, blank=True)
    zones = [Zone(table_id="T1", x1=0, y1=0, x2=20, y2=20)]
    classifier = CountingClassifier()

    run = analyze_video(video, tmp_path / "run", zones, classifier, interval_s=2)

    assert classifier.calls == 0
    assert all(frame.status_of("T1") == TableStatus.UNKNOWN for frame in run.frames)
