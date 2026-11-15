from pathlib import Path

from streamlit.testing.v1 import AppTest

from table_status.models import TableReading, TableStatus, Zone
from table_status.pipeline import analyze_video
from tests.helpers import write_test_video

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class CyclingClassifier:
    statuses = [TableStatus.GUESTS_EATING, TableStatus.GUESTS_EATING, TableStatus.NEEDS_CLEANING]

    def __init__(self):
        self.index = 0

    def classify(self, image_jpeg, table_ids):
        status = self.statuses[self.index % len(self.statuses)]
        self.index += 1
        return [TableReading(table_id=t, status=status, confidence=0.9, note="test") for t in table_ids]


def test_app_shows_message_when_no_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert "No runs yet" in app.info[0].value


def test_app_renders_all_tabs_for_a_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = write_test_video(tmp_path / "clip.mp4", seconds=6)
    zones = [Zone(table_id="T1", x1=0, y1=0, x2=30, y2=30)]
    analyze_video(video, Path("runs/demo"), zones, CyclingClassifier(), interval_s=2)
    Path("runs/demo/labels.csv").write_text(
        "timestamp_s,table_id,true_status\n0.0,T1,guests_eating\n4.0,T1,empty_clean\n", encoding="utf-8"
    )

    app = AppTest.from_file(APP_PATH).run(timeout=30)

    assert not app.exception
    assert app.metric[0].value == "50%"
