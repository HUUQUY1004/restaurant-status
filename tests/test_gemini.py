from types import SimpleNamespace

from table_status.gemini import GeminiTableClassifier, GeminiTableDetector, align_readings
from table_status.models import FrameReading, TableReading, TableStatus


class FakeModels:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.parsed)


def reading(table_id, status):
    return TableReading(table_id=table_id, status=status, confidence=0.8, note="")


def test_classifier_returns_readings_in_zone_order_and_marks_missing():
    parsed = FrameReading(tables=[reading("T2", TableStatus.GUESTS_EATING), reading("T9", TableStatus.EMPTY_CLEAN)])
    models = FakeModels(parsed)
    classifier = GeminiTableClassifier(SimpleNamespace(models=models), "test-model")

    result = classifier.classify(b"jpeg", ["T1", "T2"])

    assert [(r.table_id, r.status) for r in result] == [
        ("T1", TableStatus.UNKNOWN),
        ("T2", TableStatus.GUESTS_EATING),
    ]
    assert models.calls[0]["model"] == "test-model"
    assert "T1, T2" in models.calls[0]["contents"][1]


def test_align_readings_keeps_known_tables_only():
    result = align_readings([reading("T1", TableStatus.NEEDS_CLEANING)], ["T1"])
    assert result == [reading("T1", TableStatus.NEEDS_CLEANING)]


def test_detector_drops_malformed_boxes():
    parsed = SimpleNamespace(tables=[SimpleNamespace(box_2d=[1, 2, 3, 4]), SimpleNamespace(box_2d=[1, 2])])
    detector = GeminiTableDetector(SimpleNamespace(models=FakeModels(parsed)), "test-model")
    assert detector.detect_boxes(b"jpeg") == [[1, 2, 3, 4]]
