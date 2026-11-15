import pandas as pd

from table_status.accuracy import accuracy, compare_with_labels, confusion_matrix, write_label_template
from tests.helpers import S, make_run


def test_label_template_samples_frames_and_leaves_truth_empty(tmp_path):
    run = make_run({"T1": [S.EMPTY_CLEAN] * 30, "T2": [S.GUESTS_EATING] * 30})
    path = tmp_path / "labels.csv"
    rows = write_label_template(run, path, sample_frames=5)
    labels = pd.read_csv(path)
    assert rows == 10
    assert labels["timestamp_s"].nunique() == 5
    assert labels["true_status"].isna().all()


def test_compare_counts_only_filled_labels():
    run = make_run({"T1": [S.EMPTY_CLEAN, S.GUESTS_EATING, S.NEEDS_CLEANING]})
    labels = pd.DataFrame(
        {
            "timestamp_s": [0.0, 30.0, 60.0],
            "table_id": ["T1", "T1", "T1"],
            "true_status": ["empty_clean", "guests_waiting", None],
        }
    )
    comparison = compare_with_labels(run, labels)
    assert len(comparison) == 2
    assert accuracy(comparison) == 0.5
    assert confusion_matrix(comparison).loc["guests_waiting", "guests_eating"] == 1


def test_compare_handles_freshly_generated_template(tmp_path):
    run = make_run({"T1": [S.EMPTY_CLEAN] * 3})
    path = tmp_path / "labels.csv"
    write_label_template(run, path, sample_frames=3)
    assert accuracy(compare_with_labels(run, pd.read_csv(path))) is None


def test_compare_with_partly_filled_csv(tmp_path):
    run = make_run({"T1": [S.EMPTY_CLEAN, S.NEEDS_CLEANING]})
    path = tmp_path / "labels.csv"
    path.write_text("timestamp_s,table_id,true_status\n0.0,T1, empty_clean \n30.0,T1,\n", encoding="utf-8")
    comparison = compare_with_labels(run, pd.read_csv(path))
    assert len(comparison) == 1
    assert accuracy(comparison) == 1.0


def test_accuracy_is_none_without_labels():
    run = make_run({"T1": [S.EMPTY_CLEAN]})
    labels = pd.DataFrame({"timestamp_s": [0.0], "table_id": ["T1"], "true_status": [" "]})
    assert accuracy(compare_with_labels(run, labels)) is None
