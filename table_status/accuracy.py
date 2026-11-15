import random
from pathlib import Path

import pandas as pd

from table_status.models import RunResult

LABEL_COLUMNS = ["timestamp_s", "table_id", "image", "model_status", "true_status"]


def write_label_template(run: RunResult, path: Path, sample_frames: int = 20, seed: int = 7) -> int:
    """Write a CSV of sampled frames so a person can fill in true_status by looking at the images."""
    frames = sorted(run.frames, key=lambda frame: frame.timestamp_s)
    sampled = sorted(random.Random(seed).sample(frames, min(sample_frames, len(frames))), key=lambda f: f.timestamp_s)
    rows = [
        {
            "timestamp_s": frame.timestamp_s,
            "table_id": reading.table_id,
            "image": frame.image_path,
            "model_status": reading.status.value,
            "true_status": "",
        }
        for frame in sampled
        for reading in frame.readings
    ]
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(path, index=False)
    return len(rows)


def compare_with_labels(run: RunResult, labels: pd.DataFrame) -> pd.DataFrame:
    """Join human labels with the model's current predictions; rows without a label are dropped."""
    labelled = labels.dropna(subset=["true_status"]).copy()
    # An unfilled CSV column is read as float NaN, so force text before comparing.
    labelled["true_status"] = labelled["true_status"].astype(str).str.strip()
    labelled = labelled[labelled["true_status"] != ""]
    predictions = pd.DataFrame(
        [
            {"timestamp_s": frame.timestamp_s, "table_id": reading.table_id, "predicted": reading.status.value}
            for frame in run.frames
            for reading in frame.readings
        ]
    )
    merged = labelled[["timestamp_s", "table_id", "true_status"]].merge(
        predictions, on=["timestamp_s", "table_id"], how="inner"
    )
    merged["correct"] = merged["true_status"] == merged["predicted"]
    return merged


def accuracy(comparison: pd.DataFrame) -> float | None:
    if comparison.empty:
        return None
    return float(comparison["correct"].mean())


def confusion_matrix(comparison: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(comparison["true_status"], comparison["predicted"], rownames=["human"], colnames=["model"])
