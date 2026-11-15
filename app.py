from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from table_status.accuracy import accuracy, compare_with_labels, confusion_matrix
from table_status.display import STATUS_HEX, STATUS_LABELS, status_overlay
from table_status.metrics import all_table_metrics
from table_status.models import TableStatus
from table_status.pipeline import load_run
from table_status.timeline import build_segments
from table_status.usage import model_usage

RUNS_DIR = Path("runs")

st.set_page_config(page_title="Table Status Demo", layout="wide")
st.title("Restaurant table status from camera footage")

run_dirs = sorted(path for path in RUNS_DIR.glob("*") if (path / "results.json").exists())
if not run_dirs:
    st.info("No runs yet. Run `python -m table_status.cli analyze <video> --run-dir runs/<name>` first.")
    st.stop()

run_dir = st.sidebar.selectbox("Run", run_dirs, format_func=lambda path: path.name)
min_run = st.sidebar.slider("Frames a status must hold before it counts", 1, 4, 2)
run = load_run(run_dir)
segments = build_segments(run, min_run=min_run)
st.sidebar.caption(f"{len(run.frames)} frames, one every {run.interval_s:g}s, {len(run.zones)} tables")
usage = model_usage(run)
st.sidebar.metric("Model calls saved", f"{usage.calls_saved_ratio:.0%}", help="Compared with sending every frame")
st.sidebar.caption(
    f"{usage.model_calls} calls for {usage.frames} frames. The model was asked about "
    f"{usage.table_checks} of {usage.naive_table_checks} table snapshots; blank frames and "
    "unchanged tables are skipped."
)

live_tab, timeline_tab, accuracy_tab = st.tabs(["Live view", "Timeline & metrics", "Accuracy check"])


def to_clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


with live_tab:
    last_index = len(run.frames) - 1
    # Open on the first frame the model could read, not a black intro frame.
    first_readable = next(
        (i for i, f in enumerate(run.frames) if any(r.status != TableStatus.UNKNOWN for r in f.readings)), 0
    )
    frame_index = st.slider("Frame", 0, last_index, first_readable) if last_index > 0 else 0
    frame = run.frames[frame_index]
    image_col, detail_col = st.columns([3, 2])
    image_col.image(status_overlay(run_dir, frame, run.zones), caption=f"t = {to_clock(frame.timestamp_s)}")
    detail_col.dataframe(
        pd.DataFrame(
            [
                {
                    "Table": r.table_id,
                    "Status": STATUS_LABELS[r.status],
                    "Confidence": round(r.confidence, 2),
                    "Checked this frame": "yes" if r.table_id in frame.checked_tables else "skipped",
                    "What the model saw": r.note,
                }
                for r in frame.readings
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with timeline_tab:
    base = pd.Timestamp("2026-01-01")
    chart_data = pd.DataFrame(
        [
            {
                "Table": s.table_id,
                "Status": STATUS_LABELS[s.status],
                "Start": base + pd.Timedelta(seconds=s.start_s),
                "End": base + pd.Timedelta(seconds=s.end_s),
            }
            for s in segments
        ]
    )
    figure = px.timeline(
        chart_data,
        x_start="Start",
        x_end="End",
        y="Table",
        color="Status",
        color_discrete_map={STATUS_LABELS[k]: v for k, v in STATUS_HEX.items()},
    )
    figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(run.table_ids)))
    figure.update_xaxes(tickformat="%M:%S", title="Video time")
    st.plotly_chart(figure, width="stretch")

    metrics = all_table_metrics(run.table_ids, segments)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Table": m.table_id,
                    "Visits": m.visits,
                    "Avg visit (mm:ss)": to_clock(m.avg_visit_s),
                    "Avg wait to clean (mm:ss)": to_clock(m.avg_cleaning_wait_s),
                    "Attention events": m.attention_events,
                    "Longest attention wait (mm:ss)": to_clock(m.longest_attention_s),
                }
                for m in metrics
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Visits cut off by the start or end of the video are counted as partial.")

with accuracy_tab:
    labels_path = run_dir / "labels.csv"
    if not labels_path.exists():
        st.info(f"Run `python -m table_status.cli label-template --run-dir {run_dir}` and fill in true_status.")
    else:
        comparison = compare_with_labels(run, pd.read_csv(labels_path))
        score = accuracy(comparison)
        if score is None:
            st.warning("labels.csv has no true_status filled in yet.")
        else:
            st.metric("Model agrees with human labels", f"{score:.0%}", help=f"{len(comparison)} table checks")
            st.dataframe(confusion_matrix(comparison), width="stretch")
            st.dataframe(comparison[~comparison["correct"]], hide_index=True, width="stretch")
