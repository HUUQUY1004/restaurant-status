# Restaurant table status from camera footage

A small proof of concept: take recorded security-camera footage, ask a vision model what is happening at each table every N seconds, and turn those snapshots into a per-table timeline, a few service metrics, and an honest accuracy check against human labels.

## How it works

1. **Zones**: tables are outlined once per camera (`data/zones.json`). Gemini can propose the boxes from one frame (`detect-zones`); edit them by hand if needed.
2. **Sampling**: one frame every 30 s (configurable) is taken from the video.
3. **Classification**: the frame, with numbered table boxes drawn on it, goes to Gemini on Vertex AI, which returns structured JSON per table:
   `empty_clean | guests_waiting | guests_eating | needs_attention | needs_cleaning | unknown`, plus a confidence and a short note.
4. **Smoothing**: a status only counts once it holds for N consecutive frames, so a single wrong frame does not create a fake event.
5. **Metrics**: per table, the number of visits, the average visit length, the average wait before a table is cleaned, and the number of times guests needed attention.
6. **Accuracy check**: about 20 frames are labelled by hand and compared with the model, which shows the accuracy score and a confusion matrix, so you know how far to trust the numbers before going live.

Results are saved after every frame, so re-running never pays for the same frame twice.

### Keeping cost and rate limits under control

- **Blank frames** (fades, lost signal) are never sent; the model would invent guests on a black image.
- **Change detection**: each table's crop is compared with the last time the model checked it. Only tables that changed by more than `--change-threshold` (mean pixel change, default 10) are asked about, and every table is re-checked at least every `--max-skip` seconds. When nothing changed, no call is made.
- **Retry with backoff** on 429/500/503 (1s, 2s, 4s... plus jitter), so a rate limit slows the run down instead of stopping it.
- Calls still send the full frame. The next cost step is sending only crops of the changed tables, or putting a local person detector in front of the model.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # set GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
```

The Vertex AI API must be enabled on the project.

## Usage

```bash
python -m table_status.cli detect-zones data/restaurant.mp4            # -> data/zones.json + preview png
python -m table_status.cli analyze data/restaurant.mp4 --run-dir runs/demo --every 30
python -m table_status.cli label-template --run-dir runs/demo          # fill true_status in runs/demo/labels.csv
streamlit run app.py
```

## Project layout

| Module | Responsibility |
| --- | --- |
| `frames.py` | read frames from a video at a fixed interval |
| `zones.py` | load/save table zones, draw them on frames |
| `gemini.py` | Gemini classifier and table detector (Vertex AI) |
| `pipeline.py` | run the classifier over a video, resumable |
| `timeline.py` | smoothing and per-table status segments |
| `metrics.py` | visits, cleaning wait, attention events |
| `accuracy.py` | label template and model-vs-human comparison |
| `app.py` | Streamlit dashboard |

## Next steps toward production

- Read live RTSP streams from the NVR instead of recorded files.
- Use a small on-site detector (e.g. YOLO) for people and tables, and call the LLM only for ambiguous frames, which lowers cost and latency.
- Send alerts (e.g. WhatsApp or Slack) when a table needs attention or cleaning for more than X minutes.
- Address privacy before going live: blur faces before sending frames out, and set a short retention period.

## Tests

```bash
pytest
```
