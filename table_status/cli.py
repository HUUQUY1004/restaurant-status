import argparse
from pathlib import Path

import cv2

from table_status.accuracy import write_label_template
from table_status.change import ChangeDetector
from table_status.frames import encode_jpeg, read_frame_at
from table_status.gemini import GeminiTableClassifier, GeminiTableDetector, make_client, model_name
from table_status.models import FrameResult
from table_status.pipeline import analyze_video, load_run
from table_status.usage import model_usage
from table_status.zones import draw_zones, load_zones, save_zones, zones_from_boxes


def detect_zones(args: argparse.Namespace) -> None:
    frame = read_frame_at(args.video, args.at)
    detector = GeminiTableDetector(make_client(), model_name())
    height, width = frame.shape[:2]
    zones = zones_from_boxes(detector.detect_boxes(encode_jpeg(frame)), width, height)
    save_zones(args.out, zones)
    preview_path = args.out.with_suffix(".png")
    cv2.imwrite(str(preview_path), draw_zones(frame, zones))
    print(f"Found {len(zones)} tables -> {args.out} (check {preview_path} and edit boxes if needed)")


def analyze(args: argparse.Namespace) -> None:
    zones = load_zones(args.zones)
    classifier = GeminiTableClassifier(make_client(), model_name())
    change_detector = (
        ChangeDetector(zones, threshold=args.change_threshold, max_age_s=args.max_skip)
        if args.change_threshold > 0
        else None
    )
    run = analyze_video(
        video_path=args.video,
        run_dir=args.run_dir,
        zones=zones,
        classifier=classifier,
        interval_s=args.every,
        change_detector=change_detector,
        on_progress=print_progress,
    )
    usage = model_usage(run)
    print(
        f"Done: {usage.frames} frames, {usage.model_calls} model calls ({usage.calls_saved_ratio:.0%} saved), "
        f"{usage.table_checks}/{usage.naive_table_checks} table checks ({usage.checks_saved_ratio:.0%} fewer)"
    )


def print_progress(frame: FrameResult) -> None:
    checked = ", ".join(frame.checked_tables) or "none (unchanged or blank)"
    print(f"  {frame.timestamp_s:6.1f}s  asked model about: {checked}")


def label_template(args: argparse.Namespace) -> None:
    run = load_run(args.run_dir)
    if run is None:
        raise SystemExit(f"No results in {args.run_dir}, run 'analyze' first")
    out = args.run_dir / "labels.csv"
    rows = write_label_template(run, out, sample_frames=args.frames)
    print(f"Wrote {rows} rows to {out}; fill in true_status by looking at {args.run_dir / 'frames'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="table_status")
    commands = parser.add_subparsers(required=True)

    detect = commands.add_parser("detect-zones", help="find table boxes in one frame with Gemini")
    detect.add_argument("video", type=Path)
    detect.add_argument("--at", type=float, default=0.0, help="timestamp in seconds")
    detect.add_argument("--out", type=Path, default=Path("data/zones.json"))
    detect.set_defaults(handler=detect_zones)

    run = commands.add_parser("analyze", help="classify every table every N seconds")
    run.add_argument("video", type=Path)
    run.add_argument("--zones", type=Path, default=Path("data/zones.json"))
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--every", type=float, default=30.0, help="seconds between frames")
    run.add_argument(
        "--change-threshold", type=float, default=10.0, help="pixel change (0-255) that triggers a re-check; 0 = always ask"
    )
    run.add_argument("--max-skip", type=float, default=120.0, help="re-check a table at least this often (seconds)")
    run.set_defaults(handler=analyze)

    labels = commands.add_parser("label-template", help="create labels.csv for the accuracy check")
    labels.add_argument("--run-dir", type=Path, required=True)
    labels.add_argument("--frames", type=int, default=20)
    labels.set_defaults(handler=label_template)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
