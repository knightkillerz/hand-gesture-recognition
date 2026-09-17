#!/usr/bin/env python3
"""
cli.py
-------
Single command-line entry point for the whole project.

    python cli.py collect   --label Fist --samples 150
    python cli.py train
    python cli.py recognize
    python cli.py demo      (rule-based only, no trained model needed)

Run `python cli.py <command> --help` for the full list of options for each
command.
"""

import argparse
import sys

from src.collect_data import DEFAULT_DATA_PATH, collect
from src.recognize import run as recognize_run
from src.train_model import DEFAULT_DATA_PATH as TRAIN_DEFAULT_DATA_PATH
from src.train_model import DEFAULT_ENCODER_PATH, DEFAULT_MODEL_PATH, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Hand Gesture Recognition - collect data, train, and run recognition.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- collect ---
    collect_p = subparsers.add_parser("collect", help="Record labeled gesture samples.")
    collect_p.add_argument("--label", required=True, help="Gesture name, e.g. Fist, Open_Palm, Peace.")
    collect_p.add_argument("--samples", type=int, default=150, help="Number of samples to record.")
    collect_p.add_argument("--source", default=None, help="Camera index or video file path. Default: webcam 0.")
    collect_p.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="CSV file to append samples to.")
    collect_p.add_argument("--no-window", action="store_true", help="Run without opening a display window.")

    # --- train ---
    train_p = subparsers.add_parser("train", help="Train the ML classifier on collected data.")
    train_p.add_argument("--data-path", default=TRAIN_DEFAULT_DATA_PATH, help="CSV dataset to train on.")
    train_p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    train_p.add_argument("--encoder-path", default=DEFAULT_ENCODER_PATH, help="Where to save the label encoder.")
    train_p.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing.")
    train_p.add_argument("--trees", type=int, default=200, help="Number of trees in the Random Forest.")

    # --- recognize ---
    recognize_p = subparsers.add_parser("recognize", help="Run real-time gesture recognition.")
    recognize_p.add_argument("--source", default=None, help="Camera index or video file path. Default: webcam 0.")
    recognize_p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Trained model to load.")
    recognize_p.add_argument("--encoder-path", default=DEFAULT_ENCODER_PATH, help="Label encoder to load.")
    recognize_p.add_argument("--no-window", action="store_true", help="Run without opening a display window.")
    recognize_p.add_argument("--no-rule-based", action="store_true", help="Hide the rule-based comparison label.")
    recognize_p.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence to show a label.")
    recognize_p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing).")

    # --- demo (rule-based only, needs no training) ---
    demo_p = subparsers.add_parser("demo", help="Run the rule-based classifier only (no trained model required).")
    demo_p.add_argument("--source", default=None, help="Camera index or video file path. Default: webcam 0.")
    demo_p.add_argument("--no-window", action="store_true", help="Run without opening a display window.")
    demo_p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing).")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        collect(
            label=args.label,
            num_samples=args.samples,
            source=args.source,
            csv_path=args.data_path,
            show_window=not args.no_window,
        )

    elif args.command == "train":
        train(
            csv_path=args.data_path,
            model_path=args.model_path,
            encoder_path=args.encoder_path,
            test_size=args.test_size,
            n_estimators=args.trees,
        )

    elif args.command == "recognize":
        recognize_run(
            model_path=args.model_path,
            encoder_path=args.encoder_path,
            source=args.source,
            show_window=not args.no_window,
            compare_rule_based=not args.no_rule_based,
            confidence_threshold=args.confidence,
            max_frames=args.max_frames,
        )

    elif args.command == "demo":
        _run_rule_based_demo(source=args.source, show_window=not args.no_window, max_frames=args.max_frames)

    return 0


def _run_rule_based_demo(source, show_window: bool, max_frames) -> None:
    import cv2

    from src.hand_tracker import HandTracker
    from src.rule_based import classify

    cap_source = 0 if source is None else source
    if isinstance(cap_source, str) and cap_source.isdigit():
        cap_source = int(cap_source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {cap_source!r}.")

    frame_count = 0
    with HandTracker(max_num_hands=1) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            result = tracker.process(frame)
            display = frame

            if result is not None:
                label = classify(result.landmarks)
                display = tracker.draw(frame, result)
                cv2.putText(display, f"Rule-based: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)
                print(f"[demo] frame={frame_count} label={label}")

            if show_window:
                cv2.imshow("Rule-based Demo - press q to quit", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and frame_count >= max_frames:
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
