"""
collect_data.py
-----------------
CLI routine that opens a video source (webcam by default, or a video file
for headless/testing use), detects a hand each frame, extracts its feature
vector, and appends it to a CSV file labeled with the gesture name you
supply. Run this once per gesture class to build your training set.

CSV layout: one row per sample, columns = FEATURE_NAMES + ["label"].
"""

import csv
import os
import time
from typing import Optional

import cv2

from .features import FEATURE_NAMES, extract_feature_vector
from .hand_tracker import HandTracker

DEFAULT_DATA_PATH = os.path.join("data", "gestures.csv")


def _ensure_csv_header(csv_path: str) -> None:
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not file_exists:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES + ["label"])


def collect(
    label: str,
    num_samples: int = 150,
    source: Optional[str] = None,
    csv_path: str = DEFAULT_DATA_PATH,
    capture_every_n_frames: int = 2,
    show_window: bool = True,
) -> int:
    """
    Collect `num_samples` feature vectors for `label`.

    source: None -> default webcam (index 0). Otherwise a path to a video
    file or an integer camera index passed as a string.
    Returns the number of samples actually written.
    """
    cap_source = 0 if source is None else source
    # allow numeric strings like "1" to select camera index 1
    if isinstance(cap_source, str) and cap_source.isdigit():
        cap_source = int(cap_source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source {cap_source!r}. "
            "If you're running headless (no webcam), pass a video file with --source."
        )

    _ensure_csv_header(csv_path)

    written = 0
    frame_idx = 0
    with HandTracker(max_num_hands=1) as tracker, open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        print(f"[collect] Recording label='{label}' -> target {num_samples} samples. "
              f"Press 'q' to stop early.")

        while written < num_samples:
            ok, frame = cap.read()
            if not ok:
                print("[collect] End of video source reached.")
                break

            frame_idx += 1
            result = tracker.process(frame)

            display = frame
            if result is not None and frame_idx % capture_every_n_frames == 0:
                features = extract_feature_vector(result.landmarks)
                writer.writerow(list(features) + [label])
                written += 1
                display = tracker.draw(frame, result)

            if show_window:
                cv2.putText(
                    display,
                    f"{label}: {written}/{num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Data Collection - press q to stop", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # small delay to avoid a tight, uncontrolled capture loop
                time.sleep(0.01)

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[collect] Wrote {written} samples for label='{label}' to {csv_path}")
    return written
