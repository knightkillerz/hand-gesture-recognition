"""
recognize.py
-------------
Real-time (or video-file) gesture recognition. Loads the trained ML model
and, optionally, also runs the rule-based baseline side by side so the two
can be compared frame by frame -- handy for the project report.
"""

import os
from typing import Optional

import cv2
import joblib
import numpy as np

from .features import extract_feature_vector
from .hand_tracker import HandTracker
from .rule_based import classify as rule_based_classify
from .train_model import DEFAULT_ENCODER_PATH, DEFAULT_MODEL_PATH


def run(
    model_path: str = DEFAULT_MODEL_PATH,
    encoder_path: str = DEFAULT_ENCODER_PATH,
    source: Optional[str] = None,
    show_window: bool = True,
    compare_rule_based: bool = True,
    confidence_threshold: float = 0.5,
    max_frames: Optional[int] = None,
) -> None:
    if not os.path.isfile(model_path) or not os.path.isfile(encoder_path):
        raise FileNotFoundError(
            "Trained model not found. Run the 'train' command first "
            f"(expected {model_path} and {encoder_path})."
        )

    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    cap_source = 0 if source is None else source
    if isinstance(cap_source, str) and cap_source.isdigit():
        cap_source = int(cap_source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source {cap_source!r}. "
            "If you're running headless (no webcam), pass a video file with --source."
        )

    frame_count = 0
    with HandTracker(max_num_hands=1) as tracker:
        print("[recognize] Running. Press 'q' to quit." if show_window else
              "[recognize] Running in headless mode.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[recognize] End of video source reached.")
                break

            frame_count += 1
            result = tracker.process(frame)
            display = frame

            if result is not None:
                features = extract_feature_vector(result.landmarks).reshape(1, -1)
                probs = clf.predict_proba(features)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                ml_label = encoder.classes_[best_idx]

                if confidence < confidence_threshold:
                    ml_label = "Uncertain"

                rb_label = rule_based_classify(result.landmarks) if compare_rule_based else None

                display = tracker.draw(frame, result)
                cv2.putText(
                    display, f"ML: {ml_label} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                if rb_label is not None:
                    cv2.putText(
                        display, f"Rule-based: {rb_label}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2,
                    )
                else:
                    print(f"[recognize] frame={frame_count} ML={ml_label} ({confidence:.2f})")
            else:
                cv2.putText(
                    display, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            if show_window:
                cv2.imshow("Hand Gesture Recognition - press q to quit", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and frame_count >= max_frames:
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
