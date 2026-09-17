"""
hand_tracker.py
----------------
Thin wrapper around MediaPipe's HandLandmarker (the current Tasks API,
which replaced the older `mp.solutions.hands` API) that detects a hand in
a BGR frame (as produced by OpenCV) and returns its 21 landmarks.

Kept separate from the rest of the pipeline so the detection backend could
be swapped out later without touching feature extraction, training, or the
CLI commands.

The HandLandmarker needs a small pre-trained model bundle (hand_landmarker
.task, ~7-13 MB) that MediaPipe does not bundle with the pip package. It is
downloaded automatically to models/hand_landmarker.task the first time this
module runs, if it is not already present. This requires an internet
connection once; after that, everything runs fully offline.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Standard MediaPipe 21-point hand skeleton topology, used only for drawing.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


@dataclass
class HandResult:
    """Landmarks for a single detected hand."""
    landmarks: np.ndarray       # shape (21, 3) -> x, y, z (normalized coords)
    handedness: str             # "Left" or "Right"
    score: float                # detection confidence


def ensure_model_downloaded(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Downloads the HandLandmarker model bundle to `model_path` if it isn't
    already there. Raises a clear error with manual-download instructions
    if the download fails (e.g. no internet access).
    """
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[hand_tracker] Downloading hand landmark model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # noqa: BLE001 - re-raise with actionable message
        raise RuntimeError(
            "Could not download the required MediaPipe model file "
            f"({MODEL_URL}). Check your internet connection, or download it "
            f"manually and place it at: {os.path.abspath(model_path)}"
        ) from exc

    print("[hand_tracker] Model downloaded.")
    return model_path


class HandTracker:
    """
    Detects a single hand per frame and exposes its landmarks.

    Usage:
        tracker = HandTracker()
        result = tracker.process(frame_bgr)
        if result is not None:
            print(result.landmarks.shape)  # (21, 3)
        tracker.close()
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        model_path = ensure_model_downloaded(model_path)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        """
        Run detection on a single BGR frame.
        Returns the highest-confidence HandResult, or None if no hand found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Take the first (and, by default config, only) detected hand.
        hand_landmarks = result.hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )

        handedness_label = "Unknown"
        score = 0.0
        if result.handedness:
            category = result.handedness[0][0]
            handedness_label = category.category_name
            score = category.score

        return HandResult(landmarks=coords, handedness=handedness_label, score=score)

    def draw(self, frame_bgr: np.ndarray, result: HandResult) -> np.ndarray:
        """Draw the hand skeleton on a copy of the frame for visualization."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        points: List[Tuple[int, int]] = [
            (int(x * w), int(y * h)) for x, y, _ in result.landmarks
        ]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated, points[start_idx], points[end_idx], (0, 200, 0), 2)

        for point in points:
            cv2.circle(annotated, point, 4, (0, 120, 255), -1)

        return annotated

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
