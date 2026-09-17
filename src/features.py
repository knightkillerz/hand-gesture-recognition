"""
features.py
------------
Turns raw MediaPipe hand landmarks (21 x 3, in normalized image coordinates)
into a fixed-length feature vector that is translation- and scale-invariant,
so the classifier learns the *shape* of the gesture rather than where the
hand happens to be in the frame or how close it is to the camera.

Two kinds of features are produced:
1. Normalized landmark coordinates (wrist-centered, scale-normalized).
2. A handful of hand-crafted geometric features (finger extension flags,
   fingertip-to-wrist distances) that also back the rule-based baseline
   classifier in rule_based.py.
"""

from typing import List

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FEATURE_NAMES: List[str] = (
    [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
    + ["thumb_ext", "index_ext", "middle_ext", "ring_ext", "pinky_ext"]
    + [
        "thumb_dist",
        "index_dist",
        "middle_dist",
        "ring_dist",
        "pinky_dist",
    ]
)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate landmarks so the wrist is the origin, then scale so the
    distance from the wrist to the middle-finger MCP joint is 1.0.
    This makes the representation invariant to hand position and distance
    from the camera.
    """
    centered = landmarks - landmarks[WRIST]
    scale = np.linalg.norm(centered[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    return centered / scale


def _finger_extended(landmarks: np.ndarray, tip: int, pip: int, mcp: int) -> float:
    """
    Returns 1.0 if a finger is extended, 0.0 otherwise, using the heuristic
    that an extended finger's tip is farther from the wrist than its own
    PIP/MCP joints are.
    """
    wrist = landmarks[WRIST]
    tip_dist = np.linalg.norm(landmarks[tip] - wrist)
    mcp_dist = np.linalg.norm(landmarks[mcp] - wrist)
    return 1.0 if tip_dist > mcp_dist * 1.15 else 0.0


def geometric_features(landmarks: np.ndarray) -> np.ndarray:
    """Hand-crafted features: per-finger extension flags + fingertip distances."""
    ext = np.array(
        [
            _finger_extended(landmarks, THUMB_TIP, THUMB_IP, THUMB_MCP),
            _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
            _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            _finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
            _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ],
        dtype=np.float32,
    )

    wrist = landmarks[WRIST]
    dists = np.array(
        [np.linalg.norm(landmarks[t] - wrist) for t in FINGERTIPS],
        dtype=np.float32,
    )

    return np.concatenate([ext, dists])


def extract_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """
    Full feature vector fed to the ML classifier:
    63 normalized landmark coordinates + 5 extension flags + 5 fingertip
    distances = 73 features total.
    """
    normalized = normalize_landmarks(landmarks)
    geo = geometric_features(normalized)
    return np.concatenate([normalized.flatten(), geo]).astype(np.float32)
