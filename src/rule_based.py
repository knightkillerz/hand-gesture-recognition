"""
rule_based.py
--------------
A hand-crafted, no-training-required gesture classifier based purely on
which fingers are extended. Used as a baseline to compare against the
trained ML model (see report/ for the comparison).

Supported gestures: Open_Palm, Fist, Thumbs_Up, Peace, Pointing, OK.
Anything that doesn't match a known pattern is labeled "Unknown".
"""

from typing import Tuple

import numpy as np

from .features import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    _finger_extended,
    normalize_landmarks,
)


def _finger_states(landmarks: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Returns a 5-tuple of 0/1 flags for (thumb, index, middle, ring, pinky)."""
    norm = normalize_landmarks(landmarks)
    thumb = int(_finger_extended(norm, THUMB_TIP, THUMB_IP, THUMB_MCP))
    index = int(_finger_extended(norm, INDEX_TIP, INDEX_PIP, INDEX_MCP))
    middle = int(_finger_extended(norm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP))
    ring = int(_finger_extended(norm, RING_TIP, RING_PIP, RING_MCP))
    pinky = int(_finger_extended(norm, PINKY_TIP, PINKY_PIP, PINKY_MCP))
    return thumb, index, middle, ring, pinky


def _thumb_index_pinch(landmarks: np.ndarray, threshold: float = 0.35) -> bool:
    """True if thumb tip and index tip are close together (used for 'OK')."""
    norm = normalize_landmarks(landmarks)
    dist = np.linalg.norm(norm[THUMB_TIP] - norm[INDEX_TIP])
    return dist < threshold


def classify(landmarks: np.ndarray) -> str:
    """Rule-based gesture classification from raw (unnormalized) landmarks."""
    thumb, index, middle, ring, pinky = _finger_states(landmarks)
    fingers = (thumb, index, middle, ring, pinky)

    if _thumb_index_pinch(landmarks) and middle == 1 and ring == 1 and pinky == 1:
        return "OK"
    if fingers == (1, 1, 1, 1, 1):
        return "Open_Palm"
    if fingers == (0, 0, 0, 0, 0):
        return "Fist"
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "Thumbs_Up"
    if index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return "Peace"
    if index == 1 and middle == 0 and ring == 0 and pinky == 0 and thumb == 0:
        return "Pointing"

    return "Unknown"
