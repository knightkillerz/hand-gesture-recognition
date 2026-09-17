# Project Report: Hand Gesture Recognition

> **Before submitting**: replace this template with your course's exact
> required report format (check the instruction document / course page).
> The sections below are a generic structure common to ML/CV mini-project
> reports — reuse the content, rearrange to match your required format.

---

## 1. Abstract
A 150–200 word summary: what the project does, the approach (landmark
detection + machine learning classification), and the key result
(e.g., test accuracy achieved).

## 2. Introduction
- Problem statement: why gesture recognition is useful (HCI, accessibility,
  touchless control, sign language groundwork, etc.)
- Objective: detect a hand in a live video feed and classify it into one
  of a fixed set of gestures.
- Scope: single-hand, six gesture classes, real-time on CPU.

## 3. Literature Survey / Related Work
Briefly describe 2–4 existing approaches to hand gesture recognition
(e.g., CNN-on-image approaches, sensor-glove based systems, skeleton/
landmark-based approaches like this project) and how they compare in terms
of accuracy, speed, and data requirements.

## 4. System Architecture
Include the pipeline diagram from the README:

```
Webcam -> MediaPipe HandLandmarker (21 keypoints) -> Feature Normalization
       -> [ML Classifier | Rule-based Classifier] -> Predicted Gesture
```

Explain each stage briefly.

## 5. Methodology

### 5.1 Hand Landmark Detection
MediaPipe's HandLandmarker (Tasks API) is used to detect 21 3D keypoints
per hand per frame.

### 5.2 Feature Engineering
Landmarks are:
1. Translated so the wrist is the origin (position-invariant).
2. Scaled by the wrist-to-middle-MCP distance (distance-from-camera
   invariant).
3. Augmented with 5 finger-extension flags and 5 fingertip-to-wrist
   distances (hand-crafted geometric features).

This produces a 73-dimensional feature vector per frame.

### 5.3 Classification
Two classifiers are implemented and compared:
- **Random Forest** (scikit-learn), trained on your recorded dataset.
- **Rule-based heuristic**, using finger-extension patterns directly
  (no training data required).

### 5.4 Dataset
- Number of gesture classes: _fill in_
- Samples per class: _fill in_
- Total samples collected: _fill in_
- Collection method: recorded via `python cli.py collect` from a laptop
  webcam under varied hand position/distance/angle.

## 6. Implementation Details
- Language/libraries: Python 3.12, OpenCV, MediaPipe, scikit-learn, pandas,
  NumPy, joblib.
- Repository structure and how to run it: see README.md.
- Key design decisions: landmark-based features (not raw pixels/CNN) for a
  lightweight, CPU-trainable, explainable model.

## 7. Results

### 7.1 Quantitative Results
Paste the accuracy, classification report, and confusion matrix printed by
`python cli.py train` here. Example table to fill in:

| Gesture | Precision | Recall | F1-score |
|---|---|---|---|
| Open_Palm | | | |
| Fist | | | |
| Thumbs_Up | | | |
| Peace | | | |
| Pointing | | | |
| OK | | | |

Overall test accuracy: **____%**

### 7.2 ML vs. Rule-based Comparison
Run `python cli.py recognize` and note cases where the ML model and the
rule-based baseline agree/disagree. Discuss why (e.g., rule-based struggles
with ambiguous in-between hand poses; ML model generalizes better after
training on real examples but needs labeled data).

### 7.3 Qualitative Observations
- Lighting/background conditions that affected detection.
- Any gestures frequently confused with each other, and why.

## 8. Limitations
- Single-hand only (current implementation).
- Accuracy depends on the diversity of your own recorded training data.
- Static-pose gestures only (no temporal/dynamic gestures like waving).

## 9. Future Work
- Add dynamic/temporal gesture recognition (e.g., using an LSTM over
  landmark sequences).
- Support two-hand gestures.
- Package as a small GUI/desktop app for accessibility use cases.

## 10. Conclusion
Summarize what was built, the accuracy achieved, and what it demonstrates
about landmark-based gesture recognition as a lightweight alternative to
full image-based deep learning approaches.

## 11. References
List MediaPipe documentation, scikit-learn documentation, and any papers
consulted, in your course's required citation format.
