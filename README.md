# Hand Gesture Recognition

A computer-vision project that detects a hand in a webcam feed, extracts its
21 landmark keypoints with MediaPipe, and classifies which of six gestures
it is making using a machine-learning classifier trained on your own
recorded data.

Two classifiers are included so their behavior can be compared:
- **ML classifier** — a Random Forest trained on landmark-derived features
  you record yourself (`src/train_model.py`).
- **Rule-based baseline** — a hand-crafted heuristic based on which fingers
  are extended, requiring no training data (`src/rule_based.py`).

Supported gestures out of the box: `Open_Palm`, `Fist`, `Thumbs_Up`,
`Peace`, `Pointing`, `OK`. You can add more gestures by recording data for
new labels — nothing in the code is hard-coded to these six.

---

## 1. How it works (pipeline overview)

```
webcam frame (OpenCV)
      |
      v
MediaPipe HandLandmarker  --> 21 (x, y, z) hand keypoints
      |
      v
features.py --> normalize (wrist-centered, scale-invariant)
                 + hand-crafted geometric features
                 = 73-dimensional feature vector
      |
      v
        +----------------------------+
        |                            |
   ML classifier              Rule-based classifier
 (Random Forest,               (finger-extension
  trained on your data)         heuristics, no training)
        |                            |
        +-------------+--------------+
                       v
              predicted gesture label
```

- `src/hand_tracker.py` — wraps MediaPipe's HandLandmarker task to detect a
  hand and return its landmarks.
- `src/features.py` — converts raw landmarks into a normalized, fixed-length
  feature vector.
- `src/rule_based.py` — the no-training-needed baseline classifier.
- `src/collect_data.py` — records labeled feature vectors from your webcam
  into a CSV file.
- `src/train_model.py` — trains and evaluates the Random Forest classifier
  on the collected CSV, saves the trained model.
- `src/recognize.py` — runs real-time recognition using the trained model.
- `cli.py` — the single command-line entry point tying everything together.

---

## 2. Requirements

- Python 3.9–3.12
- A webcam (for live collection/recognition) **or** a video file (for
  headless use — see [Running without a webcam](#running-without-a-webcam))
- Internet access the *first* time you run the project, so MediaPipe can
  download its hand-landmark model file (~10 MB, one-time, then cached
  locally in `models/hand_landmarker.task`)

Tested on Python 3.12 with the package versions pinned in
`requirements.txt`.

---

## 3. Installation and setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

No other configuration is needed. The MediaPipe model file is downloaded
automatically the first time you run any command that needs hand
detection (`collect`, `recognize`, `demo`) — you do not need to download it
manually.

---

## 4. Running the project (from the terminal)

All functionality is exposed through `cli.py`. Run every command from the
repository root.

### Step 1 — Collect training data for each gesture

Run this once per gesture, showing that gesture to your webcam. A live
preview window opens (skeleton overlay + running sample count); press `q`
to stop early.

```bash
python cli.py collect --label Open_Palm --samples 150
python cli.py collect --label Fist       --samples 150
python cli.py collect --label Thumbs_Up  --samples 150
python cli.py collect --label Peace      --samples 150
python cli.py collect --label Pointing   --samples 150
python cli.py collect --label OK         --samples 150
```

Samples are appended to `data/gestures.csv`. Vary your hand's position,
distance from the camera, and slight rotation while recording — this makes
the trained model more robust.

### Step 2 — Train the classifier

```bash
python cli.py train
```

This loads `data/gestures.csv`, splits it into train/test sets, trains a
Random Forest, and prints accuracy, a classification report, and a
confusion matrix to the terminal. The trained model is saved to
`models/gesture_model.joblib` (and the label encoder to
`models/label_encoder.joblib`).

### Step 3 — Run real-time recognition

```bash
python cli.py recognize
```

Opens your webcam and overlays both the ML model's prediction and the
rule-based prediction live, so you can compare them. Press `q` to quit.

### Optional — Rule-based-only demo (no training required)

If you want to see gesture detection working immediately, without
recording any data or training a model:

```bash
python cli.py demo
```

### Full command reference

```bash
python cli.py --help
python cli.py collect --help
python cli.py train --help
python cli.py recognize --help
python cli.py demo --help
```

---

## 5. Running without a webcam

Every command accepts `--source`, which can be a camera index (`0`, `1`, ...)
or a path to a video file. This lets the whole pipeline run headlessly
(e.g., in a server/CI environment with no camera and no display):

```bash
python cli.py collect --label Fist --source path/to/video.mp4 --no-window --samples 100
python cli.py recognize --source path/to/video.mp4 --no-window --max-frames 200
```

`--no-window` disables the OpenCV display window (useful when no GUI/display
is available); predictions are printed to the terminal instead.

---

## 6. Project structure

```
hand-gesture-recognition/
├── cli.py                  # command-line entry point
├── requirements.txt
├── README.md
├── src/
│   ├── hand_tracker.py     # MediaPipe HandLandmarker wrapper
│   ├── features.py         # landmark -> feature vector
│   ├── rule_based.py       # heuristic baseline classifier
│   ├── collect_data.py     # data collection CLI logic
│   ├── train_model.py      # training + evaluation CLI logic
│   └── recognize.py        # real-time recognition CLI logic
├── data/                   # your recorded gesture CSVs (git-ignored)
├── models/                 # trained model + downloaded MediaPipe model (git-ignored)
└── report/                 # project report
```

---

## 7. Notes on design choices

- **Landmarks, not raw pixels**: classifying on 21 normalized keypoints
  (instead of raw image pixels or a CNN over images) keeps the model small,
  fast to train on a laptop CPU, and easy to reason about/explain.
- **Wrist-centered, scale-normalized features**: makes the classifier
  invariant to where the hand is in the frame and how close it is to the
  camera, so it generalizes beyond the exact recording conditions.
- **Two classifiers**: the rule-based baseline gives a zero-training
  reference point to measure how much the trained ML model actually gains
  from learning on real data — see the project report for the comparison.

---

## 8. Troubleshooting

| Problem | Likely fix |
|---|---|
| `Could not open video source` | No webcam detected, or it's in use by another app. Pass `--source <path-to-video>` instead. |
| Model download fails | Check your internet connection, or manually download the file at the URL printed in the error and place it at `models/hand_landmarker.task`. |
| Low recognition accuracy | Record more/varied samples per gesture (Step 1), keep your hand fully in frame, and retrain. |
| `ModuleNotFoundError` | Make sure the virtual environment is activated and `pip install -r requirements.txt` completed successfully. |
