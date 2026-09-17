"""
train_model.py
----------------
Loads collected gesture samples from a CSV file, trains a Random Forest
classifier on the feature vectors, evaluates it on a held-out test split,
and saves the trained model + label encoder to disk with joblib.
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .features import FEATURE_NAMES

DEFAULT_DATA_PATH = os.path.join("data", "gestures.csv")
DEFAULT_MODEL_PATH = os.path.join("models", "gesture_model.joblib")
DEFAULT_ENCODER_PATH = os.path.join("models", "label_encoder.joblib")


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No dataset found at {csv_path}. Run the 'collect' command first."
        )
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected feature columns: {missing}")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return X, y


def train(
    csv_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    encoder_path: str = DEFAULT_ENCODER_PATH,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    X, y_raw = load_dataset(csv_path)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    class_counts = pd.Series(y_raw).value_counts()
    print("[train] Class distribution:")
    print(class_counts.to_string())

    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[train] Test accuracy: {acc:.4f}\n")
    print("[train] Classification report:")
    print(report)
    print("[train] Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_))

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"\n[train] Saved model to {model_path}")
    print(f"[train] Saved label encoder to {encoder_path}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "classes": list(encoder.classes_),
    }
