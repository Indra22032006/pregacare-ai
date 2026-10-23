"""Maternal-health risk classifier based on the referenced Kaggle dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


FEATURE_NAMES = (
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
)
LABELS = ("low risk", "mid risk", "high risk")
DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "maternal_health"
    / "Maternal Health Risk Data Set.csv"
)


def _load_dataset() -> tuple[list[list[float]], list[str]]:
    with DATASET_PATH.open(newline="", encoding="utf-8-sig") as dataset_file:
        rows = list(csv.DictReader(dataset_file))
    features = [
        [float(row[name].strip()) for name in FEATURE_NAMES]
        for row in rows
    ]
    labels = [row["RiskLevel"].strip().lower() for row in rows]
    return features, labels


def _train() -> tuple[RandomForestClassifier, dict[str, Any]]:
    features, labels = _load_dataset()
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    classifier = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    report = classification_report(
        test_labels, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return classifier, {
        "rows": len(features),
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 3),
        "per_class_recall": {
            label: round(float(report[label]["recall"]), 3) for label in LABELS
        },
    }


MODEL, METRICS = _train()


def predict_risk(values: dict[str, float]) -> dict[str, Any]:
    feature_values = [float(values[name]) for name in FEATURE_NAMES]
    probabilities = MODEL.predict_proba([feature_values])[0]
    best_index = int(probabilities.argmax())
    risk_level = str(MODEL.classes_[best_index])
    probability = float(probabilities[best_index])
    return {
        "risk_level": risk_level,
        "risk_probability": round(probability, 3),
        "requires_medical_attention": risk_level == "high risk",
        "could_require_medical_attention": risk_level == "mid risk",
        "recommendation": {
            "high risk": "Seek prompt review from a qualified healthcare professional.",
            "mid risk": "Arrange a healthcare review; monitor symptoms and readings.",
            "low risk": "Continue routine prenatal care and monitoring.",
        }[risk_level],
        "disclaimer": (
            "This is a research prototype trained on a public dataset. It does not "
            "diagnose, rule out danger, or replace a clinician or emergency care."
        ),
    }


def model_info() -> dict[str, Any]:
    return {
        "model": "maternal-health-random-forest-v1",
        "features": list(FEATURE_NAMES),
        "classes": list(LABELS),
        "metrics": METRICS,
        "dataset": "Maternal Health Risk Data Set.csv",
        "source": "https://www.kaggle.com/code/yasserhessein/classification-maternal-health-5-algorithms-ml",
    }