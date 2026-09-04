"""
train_model.py
----------------
Builds the artifact used by LAYER 3: Model Inference Layer.

Trains a RandomForestClassifier on labeled (url, label) pairs, using the
same feature_extraction.extract_feature_vector() that the live inference
service calls, so train-time and serve-time features never drift apart.

Usage:
    python train_model.py --data data/urls.csv --out model/rf_phishing_model.joblib

Output:
    model/rf_phishing_model.joblib   -> trained sklearn Pipeline (scaler + RF)
    model/feature_names.json         -> ordered feature names (for explanations)
    model/metrics.json               -> held-out evaluation metrics
"""

import argparse
import csv
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_feature_vector, FEATURE_NAMES


def load_dataset(path: str):
    urls, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))
    return urls, labels


def build_feature_matrix(urls):
    return np.array([extract_feature_vector(u) for u in urls], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/urls.csv")
    parser.add_argument("--out", default="model/rf_phishing_model.joblib")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    urls, labels = load_dataset(args.data)
    print(f"  {len(urls)} rows loaded "
          f"({sum(labels)} phishing / {len(labels) - sum(labels)} legit)")

    print("Extracting features ...")
    X = build_feature_matrix(urls)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    print("Training RandomForestClassifier ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("Held-out evaluation:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # Feature importances -> used by the Response Layer for "explanation"
    importances = pipeline.named_steps["rf"].feature_importances_
    importance_map = {
        name: round(float(imp), 4)
        for name, imp in sorted(
            zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
        )
    }
    print("\nTop feature importances:")
    for name, imp in list(importance_map.items())[:8]:
        print(f"   {name:28s} {imp}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipeline, args.out)
    with open("model/feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open("model/feature_importances.json", "w") as f:
        json.dump(importance_map, f, indent=2)
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {args.out}")


if __name__ == "__main__":
    main()
