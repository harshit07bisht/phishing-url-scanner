"""
app.py
-------
LAYER 3: Model Inference Layer  +  LAYER 4: Response Layer

Exposes a REST API that the Client Layer (browser extension, mobile app,
or any API caller) sends a URL to, and that returns a phishing score,
a risk level, and an optional explanation.

Run locally:
    python app.py
    curl -X POST http://localhost:8080/predict \
         -H "Content-Type: application/json" \
         -d '{"url": "http://paypal-login.secure-verify.info/webscr"}'

Cloud deployment (see README.md for full details):
    - AWS: package as a Lambda (via Mangum/Zappa) behind API Gateway, or
      run in a container on ECS/Fargate/App Runner behind an ALB.
    - GCP: deploy this Flask app to Cloud Run (fully managed container).
    - Azure: deploy to Azure Functions (HTTP trigger) or App Service.
The model file (model/rf_phishing_model.joblib) should be loaded from
S3 / GCS / Blob Storage (or baked into the container image) at cold start.
"""

import os
import time
import logging
from typing import Optional

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

from feature_extraction import extract_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishing-classifier")

MODEL_PATH = os.environ.get("MODEL_PATH", "model/rf_phishing_model.joblib")

# Risk-level thresholds on the model's phishing probability [0, 1].
# Tune these against your validation set / business tolerance for
# false positives vs false negatives.
THRESHOLDS = {
    "safe_max": 0.30,        # score <  0.30            -> "safe"
    "suspicious_max": 0.65,  # 0.30 <= score < 0.65      -> "suspicious"
                              # score >= 0.65             -> "phishing"
}

app = Flask(__name__)

_model = None  # loaded lazily / once per process (cold start)


def get_model():
    """Load the trained pipeline once per process (cheap on warm invocations,
    the one-time cost you pay on a cold serverless start)."""
    global _model
    if _model is None:
        t0 = time.time()
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH} in {time.time() - t0:.3f}s")
    return _model


def score_to_level(score: float) -> str:
    if score < THRESHOLDS["safe_max"]:
        return "safe"
    if score < THRESHOLDS["suspicious_max"]:
        return "suspicious"
    return "phishing"


def build_explanation(model, feature_dict: dict, top_k: int = 5) -> list:
    """
    Lightweight, dependency-free explanation: combines the model's global
    feature_importances_ with how unusual this URL's value is (z-score
    against the scaler's fitted mean/std) to rank which features drove
    this particular prediction the most.

    For production-grade per-prediction explanations, swap this for
    shap.TreeExplainer(model.named_steps["rf"]) — the interface
    (list of {feature, value, contribution}) can stay the same.
    """
    rf = model.named_steps["rf"]
    scaler = model.named_steps["scaler"]
    importances = rf.feature_importances_

    values = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=float)
    z_scores = np.abs((values - scaler.mean_) / (scaler.scale_ + 1e-9))

    # Blend global importance with how anomalous the value is for this URL
    contribution = importances * z_scores
    order = np.argsort(-contribution)[:top_k]

    explanation = []
    for i in order:
        explanation.append({
            "feature": FEATURE_NAMES[i],
            "value": feature_dict[FEATURE_NAMES[i]],
            "importance": round(float(importances[i]), 4),
        })
    return explanation


@app.route("/", methods=["GET"])
def index():
    """LAYER 1 (Client Layer), served here for convenience: a simple web
    page where a person can paste a URL and see the scan result. It calls
    the same /predict endpoint below via fetch() — this is just a thin
    browser-based client, not part of the inference/response logic."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body:  {"url": "<string>", "explain": true|false (optional, default true)}
    Response body: {
        "url": "...",
        "score": 0.0-1.0,          # probability the URL is phishing
        "level": "safe|suspicious|phishing",
        "explanation": [ {feature, value, importance}, ... ],  # optional
        "model_version": "...",
        "latency_ms": 12.4
    }
    """
    t0 = time.time()
    payload = request.get_json(silent=True) or {}
    url: Optional[str] = payload.get("url")
    explain = payload.get("explain", True)

    if not url or not isinstance(url, str):
        return jsonify({"error": "Request body must include a non-empty 'url' string."}), 400

    model = get_model()
    feature_dict = extract_features(url)
    vector = np.array([[feature_dict[name] for name in FEATURE_NAMES]])

    proba = model.predict_proba(vector)[0]
    score = float(proba[1])  # probability of class 1 = phishing
    level = score_to_level(score)

    response = {
        "url": url,
        "score": round(score, 4),
        "level": level,
        "model_version": os.environ.get("MODEL_VERSION", "rf-v1"),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
    if explain:
        response["explanation"] = build_explanation(model, feature_dict)

    logger.info(f"url={url!r} score={score:.4f} level={level}")
    return jsonify(response), 200


if __name__ == "__main__":
    get_model()  # warm the model at startup for local testing
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
