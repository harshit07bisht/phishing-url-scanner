# Phishing URL Classification on the Cloud (Random Forest)

A 4-layer, cloud-deployable system that classifies a URL as **safe**,
**suspicious**, or **phishing** using lexical/structural URL features and a
Random Forest classifier.

```
┌─────────────────────┐     URL      ┌──────────────────────────┐
│   1. CLIENT LAYER    │ ───────────▶ │ 2. FEATURE EXTRACTION     │
│  browser / mobile /  │              │    LAYER                  │
│  API request         │              │  url -> numeric vector    │
└─────────────────────┘              └──────────────┬────────────┘
          ▲                                          │ feature vector
          │ score, level, explanation                ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  4. RESPONSE LAYER   │ ◀─────────── │ 3. MODEL INFERENCE LAYER  │
│  JSON: score/level/  │   raw proba  │  RandomForestClassifier   │
│  explanation         │              │  (cloud-hosted)           │
└─────────────────────┘              └──────────────────────────┘
```

## Files

| File                          | Layer | Purpose |
|--------------------------------|-------|---------|
| `client/client_example.py`     | 1     | Example caller (stand-in for browser ext / mobile app) |
| `feature_extraction.py`        | 2     | URL → 27-dimensional numeric feature vector |
| `generate_demo_dataset.py`     | -     | Builds a small synthetic labeled dataset for demo training |
| `train_model.py`               | 3     | Trains & saves the RandomForest pipeline |
| `app.py`                       | 3 + 4 | Flask API: loads the model, scores requests, formats the response |
| `model/rf_phishing_model.joblib` | 3   | Trained model artifact (scaler + RandomForest pipeline) |
| `Dockerfile`                   | -     | Container for cloud deployment |
| `requirements.txt`             | -     | Python dependencies |

## 1. Client Layer

Anything that has a URL to check: a browser extension intercepting
navigation, a mobile app scanning a link from SMS/email, or a backend
service validating user-submitted links. It just needs to `POST` JSON to
the API:

```bash
curl -X POST https://<your-api>/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "http://paypal-login.secure-verify.info/webscr"}'
```

See `client/client_example.py` for a runnable stand-in.

## 2. Feature Extraction Layer

`feature_extraction.py` converts a raw URL string into **27 numeric
features**, purely from the URL text (no DNS/WHOIS/HTTP calls, so it's
fast and safe to run on every request inside a serverless function):

- Structural: `url_length`, `hostname_length`, `path_length`, `num_dots`,
  `num_hyphens`, `num_slashes`, `num_subdomains`, `path_depth`, ...
- Security-relevant: `has_ip_address`, `has_https_scheme`,
  `has_https_token_in_domain` (fake "https"/"ssl" text stuffed into the
  domain), `is_shortening_service`, `has_double_slash_in_path`
- Content-based: `suspicious_word_count` (login/verify/secure/confirm/...),
  `digit_ratio`, `url_entropy` (randomness of the string — phishing
  domains are often auto-generated and look "noisier")

This module is shared by both training (`train_model.py`) and serving
(`app.py`), so the training-time and serving-time features are guaranteed
to match — a common source of silent bugs in ML systems ("training/serving
skew").

**Extending it:** if you want stronger signals, add a second, optional
block for enrichment features that *do* need network calls — domain age
(WHOIS), certificate issuer/age (TLS), or blocklist hits (Google Safe
Browsing / PhishTank API). Keep those behind a flag/cache since they add
latency and external dependencies.

## 3. Model Inference Layer

`train_model.py` trains a `Pipeline(StandardScaler -> RandomForestClassifier)`
and saves it with `joblib`. A demo dataset (`generate_demo_dataset.py`) is
included so you can run the whole pipeline immediately, but **it's
synthetic and easily separable — replace `data/urls.csv` with real labeled
data before trusting this for anything real**. Good sources:

- [PhishTank](https://phishtank.org) — verified phishing URL feed
- [OpenPhish](https://openphish.com) — free phishing feed
- UCI ML Repository — "Phishing Websites" dataset
- Kaggle — "Phishing Site URLs" / "Malicious URLs dataset"

Retrain with:
```bash
pip install -r requirements.txt
python generate_demo_dataset.py      # or supply your own data/urls.csv
python train_model.py --n-estimators 300 --max-depth 12
```

`app.py` loads the saved model **once per process** (cheap on warm
invocations; the model load is the only real "cold start" cost) and calls
`model.predict_proba()` per request.

**Why Random Forest here:** it handles the mixed-scale, non-linear feature
set well without heavy tuning, gives you `feature_importances_` for free
(used by the Response Layer's explanation), is robust to outliers/noisy
lexical features, and is cheap enough to serve with sub-50ms latency on a
small CPU instance — no GPU needed.

## 4. Response Layer

`app.py`'s `/predict` endpoint returns:

```json
{
  "url": "http://paypal-login.secure-verify.info/webscr?cmd=confirm",
  "score": 0.9794,
  "level": "phishing",
  "explanation": [
    {"feature": "suspicious_word_count", "value": 6, "importance": 0.0857},
    {"feature": "has_https_scheme", "value": 0, "importance": 0.3212},
    {"feature": "url_length", "value": 57, "importance": 0.1382}
  ],
  "model_version": "rf-v1",
  "latency_ms": 34.5
}
```

- **score**: raw phishing probability (0–1) from the Random Forest.
- **level**: bucketed decision — `safe` / `suspicious` / `phishing` — via
  the thresholds in `THRESHOLDS` in `app.py`. Tune these against your
  validation set: lower `safe_max` if you want to be aggressive about
  flagging, raise it if false positives (blocking legit sites) are costly.
- **explanation**: top contributing features for *this* prediction
  (importance × how unusual the value is vs. the training distribution).
  For production-grade, mathematically rigorous per-prediction
  explanations, swap in `shap.TreeExplainer` — the function signature in
  `build_explanation()` is designed so that's a drop-in change.

## Running locally

```bash
pip install -r requirements.txt
python generate_demo_dataset.py
python train_model.py
python app.py
# in another terminal:
python client/client_example.py
```

## Deploying to the cloud

The API (`app.py`) is a stateless container — deploy it anywhere that
runs containers or Python WSGI apps:

**AWS**
- Container route: push the `Dockerfile` image to ECR, run on **App
  Runner**, **ECS/Fargate**, or **EKS**, behind an ALB or API Gateway
  (HTTP API + VPC Link).
- Serverless route: wrap `app.py` with [Mangum](https://github.com/jordaneremieff/mangum)
  and deploy as a **Lambda** behind **API Gateway**; store the model in
  **S3** and load it at cold start (or bake it into a Lambda container
  image, which is simpler for a ~few-MB RandomForest).
- For a fully managed model host instead of self-serving: train the same
  pipeline and deploy via **SageMaker** (scikit-learn container +
  endpoint), and keep this repo's Flask app only as the feature-extraction
  + response-formatting proxy in front of the SageMaker endpoint.

**GCP**
- Build the `Dockerfile` and deploy to **Cloud Run** (`gcloud run deploy`)
  — scales to zero, pay-per-request, simplest option for this workload.
- Or host the trained model on **Vertex AI Prediction** and keep this
  Flask app as the feature-extraction/response layer in front of it.

**Azure**
- **Azure Functions** (HTTP trigger, Python) for a serverless deploy, or
  **Azure App Service** / **Container Apps** for the containerized route.
- Model can be hosted via **Azure ML managed endpoints** if you want the
  inference layer fully managed.

In every case, keep the layers logically separate even if they're
deployed in the same container to start:
`Client → API Gateway/Load Balancer → (Feature Extraction → Model
Inference → Response Formatting)`. That separation is what lets you later
swap the Random Forest for a different model, add caching, or move
feature extraction to an edge function, without touching the other layers.

## Extending this project

- **Explainability**: swap the heuristic explanation for `shap.TreeExplainer`.
- **Feedback loop**: log `(url, features, score, user_feedback)` to a data
  store and periodically retrain — phishing patterns drift fast.
- **Caching**: cache recent URL verdicts (Redis/DynamoDB/Memorystore) since
  the same phishing links get shared repeatedly in short bursts.
- **Blocklist fast-path**: check known blocklists (PhishTank/Safe Browsing)
  before running the model, to short-circuit obvious cases and save
  inference cost.
- **Monitoring**: track score distribution drift and label delay (once
  ground truth arrives) to catch model decay.
