# FinTech Risk Scoring Platform

An end-to-end FinTech risk scoring product prototype with:

- Real-time PD scoring
- Policy decision layer (Approve / Review / Reject)
- Explainability output
- Portfolio analytics by region/channel/product
- FastAPI backend + React frontend + Docker support

## Demo Flow

1. Open Applicant Scoring page.
2. Modify applicant inputs.
3. Click `Score Applicant`.
4. Review:
   - PD (probability of default)
   - Decision (Approve / Review / Reject)
   - Explainability reasons
5. Go to Portfolio page.
6. Switch `group by` region/channel/product.
7. Observe risk distribution and portfolio metrics.

## Quick Start (Recommended: Docker)

### 1. Clone the repo

```bash
git clone https://github.com/yourname/risk-scoring-platform.git
cd risk-scoring-platform
```

### 2. Start services

```bash
docker compose up --build
```

If your machine uses the old command:

```bash
docker-compose up --build
```

### 3. Open in browser

- Frontend: `http://localhost:5173`
- Backend API docs: `http://localhost:8000/docs`

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### URLs

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Test API

### Score an applicant

```bash
curl -X POST http://127.0.0.1:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 65000,
    "employment_length": 5,
    "dti": 0.25,
    "utilization": 0.45,
    "delinquencies": 0,
    "history_length": 8,
    "tx_30d_count": 40,
    "refund_rate_30d": 0.02,
    "active_days_30d": 18,
    "channel": "online",
    "region": "NE",
    "product": "pl"
  }'
```

## Environment Variables

The frontend uses `VITE_API_BASE_URL`.

1. Copy `.env.example` to `frontend/.env`.
2. Update the value if your backend is not running on `localhost:8000`.

Example:

```env
VITE_API_BASE_URL=http://localhost:8000/v1
```

## Useful API Endpoints

- `GET /v1/health`
- `GET /v1/model/version`
- `GET /v1/metrics`
- `POST /v1/score`
- `POST /v1/score/batch`
- `POST /v1/explain`
- `GET /v1/portfolio/summary?group_by=region`

## Model Validation / Overfitting Check

Run training with a fixed split:

```bash
python ml/train.py \
  --data ml/data/processed/train.csv \
  --schema backend/app/ml/feature_schema.json \
  --out backend/app/ml/artifacts \
  --version v0.4.0 \
  --seed 42 \
  --target default \
  --fixed_fpr 0.05
```

The script writes:

- `metrics.train / metrics.val / metrics.test` (ROC-AUC, PR-AUC, KS, optional Recall@FPR)
- `overfit_check.roc_auc_gap_train_val`
- `overfit_check.pr_auc_gap_train_val`
- `overfit_check.*_status` (`ok`, `watch`, `high_risk`)

Quick interpretation:

- Train much higher than Val/Test => likely overfitting
- Train/Val/Test close => generalization is acceptable
- All three low => likely underfitting or weak signal

## Project Structure

```text
risk-scoring-platform/
|- backend/            # FastAPI backend
|- frontend/           # React + Vite frontend
|- ml/                 # Training and data scripts
|- scripts/            # Utility scripts
|- docker-compose.yml
`- README.md
```


