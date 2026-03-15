# FinTech Risk Scoring Platform

## Overview

An end-to-end FinTech risk scoring platform prototype that provides real-time probability-of-default (PD) predictions, batch scoring, explainability (SHAP), portfolio analytics, and threshold policy simulation — built with production-style architecture.

**Note:** This project is currently a work-in-progress prototype. Not all features are fully implemented or tested.

## Features

- **Real-time PD Scoring**: API endpoint for scoring individual applicants
- **Batch Scoring**: Process multiple applications in bulk
- **Model Explainability**: SHAP-based explanations for predictions
- **Portfolio Analytics**: Aggregate risk metrics and visualizations
- **Threshold Policy Simulation**: Test different approval thresholds
- **Data Validation**: Input validation and schema enforcement
- **Health Monitoring**: System health checks and metrics

## Architecture

The platform consists of three main components:

- **Backend**: FastAPI-based REST API server
- **Frontend**: Web interface (under development)
- **ML Pipeline**: Model training, evaluation, and artifact management

### Backend Structure

- `app/main.py`: Application entry point
- `app/api/v1/`: API routes and endpoints
- `app/core/`: Core utilities (config, logging, schema loading)
- `app/ml/`: ML artifacts and feature schema
- `app/schemas/`: Pydantic models for data validation

### ML Pipeline

- `ml/train.py`: Model training script
- `ml/evaluate.py`: Model evaluation
- `ml/prepare_realdata.py`: Data preprocessing
- `data/`: Raw and processed datasets

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd risk-scoring-platform
   ```

2. Set up the backend environment:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. (Optional) Use Docker Compose for full setup:
   ```bash
   docker-compose up --build
   ```

## Usage

### Running the Backend

```bash
cd backend
./run.sh
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /health`: Health check
- `GET /metrics`: System metrics
- `POST /v1/score`: Score a single application
- `POST /v1/batch-score`: Batch scoring
- `POST /v1/validate`: Validate application data
- `GET /v1/schema`: Get application schema

### Training the Model

```bash
cd ml
python train.py
```

## Development

### Project Structure

```
risk-scoring-platform/
├── backend/          # FastAPI backend
├── frontend/         # Frontend application
├── ml/              # ML pipeline
├── scripts/         # Utility scripts
├── docker-compose.yml
└── README.md
```


